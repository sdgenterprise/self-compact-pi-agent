import { randomUUID } from "node:crypto";
import {
	compact,
	findCutPoint,
	sessionEntryToContextMessages,
	SettingsManager,
	type CompactionEntry,
	type ExtensionContext,
	type SessionBeforeCompactEvent,
	type SessionEntry,
} from "@earendil-works/pi-coding-agent";
import { createAssistantMessageEventStream, type Context } from "@earendil-works/pi-ai";
import type { LoadedPrompt } from "./prompts.ts";

/**
 * Mirrors Pi's prepareCompaction(): true when a compaction of this branch would have at least one message to
 * summarize, false when Pi would fail with "Nothing to compact" (everything fits inside keepRecentTokens, or the
 * last entry is already a compaction).
 */
export function hasCompactionMaterial(entries: SessionEntry[], keepRecentTokens: number): boolean {
	if (entries.length === 0 || entries[entries.length - 1]!.type === "compaction") return false;
	let previous = -1;
	for (let i = entries.length - 1; i >= 0; i--) {
		if (entries[i]!.type === "compaction") {
			previous = i;
			break;
		}
	}
	let boundaryStart = 0;
	if (previous >= 0) {
		const firstKept = entries.findIndex((entry) => entry.id === (entries[previous] as CompactionEntry).firstKeptEntryId);
		boundaryStart = firstKept >= 0 ? firstKept : previous + 1;
	}
	const cut = findCutPoint(entries, boundaryStart, entries.length, keepRecentTokens);
	const historyEnd = cut.isSplitTurn ? cut.turnStartIndex : cut.firstKeptEntryIndex;
	for (let i = boundaryStart; i < historyEnd; i++) {
		const entry = entries[i]!;
		if (entry.type !== "compaction" && sessionEntryToContextMessages(entry).length > 0) return true;
	}
	if (cut.isSplitTurn) {
		for (let i = cut.turnStartIndex; i < cut.firstKeptEntryIndex; i++) {
			if (sessionEntryToContextMessages(entries[i]!).length > 0) return true;
		}
	}
	return false;
}

/** Pi's retained recent history for this working directory (global settings merged with the project's). */
export function keepRecentTokens(cwd: string): number {
	return SettingsManager.create(cwd).getCompactionKeepRecentTokens();
}

function summaryInstructions(event: SessionBeforeCompactEvent, prompt: LoadedPrompt): string {
	return [
		"Summarize the supplied historical data. Do not continue the task, simulate tools, or claim actions without tool-result evidence. Keep pending actions pending.",
		prompt.text,
		event.preparation.isSplitTurn ? "This is a split turn. Summarize only the supplied history or turn prefix; the recent suffix remains available." : "",
		event.customInstructions ? `Additional summarization instructions from the operator: ${event.customInstructions}` : "",
	].filter(Boolean).join("\n\n");
}

/**
 * Keep Pi's serialized history and replace its instruction tail with ours. Pi's summarization
 * requests come in two wrappers (the exact base prompts change between Pi versions, so parse the
 * wrapper instead of reconstructing the input byte-for-byte):
 *   history:     <conversation>\n…\n</conversation>\n\n[<previous-summary>\n…\n</previous-summary>\n\n]<pi base prompt>
 *   turn prefix: # Conversation\n…\n\n# Instructions\n<pi base prompt>
 */
function replaceInstructions(context: Context, instructions: string, budgetChars: number) {
	let truncated = false;
	const messages = context.messages.map(message => {
		if (message.role !== "user") return message;
		const content = typeof message.content === "string" ? [{ type: "text" as const, text: message.content }] : message.content;
		return {
			...message,
			content: content.map(block => {
				if (block.type !== "text") return block;
				const text = block.text;
				let head: string;
				if (text.startsWith("<conversation>\n")) {
					let end = text.indexOf("\n</conversation>\n\n");
					if (end < 0) throw new Error("Unrecognized Pi summary input; cannot replace instructions safely.");
					end += "\n</conversation>\n\n".length;
					if (text.slice(end).startsWith("<previous-summary>\n")) {
						const prevEnd = text.indexOf("\n</previous-summary>\n\n", end);
						if (prevEnd < 0) throw new Error("Unrecognized Pi summary input; cannot replace instructions safely.");
						end = prevEnd + "\n</previous-summary>\n\n".length;
					}
					head = text.slice(0, end);
				} else if (text.startsWith("# Conversation\n")) {
					const marker = "\n\n# Instructions\n";
					const idx = text.indexOf(marker);
					if (idx < 0) throw new Error("Unrecognized Pi summary input; cannot replace instructions safely.");
					head = text.slice(0, idx + marker.length);
				} else {
					throw new Error("Unrecognized Pi summary input; cannot replace instructions safely.");
				}
				if (head.length > budgetChars) {
					head = `[earlier conversation truncated to fit summary budget]\n${head.slice(-budgetChars)}`;
					truncated = true;
				}
				return { ...block, text: `${head}${instructions}` };
			}),
		};
	});
	return { messages, truncated };
}

/** Pi owns split turns, summary updates, file tracking and configured transport retries. */
export async function generateSummary(event: SessionBeforeCompactEvent, ctx: ExtensionContext, system: LoadedPrompt, instructions: LoadedPrompt) {
	if (!ctx.model) throw new Error("No model available for compaction.");
	const userInstructions = summaryInstructions(event, instructions);
	let truncatedInput = false;
	const result = await compact(
		event.preparation, ctx.model, undefined, undefined, event.customInstructions, event.signal, ctx.thinkingLevel,
		async (model, context, options) => {
			const maxTokens = Math.min(options?.maxTokens ?? 8192, model.maxTokens || 8192, 8192);
			const budgetChars = Math.max(8000, (model.contextWindow - maxTokens - 2000) * 4 - system.text.length - userInstructions.length);
			const { messages, truncated } = replaceInstructions(context, userInstructions, budgetChars);
			truncatedInput ||= truncated;
			const response = await ctx.modelRegistry.complete(model, { ...context, systemPrompt: system.text, messages }, {
				...options, maxTokens, signal: event.signal, cacheRetention: "none", sessionId: randomUUID(),
				...(model.api === "openai-completions" && model.reasoning ? { reasoningEffort: "low" as const } : {}),
			});
			if (event.signal.aborted || response.stopReason === "aborted") throw new Error("Compaction summary cancelled.");
			if (response.stopReason !== "error" && !response.content.some(block => block.type === "text" && block.text.trim())) throw new Error("Summary response was empty.");
			const stream = createAssistantMessageEventStream();
			stream.end(response);
			return stream;
		},
		undefined, SettingsManager.create(ctx.cwd).getRetrySettings(),
	);
	if (event.signal.aborted) throw new Error("Compaction summary cancelled.");
	return { result, truncatedInput };
}
