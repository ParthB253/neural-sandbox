import type { RunTrace } from "./diagnostics.ts";
import type { Connection, Message, Tool, ToolCall } from "./types.ts";

export async function complete(
  connection: Connection,
  messages: Message[],
  tools: Tool[],
  signal: AbortSignal,
  trace?: RunTrace,
): Promise<Message> {
  const body = {
      model: connection.model,
      messages,
      tools,
      tool_choice: "auto",
      parallel_tool_calls: false,
      provider: { require_parameters: true },
      stream: false,
  };
  const event = trace?.start("model", "OpenRouter completion", body);
  let diagnosticResponse: unknown;
  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      signal: AbortSignal.any([signal, AbortSignal.timeout(90_000)]),
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${connection.apiKey}`,
        "X-OpenRouter-Title": "NeuralSandbox",
      },
      body: JSON.stringify(body),
    });
    const payload = await response.json().catch(() => null);
    diagnosticResponse = { status: response.status, requestId: response.headers.get("x-request-id"), body: payload };
    if (!response.ok || payload?.error) {
      throw new Error(payload?.error?.message ?? `OpenRouter request failed (${response.status}).`);
    }

    const choice = payload?.choices?.[0];
    const message = choice?.message;
    if (choice?.finish_reason === "length") throw new Error("The model reached its response limit. Try a smaller change.");
    if (!message || (typeof message.content !== "string" && !message.tool_calls?.length)) {
      throw new Error("The model returned no response. Try again or choose another tool-capable model.");
    }
    if (message.tool_calls && (!Array.isArray(message.tool_calls) || message.tool_calls.some((call: ToolCall) =>
      !call || typeof call.id !== "string" || call.type !== "function" ||
      typeof call.function?.name !== "string" || typeof call.function?.arguments !== "string"))) {
      throw new Error("The model returned an invalid tool call.");
    }
    event?.done(diagnosticResponse);
    return {
      role: "assistant",
      content: message.content ?? null,
      ...(message.tool_calls?.length ? { tool_calls: message.tool_calls } : {}),
      ...(message.reasoning_details ? { reasoning_details: message.reasoning_details } : {}),
    };
  } catch (error) {
    event?.fail(error, signal.aborted, diagnosticResponse);
    throw error;
  }
}
