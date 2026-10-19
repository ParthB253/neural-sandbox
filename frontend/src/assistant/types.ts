export interface ToolCall {
  id: string;
  type: "function";
  function: { name: string; arguments: string };
}

export interface Message {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  // OpenRouter requires these to be replayed for some reasoning models.
  reasoning_details?: unknown[];
}

export interface Tool {
  type: "function";
  function: { name: string; description: string; parameters: Record<string, unknown> };
}

export interface Connection {
  apiKey: string;
  model: string;
}

export interface Activity {
  id: string;
  label: string;
  state: "running" | "done" | "error";
}

export const messageError = (error: unknown): string =>
  error instanceof Error ? error.message : "Something went wrong. Please try again.";
