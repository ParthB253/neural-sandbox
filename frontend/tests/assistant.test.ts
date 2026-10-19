import { test } from "node:test";
import assert from "node:assert/strict";
import { runAgent } from "../src/assistant/runner.ts";
import { complete } from "../src/assistant/openrouter.ts";
import { createTools, expandSchema, type BackendRequest } from "../src/assistant/tools.ts";
import type { Message, ToolCall } from "../src/assistant/types.ts";
import type { ProjectSpec } from "../src/network.ts";

const call = (name: string, args = {}): ToolCall => ({ id: name, type: "function", function: { name, arguments: JSON.stringify(args) } });
const signal = () => new AbortController().signal;
const initial = (): ProjectSpec => ({ network: { id: "network", title: "Test", layers: [], links: [] }, datasets: [], feeds: [] });

function fixture() {
  let project = initial();
  const writes: ProjectSpec[] = [];
  const requests: string[] = [];
  const request: BackendRequest = async <T>(path: string, _method?: string, body?: unknown): Promise<T> => {
    requests.push(path);
    if (path === "/openapi.json") return { components: { schemas: { ProjectSpec: { type: "object" } } } } as T;
    if (path === "/project/analysis") {
      if ((body as ProjectSpec).network.layers.some((layer) => layer.size <= 0)) throw new Error("Size must be positive");
      return { issues: [] } as T;
    }
    return [] as T;
  };
  return {
    host: { request, getProject: () => project, commit: async (next: ProjectSpec, expected: ProjectSpec) => {
      assert.deepEqual(expected, project);
      writes.push(next); project = next; return next;
    } },
    requests, writes,
    change: () => { project = { ...project, network: { ...project.network, title: "User edit" } }; },
  };
}

test("ReAct loop replays tool calls and observations, including recoverable tool errors", async () => {
  const messages: Message[] = [{ role: "user", content: "Edit" }];
  let round = 0;
  const result = await runAgent({
    messages, tools: [], signal: signal(), activity: () => {},
    complete: async (history) => {
      if (round++ === 0) return { role: "assistant", content: null, tool_calls: [call("invalid"), call("get_project")] };
      assert.equal(history[1].tool_calls?.length, 2);
      assert.equal(history[2].tool_call_id, "invalid");
      assert.match(history[2].content!, /Unknown tool/);
      assert.equal(history[3].tool_call_id, "get_project");
      return { role: "assistant", content: "Done" };
    },
    execute: async (tool) => { if (tool.function.name === "invalid") throw new Error("Unknown tool"); return initial(); },
  });
  assert.equal(result, "Done");
});

test("loop stops at its limit and cancellation prevents further tool execution", async () => {
  await assert.rejects(runAgent({ messages: [], tools: [], signal: signal(), activity: () => {},
    complete: async () => ({ role: "assistant", content: null, tool_calls: [call("get_project")] }), execute: async () => ({}),
  }), /12 steps/);
  const controller = new AbortController();
  let executed = false;
  await assert.rejects(runAgent({ messages: [], tools: [], signal: controller.signal, activity: () => {},
    complete: async () => { controller.abort(); return { role: "assistant", content: null, tool_calls: [call("update_project")] }; },
    execute: async () => { executed = true; },
  }), { name: "AbortError" });
  assert.equal(executed, false);
});

test("tool registry requires fresh inspection and validates before committing", async () => {
  const f = fixture();
  const tools = await createTools(f.host, signal());
  const next = initial();
  next.network.layers.push({ id: "hidden", size: 32, activation: "relu", pos: [330, 100] });
  await assert.rejects(tools.execute(call("update_project", { project: next }), signal()), /get_project/);
  await tools.execute(call("get_project"), signal());
  await tools.execute(call("update_project", { project: next }), signal());
  assert.equal(f.writes.length, 1);
  assert.equal(f.requests.at(-1), "/project/analysis");
  f.change();
  await assert.rejects(tools.execute(call("update_project", { project: next }), signal()), /user changed/);
  assert.equal(f.writes.length, 1);
  await tools.execute(call("get_project"), signal());
  const invalid = structuredClone(f.host.getProject());
  invalid.network.layers[0].size = 0;
  await assert.rejects(tools.execute(call("update_project", { project: invalid }), signal()), /positive/);
  assert.equal(f.writes.length, 1);
  await assert.rejects(tools.execute(call("delete_file"), signal()), /Unknown tool/);
  await assert.rejects(tools.execute(call("update_project", { project: { ...initial(), network: { ...initial().network, id: "other" } } }), signal()), /preserve/);
});

test("backend schema expansion preserves title properties and resolves references", () => {
  assert.deepEqual(expandSchema({ type: "object", title: "Project", properties: { title: { type: "string", title: "Title" }, layer: { $ref: "#/components/schemas/Layer" } } }, {
    Layer: { type: "object", properties: { size: { type: "integer", default: 16 } } },
  }), { type: "object", properties: { title: { type: "string" }, layer: { type: "object", properties: { size: { type: "integer" } } } } });
});

test("OpenRouter adapter sends tools every round and preserves reasoning metadata", async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    assert.equal(url, "https://openrouter.ai/api/v1/chat/completions");
    const body = JSON.parse(init!.body as string);
    assert.deepEqual(body.tools, []);
    assert.equal(body.provider.require_parameters, true);
    return Response.json({ choices: [{ finish_reason: "tool_calls", message: { content: null, tool_calls: [call("get_project")], reasoning_details: [{ type: "reasoning.encrypted", data: "opaque" }] } }] });
  };
  try {
    const result = await complete({ apiKey: "test-key", model: "test/model" }, [], [], signal());
    assert.deepEqual(result.reasoning_details, [{ type: "reasoning.encrypted", data: "opaque" }]);
    globalThis.fetch = async () => Response.json({ error: { message: "Invalid API key" } }, { status: 401 });
    await assert.rejects(complete({ apiKey: "test", model: "test" }, [], [], signal()), /Invalid API key/);
  } finally { globalThis.fetch = original; }
});

test("partial edit changes only the requested property, preserving CSV quoting and topology", async () => {
  const f = fixture();
  const tools = await createTools(f.host, signal());
  await tools.execute(call("get_project"), signal());
  await tools.execute(call("edit_project", { edits: [{ op: "replace", path: "/network/title", value: "Updated" }] }), signal());
  assert.deepEqual(f.writes[0], { ...initial(), network: { ...initial().network, title: "Updated" } });
  const count = f.writes.length;
  await assert.rejects(tools.execute(call("update_project", { project: { network: { id: "network", links: [] }, datasets: [], feeds: [] } }), signal()), /Incomplete.*network.layers/);
  assert.equal(f.writes.length, count);
});

test("repeated identical failed edits stop early instead of spending all model rounds", async () => {
  let count = 0;
  await assert.rejects(runAgent({ messages: [], tools: [], signal: signal(), activity: () => {},
    complete: async () => ({ role: "assistant", content: null, tool_calls: [{ ...call("update_project"), id: `call-${++count}` }] }),
    execute: async () => { throw new Error("Incomplete replacement"); },
  }), /same tool failure repeated three times/);
  assert.equal(count, 3);
});
