import { test } from "node:test";
import assert from "node:assert/strict";
import { RunLogStore, serializeDetail } from "../src/assistant/diagnostics.ts";
import { runAgent } from "../src/assistant/runner.ts";
import { complete } from "../src/assistant/openrouter.ts";

const metadata = (id = "run") => ({ id, networkId: "network", networkTitle: "Test", model: "test/model", prompt: "Add a layer" });
function storage() {
  let data: string | null = null;
  return { getItem: () => data, setItem: (_key: string, value: string) => { data = value; } };
}

test("logs capture tool arguments, observations and errors without breaking the loop", async () => {
  const store = new RunLogStore(storage());
  const trace = store.start(metadata());
  let round = 0;
  await runAgent({ messages: [], tools: [], signal: new AbortController().signal, trace, activity: () => {},
    complete: async () => round++ ? { role: "assistant", content: "Could not edit" } : {
      role: "assistant", content: null, tool_calls: [{ id: "call-1", type: "function", function: { name: "update_project", arguments: '{"size":32}' } }],
    },
    execute: async () => { throw new Error("Invalid layer", { cause: { status: 422, detail: [{ loc: ["body", "network", "layers"], msg: "Missing layer" }] } }); },
  });
  trace.finish("done", "Could not edit");
  const run = store.getSnapshot()[0];
  assert.equal(run.state, "done");
  assert.equal(run.events[0].state, "error");
  assert.equal(JSON.parse(run.events[0].input).arguments.size, 32);
  assert.match(run.events[0].error!, /422/);
  assert.match(run.events[0].error!, /Missing layer/);
  assert.ok(run.events[0].durationMs! >= 0);
});

test("redaction removes credentials and private reasoning before storage and export", () => {
  const value = serializeDetail({ authorization: "Bearer another-key", api_key: "other-key",
    reasoning_details: [{ text: "private" }], nested: { password: "secret" }, error: new Error("Failed: sk-example-key") }, "sk-example-key");
  assert.doesNotMatch(value, /another-key|other-key|private|sk-example-key|"secret"/);
  assert.match(value, /redacted/);
  const saved = storage();
  const store = new RunLogStore(saved);
  const trace = store.start({ ...metadata(), prompt: "Inspect sk-example-key" }, "sk-example-key");
  trace.finish("error", new Error("Rejected sk-example-key"));
  assert.doesNotMatch(saved.getItem()!, /sk-example-key/);
});

test("reload retains history and marks unfinished requests interrupted", () => {
  const saved = storage();
  const store = new RunLogStore(saved);
  store.start(metadata()).start("tool", "get_project", {});
  const restored = new RunLogStore(saved).getSnapshot()[0];
  assert.equal(restored.state, "interrupted");
  assert.equal(restored.events[0].state, "interrupted");
});

test("logging remains usable if browser storage fails and retains bounded history", async () => {
  const store = new RunLogStore({ getItem: () => "[]", setItem: () => { throw new Error("Quota exceeded"); } });
  const oldSnapshot = store.getSnapshot();
  const trace = store.start(metadata());
  await assert.rejects(trace.step("backend", "GET /files", {}, async () => { throw new Error("Unavailable"); }), /Unavailable/);
  assert.equal(oldSnapshot.length, 0);
  assert.match(store.warning, /storage/);
  trace.finish("error", "Unavailable");
  for (let index = 0; index < 25; index++) store.start(metadata(`run-${index}`)).finish("done", "OK");
  assert.equal(store.getSnapshot().length, 20);
  const active = store.start(metadata("active"));
  store.clearFinished();
  assert.equal(store.getSnapshot().length, 1);
  active.finish("cancelled", "Stopped");
  assert.equal(store.getSnapshot()[0].state, "cancelled");
});

test("large details are explicitly truncated and a large prompt cannot break run creation", () => {
  assert.match(serializeDetail("x".repeat(20_000)), /truncated/);
  const store = new RunLogStore();
  store.start({ ...metadata(), prompt: "x".repeat(50_000) });
  assert.ok(store.getSnapshot()[0].prompt.length <= 8000);
});

test("OpenRouter errors retain HTTP status, request ID and provider response in logs", async () => {
  const original = globalThis.fetch;
  const store = new RunLogStore();
  const trace = store.start(metadata(), "test-key");
  globalThis.fetch = async () => Response.json({ error: { message: "Invalid tool schema", metadata: { provider_name: "Test" } } }, {
    status: 400, headers: { "x-request-id": "provider-request" },
  });
  try {
    await assert.rejects(complete({ apiKey: "test-key", model: "test/model" }, [], [], new AbortController().signal, trace), /Invalid tool schema/);
    const event = store.getSnapshot()[0].events[0];
    assert.equal(event.state, "error");
    assert.equal(JSON.parse(event.output!).status, 400);
    assert.match(event.output!, /provider-request/);
    assert.doesNotMatch(JSON.stringify(store.getSnapshot()), /test-key/);
  } finally { globalThis.fetch = original; }
});

test("invisible BOM characters in CSV settings are visible in diagnostic details", () => {
  const detail = serializeDetail({ quotechar: "\uFEFF" });
  assert.ok(detail.includes("\\uFEFF"));
  assert.equal(JSON.parse(detail).quotechar, "\uFEFF");
});
