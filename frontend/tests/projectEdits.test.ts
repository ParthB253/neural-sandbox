import { test } from "node:test";
import assert from "node:assert/strict";
import { applyProjectEdits, rebaseLayout, requireCompleteProject } from "../src/assistant/projectEdits.ts";
import type { ProjectSpec } from "../src/network.ts";

const project = (): ProjectSpec => ({ network: { id: "network", title: "Original", layers: [
  { id: "input", size: 32, size_mode: "auto", activation: "identity", pos: [20, 40] },
], links: [] }, datasets: [{ id: "data", source: { type: "csv", path: "data.csv", delimiter: ",", quotechar: '"' }, pos: [0, 0] }],
feeds: [{ id: "inputs", dataset: "data", layer: "input", role: "input", field: "age", transforms: [] }] });

test("concatenation edits preserve unrelated configuration and do not mutate their base", () => {
  const before = project();
  const next = applyProjectEdits(before, [
    { op: "add", path: "/feeds/0/transforms/-", value: { type: "concatenate", fields: ["age", "occupation"], field: "features" } },
    { op: "replace", path: "/feeds/0/field", value: "features" },
  ]);
  assert.equal(next.feeds[0].field, "features");
  assert.equal(next.feeds[0].transforms.length, 1);
  assert.deepEqual(next.network, before.network);
  assert.deepEqual(next.datasets, before.datasets);
  assert.equal(before.feeds[0].field, "age");
});

test("edits reject unsafe paths, missing values, invalid indexes and identity changes", () => {
  for (const edit of [
    { op: "replace", path: "/network/id", value: "other" },
    { op: "add", path: "/feeds/__proto__/bad", value: true },
    { op: "add", path: "/feeds/constructor/bad", value: true },
    { op: "replace", path: "/network/title" },
    { op: "remove", path: "/network/layers/3" },
    { op: "replace", path: "/feeds/0/transforms/-", value: {} },
    { op: "add", path: "/feeds/0/unknown/field", value: "value" },
  ]) assert.throws(() => applyProjectEdits(project(), [edit]));
});

test("layout-only edits merge while changes to configuration still conflict", () => {
  const before = project();
  const moved = structuredClone(before);
  moved.network.layers[0].pos = [300, 100];
  moved.datasets[0].pos = [90, 30];
  const next = applyProjectEdits(before, [{ op: "replace", path: "/network/layers/0/activation", value: "relu" }]);
  const merged = rebaseLayout(next, before, moved);
  assert.deepEqual(merged.network.layers[0].pos, [300, 100]);
  assert.deepEqual(merged.datasets[0].pos, [90, 30]);
  assert.equal(merged.network.layers[0].activation, "relu");
  moved.network.layers[0].size = 29;
  assert.throws(() => rebaseLayout(next, before, moved), /user changed/);
  assert.throws(() => requireCompleteProject({ network: { id: "network" } } as ProjectSpec, before), /edit_project/);
});
