import { test } from "node:test";
import assert from "node:assert/strict";
import {
  clearCanvas,
  computationKey,
  connectionError,
  connections,
  emptyProject,
  nextPosition,
  removeItems,
  type ProjectSpec,
} from "../src/network.ts";

const project: ProjectSpec = {
  network: {
    id: "3b813f54-c197-4fc4-a079-66e2b5868149",
    title: "Classifier",
    layers: [
      { id: "a", size: 4, activation: "relu", pos: [330, 0] },
      { id: "b", size: 2, activation: "softmax", pos: [660, 0] },
    ],
    links: [{ id: "link", source: "a", target: "b" }],
  },
  datasets: [
    {
      id: "data",
      source: { type: "csv", path: "/files/data.csv" },
      transforms: [],
      pos: [0, 0],
    },
  ],
  feeds: [
    { id: "feed", dataset: "data", layer: "a", field: "value", role: "input" },
  ],
};

test("both connection kinds participate in validation", () => {
  assert.equal(connections(project).length, 2);
  assert.match(connectionError(project, "data", "a")!, /already connected/);
  assert.match(connectionError(project, "b", "a")!, /cycle/);
  assert.match(connectionError(project, "a", "a")!, /itself/);
  assert.match(connectionError(project, "a", "data")!, /end at a layer/);
  assert.equal(connectionError(project, "data", "b"), undefined);
});

test("deleting a layer removes incoming feeds and outgoing links atomically", () => {
  const next = removeItems(project, ["a"]);
  assert.deepEqual(
    next.network.layers.map((item) => item.id),
    ["b"],
  );
  assert.equal(next.datasets.length, 1);
  assert.deepEqual(connections(next), []);
  assert.equal(
    project.network.layers.length,
    2,
    "original project remains unchanged",
  );
  assert.equal(next.network.id, project.network.id);
  assert.equal(next.network.title, project.network.title);
});

test("new networks start empty with distinct identities, while clearing retains identity", () => {
  const first = emptyProject();
  const second = emptyProject();
  assert.notEqual(first.network.id, second.network.id);
  assert.equal(first.network.title, "Untitled network");
  assert.deepEqual(first.network.layers, []);
  assert.deepEqual(first.datasets, []);
  const cleared = clearCanvas(project);
  assert.equal(cleared.network.id, project.network.id);
  assert.equal(cleared.network.title, project.network.title);
  assert.deepEqual(connections(cleared), []);
  assert.deepEqual(cleared.network.layers, []);
  assert.deepEqual(cleared.datasets, []);
});

test("deleting a source or a connection preserves unrelated topology", () => {
  const next = removeItems(project, ["data"]);
  assert.equal(next.datasets.length, 0);
  assert.equal(next.feeds.length, 0);
  assert.equal(next.network.links.length, 1);
  const disconnected = removeItems(project, [], ["feed", "link"]);
  assert.deepEqual(connections(disconnected), []);
  assert.equal(disconnected.network.layers.length, 2);
  assert.equal(disconnected.datasets.length, 1);
});

test("branches are placed right of their parent without overlapping siblings", () => {
  const [x, y] = nextPosition(project, "a");
  assert.equal(x, 660);
  assert.ok(y >= 180);
});

test("result identity follows computation changes but ignores layout and title", () => {
  const next = structuredClone(project);
  next.network.title = "Renamed";
  next.network.layers[0].pos = [1200, 200];
  next.datasets[0].pos = [600, 100];
  next.network.layers[0].size_mode = "manual";
  assert.equal(computationKey(next), computationKey(project));
  next.network.layers[0].size_mode = "auto";
  assert.notEqual(computationKey(next), computationKey(project));
  const transformed = structuredClone(project);
  transformed.datasets[0].transforms.push({ type: "flatten", field: "value" });
  assert.notEqual(computationKey(transformed), computationKey(project));
});
