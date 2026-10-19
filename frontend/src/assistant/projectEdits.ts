import type { ProjectSpec } from "../network.ts";

export function canonical(value: unknown): string {
  return JSON.stringify(value, (_key, item: unknown) => item && typeof item === "object" && !Array.isArray(item)
    ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
}

export function applyProjectEdits(project: ProjectSpec, edits: unknown): ProjectSpec {
  if (!Array.isArray(edits) || !edits.length || edits.length > 100) throw new Error("Supply between 1 and 100 edits.");
  const next = structuredClone(project);
  for (const edit of edits) {
    if (!edit || !["add", "replace", "remove"].includes(edit.op) || typeof edit.path !== "string") throw new Error("Each edit needs an op and JSON pointer path.");
    const parts: string[] = edit.path.split("/").slice(1).map((part: string) => part.replaceAll("~1", "/").replaceAll("~0", "~"));
    if (!edit.path.startsWith("/") || !parts.length || parts.some((part) => ["__proto__", "prototype", "constructor", ""].includes(part))) throw new Error("Invalid edit path.");
    if (!["network", "datasets", "feeds", "training"].includes(parts[0]) ||
      (parts[0] === "network" && !["title", "layers", "links"].includes(parts[1]))) throw new Error("Edit paths must target project configuration and preserve network identity.");
    if (edit.op !== "remove" && !Object.hasOwn(edit, "value")) throw new Error("Add and replace edits require a value.");

    let parent: unknown = next;
    for (const part of parts.slice(0, -1)) {
      if (!parent || typeof parent !== "object" || !Object.hasOwn(parent, part)) throw new Error(`Path does not exist: ${edit.path}`);
      parent = (parent as Record<string, unknown>)[part];
    }
    const key = parts.at(-1)!;
    if (Array.isArray(parent)) {
      const index = key === "-" && edit.op === "add" ? parent.length : /^(0|[1-9]\d*)$/.test(key) ? Number(key) : -1;
      if (index < 0 || index > parent.length || (edit.op !== "add" && index === parent.length)) throw new Error(`Invalid array index: ${edit.path}`);
      if (edit.op === "add") parent.splice(index, 0, structuredClone(edit.value));
      else if (edit.op === "remove") parent.splice(index, 1);
      else parent[index] = structuredClone(edit.value);
    } else {
      if (!parent || typeof parent !== "object") throw new Error(`Path does not exist: ${edit.path}`);
      if (edit.op !== "add" && !Object.hasOwn(parent, key)) throw new Error(`Path does not exist: ${edit.path}`);
      if (edit.op === "remove") delete (parent as Record<string, unknown>)[key];
      else (parent as Record<string, unknown>)[key] = structuredClone(edit.value);
    }
  }
  return next;
}

export function requireCompleteProject(next: ProjectSpec, inspected: ProjectSpec) {
  const missing = ["network", "datasets", "feeds", ...(inspected.training ? ["training"] : [])].filter((key) => !Object.hasOwn(next, key));
  if (next.network) {
    missing.push(...["id", "title", "layers", "links"].filter((key) => !Object.hasOwn(next.network, key)).map((key) => `network.${key}`));
  }
  if (next.training && inspected.training) {
    missing.push(...Object.keys(inspected.training).filter((key) => !Object.hasOwn(next.training!, key)).map((key) => `training.${key}`));
  }
  if (missing.length) throw new Error(`Incomplete project replacement: missing ${missing.join(", ")}. Use edit_project for partial edits, or supply the full project from get_project.`);
}

function withoutPositions(project: ProjectSpec): ProjectSpec {
  const copy = structuredClone(project);
  for (const node of [...copy.network.layers, ...copy.datasets]) node.pos = [0, 0];
  return copy;
}

// Harmless canvas drags should not invalidate a configuration edit.
export function rebaseLayout(next: ProjectSpec, inspected: ProjectSpec, current: ProjectSpec): ProjectSpec {
  if (canonical(withoutPositions(inspected)) !== canonical(withoutPositions(current))) {
    throw new Error("The user changed the canvas configuration. Call get_project and rebase your edit on their changes.");
  }
  const rebased = structuredClone(next);
  for (const node of [...rebased.network.layers, ...rebased.datasets]) {
    const before = [...inspected.network.layers, ...inspected.datasets].find((item) => item.id === node.id);
    const latest = [...current.network.layers, ...current.datasets].find((item) => item.id === node.id);
    if (!before || !latest || canonical(latest.pos) === canonical(before.pos)) continue;
    if (canonical(node.pos) !== canonical(before.pos)) throw new Error("The user moved a node you are repositioning. Call get_project and rebase your edit.");
    node.pos = [...latest.pos];
  }
  return rebased;
}
