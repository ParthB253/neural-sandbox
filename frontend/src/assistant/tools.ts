import { applyProjectEdits, requireCompleteProject, rebaseLayout } from "./projectEdits.ts";
import type { ProjectSpec } from "../network.ts";
import type { Tool, ToolCall } from "./types.ts";

type Schema = Record<string, unknown>;
export type BackendRequest = <T>(path: string, method?: string, body?: unknown, signal?: AbortSignal) => Promise<T>;

// Expand the backend's wire schema rather than maintaining another ProjectSpec schema.
export function expandSchema(value: unknown, schemas: Record<string, Schema>): unknown {
  if (Array.isArray(value)) return value.map((item) => expandSchema(item, schemas));
  if (!value || typeof value !== "object") return value;
  const object = value as Schema;
  if (typeof object.$ref === "string") {
    const name = object.$ref.split("/").pop()!;
    if (!schemas[name]) throw new Error(`Missing backend schema: ${name}`);
    return expandSchema(schemas[name], schemas);
  }
  return Object.fromEntries(Object.entries(object)
    .filter(([key]) => !["title", "default", "discriminator"].includes(key))
    .map(([key, item]) => [key, key === "properties"
      ? Object.fromEntries(Object.entries(item as Schema).map(([name, property]) => [name, expandSchema(property, schemas)]))
      : expandSchema(item, schemas)]));
}

export async function createTools(host: {
  getProject: () => ProjectSpec;
  commit: (next: ProjectSpec, expected: ProjectSpec) => Promise<ProjectSpec>;
  request: BackendRequest;
}, signal: AbortSignal): Promise<{ definitions: Tool[]; execute: (call: ToolCall, signal: AbortSignal) => Promise<unknown> }> {
  const api = await host.request<{ components: { schemas: Record<string, Schema> } }>("/openapi.json", "GET", undefined, signal);
  const projectSchema = expandSchema(api.components.schemas.ProjectSpec, api.components.schemas) as Schema;
  projectSchema.required = ["network", "datasets", "feeds", "training"];
  const networkSchema = (projectSchema.properties as Record<string, Schema> | undefined)?.network;
  if (networkSchema) networkSchema.required = ["id", "title", "layers", "links"];
  const empty: Schema = { type: "object", properties: {}, additionalProperties: false };
  const definition = (name: string, description: string, parameters = empty): Tool => ({
    type: "function", function: { name, description, parameters },
  });
  let inspected: ProjectSpec | undefined;
  return {
    definitions: [
      definition("get_project", "Read the latest canvas draft before editing. Returns all layers, links, datasets, feeds and training settings."),
      definition("list_files", "List uploaded files that can be added as data sources."),
      definition("get_node_types", "Discover supported node types and configuration."),
      definition("analyze_project", "Analyze the current canvas: resolved dataset fields, shapes and issues."),
      definition("edit_project", "Prefer this for small changes. Apply JSON Patch operations to the last inspected project; preserve everything else. Paths use array indexes from get_project, e.g. /network/layers/0/size_mode or /feeds/1/transforms/-. Use add to append a transform; replace to change a field. Analyze after editing.", {
        type: "object", properties: { edits: { type: "array", minItems: 1, maxItems: 100, items: {
          type: "object", properties: { op: { type: "string", enum: ["add", "replace", "remove"] }, path: { type: "string" }, value: {} },
          required: ["op", "path"], additionalProperties: false,
        } } }, required: ["edits"], additionalProperties: false,
      }),
      definition("update_project", "Replace the complete project atomically after get_project. Preserve unrelated configuration. Returns the saved project and analysis.", {
        type: "object", properties: { project: projectSchema }, required: ["project"], additionalProperties: false,
      }),
    ],
    async execute(call, toolSignal) {
      toolSignal.throwIfAborted();
      const args: unknown = JSON.parse(call.function.arguments);
      if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Tool arguments must be an object.");
      switch (call.function.name) {
        case "get_project":
          inspected = structuredClone(host.getProject());
          return inspected;
        case "list_files":
          return host.request("/files", "GET", undefined, toolSignal);
        case "get_node_types":
          return host.request("/node-types", "GET", undefined, toolSignal);
        case "analyze_project":
          return host.request("/project/analysis", "POST", host.getProject(), toolSignal);
        case "edit_project":
        case "update_project": {
          if (!inspected) throw new Error("Call get_project before making edits.");
          const candidate = call.function.name === "edit_project"
            ? applyProjectEdits(inspected, (args as { edits?: unknown }).edits)
            : (args as { project?: ProjectSpec }).project;
          if (!candidate || typeof candidate !== "object") throw new Error("Supply a complete project, or use edit_project for partial edits.");
          requireCompleteProject(candidate, inspected);
          if (candidate.network.id !== inspected.network.id) throw new Error("Edits must preserve the current network ID.");
          const next = rebaseLayout(candidate, inspected, host.getProject());
          // Analysis validates the entire Pydantic document before any write.
          const analysis = await host.request("/project/analysis", "POST", next, toolSignal);
          toolSignal.throwIfAborted();
          const expected = structuredClone(host.getProject());
          const rebased = rebaseLayout(candidate, inspected, expected);
          const saved = await host.commit(rebased, expected);
          inspected = structuredClone(saved);
          return { project: saved, analysis };
        }
        default:
          throw new Error(`Unknown tool: ${call.function.name}`);
      }
    },
  };
}
