// Topology and data configuration only. Python owns all computation.
export interface LayerSpec {
  id: string;
  size: number;
  activation: string;
  pos: [number, number];
}
export interface LinkSpec { id: string; source: string; target: string }
export interface NetworkSpec { layers: LayerSpec[]; links: LinkSpec[] }
export type TransformSpec =
  | { type: 'flatten'; field: string }
  | { type: 'cast'; field: string; dtype: string }
  | { type: 'scale'; field: string; divisor: number }
  | { type: 'one_hot'; field: string; categories?: (string | number | boolean | null)[] | null }
  | { type: 'select'; fields: string[] };
export interface DatasetSpec {
  id: string;
  source: { type: 'csv' | 'json' | 'npy' | 'idx' | 'image_folder'; path: string; delimiter?: string; quotechar?: string; extensions?: string[]; mode?: string };
  transforms: TransformSpec[];
  pos: [number, number];
}
export interface FeedSpec { id: string; dataset: string; layer: string; field: string; role: 'input' | 'target' }
export interface ProjectSpec { network: NetworkSpec; datasets: DatasetSpec[]; feeds: FeedSpec[] }
export interface StoredFile { id: string; name: string; size: number; path: string }

const API = import.meta.env?.VITE_API_URL ?? 'http://localhost:8000';
export async function request<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : body instanceof FormData ? body : JSON.stringify(body),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    const detail = error.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map((item: { msg: string }) => item.msg).join('; ') : `Request failed (${res.status})`);
  }
  return res.status === 204 ? undefined as T : res.json();
}
export const emptyProject = (): ProjectSpec => ({ network: { layers: [], links: [] }, datasets: [], feeds: [] });
export function connections(project: ProjectSpec): LinkSpec[] {
  return [...project.network.links, ...project.feeds.map(feed => ({ id: feed.id, source: feed.dataset, target: feed.layer }))];
}
export function connectionError(project: ProjectSpec, source: string, target: string): string | undefined {
  const edges = connections(project);
  if (source === target) return 'A node cannot connect to itself.';
  if (!project.network.layers.some(layer => layer.id === target)) return 'Connections must end at a layer.';
  if (![...project.network.layers, ...project.datasets].some(node => node.id === source)) return 'Choose a source node.';
  if (edges.some(edge => edge.source === source && edge.target === target)) return 'These nodes are already connected.';
  const pending = [target];
  const seen = new Set<string>();
  while (pending.length) {
    const id = pending.pop()!;
    if (id === source) return 'This connection would create a cycle.';
    if (seen.has(id)) continue;
    seen.add(id);
    pending.push(...edges.filter(edge => edge.source === id).map(edge => edge.target));
  }
}
export function removeItems(project: ProjectSpec, nodes: string[], edges: string[] = []): ProjectSpec {
  return {
    network: {
      layers: project.network.layers.filter(layer => !nodes.includes(layer.id)),
      links: project.network.links.filter(link => !edges.includes(link.id) && !nodes.includes(link.source) && !nodes.includes(link.target)),
    },
    datasets: project.datasets.filter(dataset => !nodes.includes(dataset.id)),
    feeds: project.feeds.filter(feed => !edges.includes(feed.id) && !nodes.includes(feed.dataset) && !nodes.includes(feed.layer)),
  };
}
export function nextPosition(project: ProjectSpec, parent?: string): [number, number] {
  const nodes = [...project.network.layers, ...project.datasets];
  const anchor = nodes.find(node => node.id === parent);
  const x = anchor ? anchor.pos[0] + 330 : nodes.length ? Math.min(...nodes.map(node => node.pos[0])) : 80;
  let y = anchor?.pos[1] ?? 100;
  while (nodes.some(node => Math.abs(node.pos[0] - x) < 240 && Math.abs(node.pos[1] - y) < 180)) y += 190;
  return [x, y];
}
