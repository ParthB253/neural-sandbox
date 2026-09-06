// Mirrors backend/src/nnsandbox/schemas.py -- topology only, no weights.
// The backend is the source of truth; this is just the wire format.

export interface LayerSpec {
  id: string;
  size: number;
  activation: string;
  pos: [number, number]; // canvas position, owned by the UI
}

export interface LinkSpec {
  id: string;
  source: string; // LayerSpec.id
  target: string; // LayerSpec.id
}

export interface NetworkSpec {
  layers: LayerSpec[];
  links: LinkSpec[];
}

const API = "http://localhost:8000";

// The notebook's fixed FFN -- same seed the API uses, so the canvas shows
// something even when the backend isn't running.
export const SAMPLE_NETWORK: NetworkSpec = {
  layers: [
    { id: "in", size: 784, activation: "identity", pos: [0, 0] },
    { id: "h1", size: 16, activation: "relu", pos: [240, 0] },
    { id: "h2", size: 16, activation: "relu", pos: [480, 0] },
    { id: "out", size: 10, activation: "softmax", pos: [720, 0] },
  ],
  links: [
    { id: "in-h1", source: "in", target: "h1" },
    { id: "h1-h2", source: "h1", target: "h2" },
    { id: "h2-out", source: "h2", target: "out" },
  ],
};

export async function fetchNetwork(): Promise<NetworkSpec> {
  const res = await fetch(`${API}/network`);
  if (!res.ok) throw new Error(`GET /network -> ${res.status}`);
  return res.json();
}
