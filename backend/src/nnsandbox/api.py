"""FastAPI app. For now it just holds one in-memory NetworkSpec and lets the
frontend read and replace it. No persistence, no bridge to core.Network yet --
that (spec -> real Layers/Links, plus a training stream) is the next step.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .schemas import LayerSpec, LinkSpec, NetworkSpec

app = FastAPI(title="nn-sandbox")

# Vite (5173) and CRA-style (3000) dev servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared draft, resets on restart. Seeded with the notebook's FFN.
_network = NetworkSpec(
    layers=[
        LayerSpec(id="in", size=784, activation="identity", pos=(0, 0)),
        LayerSpec(id="h1", size=16, activation="relu", pos=(240, 0)),
        LayerSpec(id="h2", size=16, activation="relu", pos=(480, 0)),
        LayerSpec(id="out", size=10, activation="softmax", pos=(720, 0)),
    ],
    links=[
        LinkSpec(id="in-h1", source="in", target="h1"),
        LinkSpec(id="h1-h2", source="h1", target="h2"),
        LinkSpec(id="h2-out", source="h2", target="out"),
    ],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/network")
def get_network() -> NetworkSpec:
    return _network


@app.put("/network")
def put_network(spec: NetworkSpec) -> NetworkSpec:
    global _network
    _network = spec
    return _network
