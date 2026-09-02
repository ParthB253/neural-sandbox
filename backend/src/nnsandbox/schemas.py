"""Wire format for a network's topology -- what the frontend renders and edits.

Deliberately *not* the training state: no weight matrices live here, just the
shape of the graph. Weights, activations and gradients get their own
(summarised) endpoints once there's something to stream.
"""
from pydantic import BaseModel, Field


class LayerSpec(BaseModel):
    id: str
    size: int = Field(gt=0)
    # Name only; the Python side maps it to an (activation, d_activation) pair.
    activation: str = "relu"
    # Canvas position, owned by the UI. The math never looks at it.
    pos: tuple[float, float] = (0.0, 0.0)


class LinkSpec(BaseModel):
    id: str
    source: str  # LayerSpec.id
    target: str  # LayerSpec.id


class NetworkSpec(BaseModel):
    layers: list[LayerSpec] = []
    links: list[LinkSpec] = []
