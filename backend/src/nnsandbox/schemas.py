"""Wire format for a network's topology -- what the frontend renders and edits.

Deliberately *not* the training state: no weight matrices live here, just the
shape of the graph. Weights, activations and gradients get their own
(summarised) endpoints once there's something to stream.
"""
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class LayerSpec(BaseModel):
    id: str = Field(min_length=1)
    size: int = Field(gt=0)
    # Name only; the Python side maps it to an (activation, d_activation) pair.
    activation: str = "relu"
    # Canvas position, owned by the UI. The math never looks at it.
    pos: tuple[float, float] = (0.0, 0.0)


class LinkSpec(BaseModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)  # LayerSpec.id
    target: str = Field(min_length=1)  # LayerSpec.id


class NetworkSpec(BaseModel):
    layers: list[LayerSpec] = Field(default_factory=list)
    links: list[LinkSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_topology(self) -> "NetworkSpec":
        layer_ids = [layer.id for layer in self.layers]
        link_ids = [link.id for link in self.links]
        if len(layer_ids) != len(set(layer_ids)):
            raise ValueError("Layer IDs must be unique")
        if len(link_ids) != len(set(link_ids)):
            raise ValueError("Link IDs must be unique")

        known_layers = set(layer_ids)
        for link in self.links:
            if link.source not in known_layers or link.target not in known_layers:
                raise ValueError(f"Link {link.id!r} references an unknown layer")
            if link.source == link.target:
                raise ValueError(f"Link {link.id!r} cannot connect a layer to itself")

        # Kahn's algorithm keeps the API from accepting a graph the runtime
        # cannot execute.
        outgoing: dict[str, list[str]] = {layer_id: [] for layer_id in layer_ids}
        indegree = dict.fromkeys(layer_ids, 0)
        for link in self.links:
            outgoing[link.source].append(link.target)
            indegree[link.target] += 1
        ready = [layer_id for layer_id, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            layer_id = ready.pop()
            visited += 1
            for target in outgoing[layer_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(layer_ids):
            raise ValueError("Network links must form a directed acyclic graph")
        return self


class IdxSourceSpec(BaseModel):
    type: Literal["idx"] = "idx"
    path: Path


class NpySourceSpec(BaseModel):
    type: Literal["npy"] = "npy"
    path: Path


class CsvSourceSpec(BaseModel):
    type: Literal["csv"] = "csv"
    path: Path
    delimiter: str = Field(default=",", min_length=1, max_length=1)
    quotechar: str = Field(default='"', min_length=1, max_length=1)


class JsonSourceSpec(BaseModel):
    type: Literal["json"] = "json"
    path: Path


class ImageFolderSourceSpec(BaseModel):
    type: Literal["image_folder"] = "image_folder"
    path: Path
    extensions: tuple[str, ...] = (".bmp", ".jpeg", ".jpg", ".png")
    mode: str = "RGB"


SourceSpec = Annotated[
    IdxSourceSpec | NpySourceSpec | CsvSourceSpec | JsonSourceSpec | ImageFolderSourceSpec,
    Field(discriminator="type"),
]


class FlattenSpec(BaseModel):
    type: Literal["flatten"] = "flatten"
    field: str


class CastSpec(BaseModel):
    type: Literal["cast"] = "cast"
    field: str
    dtype: str


class ScaleSpec(BaseModel):
    type: Literal["scale"] = "scale"
    field: str
    divisor: float

    @field_validator("divisor")
    @classmethod
    def divisor_cannot_be_zero(cls, value: float) -> float:
        if value == 0:
            raise ValueError("Scale divisor cannot be zero")
        return value


JsonScalar = str | int | float | bool | None


class OneHotSpec(BaseModel):
    type: Literal["one_hot"] = "one_hot"
    field: str
    categories: tuple[JsonScalar, ...] | None = None


class SelectFieldsSpec(BaseModel):
    type: Literal["select"] = "select"
    fields: tuple[str, ...] = Field(min_length=1)


TransformSpec = Annotated[
    FlattenSpec | CastSpec | ScaleSpec | OneHotSpec | SelectFieldsSpec,
    Field(discriminator="type"),
]


class DatasetSpec(BaseModel):
    id: str = Field(min_length=1)
    source: SourceSpec
    transforms: list[TransformSpec] = Field(default_factory=list)
    # Canvas position is UI state, as it is for LayerSpec.
    pos: tuple[float, float] = (0.0, 0.0)


class FeedSpec(BaseModel):
    id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    field: str = Field(min_length=1)
    layer: str = Field(min_length=1)
    role: Literal["input", "target"] = "input"


class ProjectSpec(BaseModel):
    network: NetworkSpec = Field(default_factory=NetworkSpec)
    datasets: list[DatasetSpec] = Field(default_factory=list)
    feeds: list[FeedSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_project(self) -> "ProjectSpec":
        layers = {layer.id for layer in self.network.layers}
        datasets = {dataset.id for dataset in self.datasets}
        if len(datasets) != len(self.datasets) or layers & datasets:
            raise ValueError("Node IDs must be unique across layers and data sources")
        connections = [link.id for link in self.network.links] + [feed.id for feed in self.feeds]
        if len(connections) != len(set(connections)):
            raise ValueError("Connection IDs must be unique")
        pairs = [(link.source, link.target) for link in self.network.links]
        for feed in self.feeds:
            if feed.dataset not in datasets or feed.layer not in layers:
                raise ValueError("A connection references an unknown node")
            pairs.append((feed.dataset, feed.layer))
        if len(pairs) != len(set(pairs)):
            raise ValueError("These nodes are already connected")
        return self
