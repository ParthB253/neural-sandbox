"""Compile a project snapshot and return a single forward result, without traces.

Source/Feed own shape metadata. Analysis only resolves bindings and reports
problems. Runtime objects never mutate the editable project.
"""
import hashlib
import json
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .core import Layer, Network
from .datasets.factory import build_data_source
from .datasets.feed import Feed
from .schemas import ProjectSpec

router = APIRouter()
MAX_PARAMETERS = 2_000_000
MAX_OUTPUT_VALUES = 100_000


def _identity(x):
    return x


def _sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def _softmax(x):
    exp = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


ACTIVATIONS = {
    "identity": _identity,
    "relu": lambda x: np.maximum(x, 0),
    "sigmoid": _sigmoid,
    "tanh": np.tanh,
    "softmax": _softmax,
}


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def project_revision(project: ProjectSpec) -> str:
    # Moving a node or renaming a document does not change the computation.
    config = project.model_dump(mode="json")
    config["network"].pop("title")
    for node in config["network"]["layers"] + config["datasets"]:
        node.pop("pos")
    return _hash(config)


def _prepare(project: ProjectSpec):
    sources, bindings, sizes = {}, {}, {}
    report = {"project_revision": project_revision(project), "datasets": {},
              "layers": {}, "links": {}, "issues": []}

    def issue(node_id, message):
        report["issues"].append({"node_id": node_id, "message": message})

    for spec in project.datasets:
        try:
            source = build_data_source(spec)
            schema = source.schema
            sources[spec.id] = source
            report["datasets"][spec.id] = {
                "sample_count": source.sample_count,
                "fields": {name: {"dtype": field.dtype, "sample_shape": field.sample_shape}
                           for name, field in schema.items()},
            }
        except (ValueError, TypeError, OSError, KeyError, OverflowError, EOFError) as error:
            issue(spec.id, str(error))

    for spec in project.feeds:
        if spec.dataset not in sources:
            continue
        feed = Feed(sources[spec.dataset], spec.field, None, spec.role)
        try:
            width = feed.feature_count
            dtype = np.dtype(feed.source.schema[feed.field].dtype)
            if width <= 0:
                raise ValueError("A feed must contain at least one feature")
            if dtype.kind not in "biuf":
                raise ValueError(f"Field {feed.field!r} is {dtype}; cast or encode it as numeric data")
            bindings[spec.id] = (feed, width)
        except (ValueError, TypeError) as error:
            issue(spec.id, str(error))

    for layer in project.network.layers:
        inputs = [feed for feed in project.feeds if feed.layer == layer.id and feed.role == "input"]
        predecessors = [link for link in project.network.links if link.target == layer.id]
        width = layer.size
        if len(inputs) > 1:
            issue(layer.id, "Multiple input feeds require an explicit combining operation")
        if inputs and predecessors:
            issue(layer.id, "A layer cannot receive both an input feed and weighted links")
        if not predecessors and not inputs:
            issue(layer.id, "Input layer needs a data feed")
        if layer.size_mode == "auto":
            width = bindings[inputs[0].id][1] if len(inputs) == 1 and inputs[0].id in bindings else None
            if width is None:
                issue(layer.id, "Auto size requires one valid input feed")
        for feed in inputs:
            if feed.id in bindings and width != bindings[feed.id][1]:
                issue(feed.id, f"Feed width {bindings[feed.id][1]} does not match layer width {width}; use Auto size")
        if layer.activation not in ACTIVATIONS:
            issue(layer.id, f"Unsupported activation: {layer.activation}")
        sizes[layer.id] = width
        report["layers"][layer.id] = {"size": width, "size_mode": layer.size_mode}
    for link in project.network.links:
        report["links"][link.id] = {"weight_shape": [sizes[link.source], sizes[link.target]]}
    return report, sources, bindings, sizes


@router.post("/project/analysis")
def analyze(project: ProjectSpec):
    return _prepare(project)[0]


class ForwardRequest(BaseModel):
    project: ProjectSpec
    sample_index: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0, le=2**32 - 1)


@router.post("/executions/forward")
def forward(request: ForwardRequest):
    report, sources, bindings, sizes = _prepare(request.project)
    if report["issues"]:
        raise HTTPException(422, detail="; ".join(f"{i['node_id']}: {i['message']}" for i in report["issues"]))
    spec = request.project.network
    if not spec.layers:
        raise HTTPException(422, detail="Add at least one layer")
    input_specs = [feed for feed in request.project.feeds if feed.role == "input"]
    counts = {sources[feed.dataset].sample_count for feed in input_specs}
    if len(counts) != 1:
        raise HTTPException(422, detail="Input sources must have equal example counts and aligned rows")
    if request.sample_index >= next(iter(counts)):
        raise HTTPException(422, detail="Sample index out of range (or dataset is empty)")
    parameters = sum(sizes[link.source] * sizes[link.target] for link in spec.links) + sum(sizes.values())
    if parameters > MAX_PARAMETERS:
        raise HTTPException(422, detail=f"Network exceeds the v0 limit of {MAX_PARAMETERS:,} parameters")
    output_ids = [layer.id for layer in spec.layers if not any(link.source == layer.id for link in spec.links)]
    if sum(sizes[id] for id in output_ids) > MAX_OUTPUT_VALUES:
        raise HTTPException(422, detail="Output is too large to return")

    # Constructors receive explicit arrays so they don't touch global RNG state.
    rng = np.random.default_rng(request.seed)
    def no_backward(_):
        raise NotImplementedError("This runtime supports forward execution only")
    layers = {layer.id: Layer(ACTIVATIONS[layer.activation], no_backward,
                             from_array=np.zeros(sizes[layer.id]), bias=np.zeros((1, sizes[layer.id])))
              for layer in spec.layers}
    network = Network(None, None, list(layers.values()))
    for link in spec.links:
        shape = (sizes[link.source], sizes[link.target])
        network.connect(layers[link.source], layers[link.target], rng.uniform(-0.5, 0.5, shape))
    for layer in spec.layers:
        if layers[layer.id].from_links:
            layers[layer.id].bias = rng.uniform(-0.5, 0.5, (1, sizes[layer.id]))
    for feed_spec in input_specs:
        feed = bindings[feed_spec.id][0]
        feed.layer = layers[feed_spec.layer]
        values = feed.feed(request.sample_index).astype(np.float64)
        if not np.isfinite(values).all():
            raise HTTPException(422, detail=f"{feed_spec.id}: input contains NaN or infinity")
        feed.layer.values = values
    with np.errstate(over="ignore", invalid="ignore"):
        for layer in network.topo_order():
            if not np.isfinite(layer.compute()).all():
                raise HTTPException(422, detail="Forward pass produced NaN or infinity")

    # Hash actual parameters and input dataset bytes: revisions identify this
    # computation even when a server-side file is replaced at the same path.
    data_hash = hashlib.sha256()
    for id in sorted({feed.dataset for feed in input_specs}):
        data_hash.update(id.encode())
        for name, values in sorted(sources[id].data.fields.items()):
            data_hash.update(name.encode())
            data_hash.update(str((values.dtype, values.shape)).encode())
            data_hash.update(values.tobytes() if values.dtype.kind != "O" else repr(values.tolist()).encode())
    model_hash = hashlib.sha256()
    for layer in spec.layers:
        model_hash.update(layers[layer.id].bias.tobytes())
    for link in network.links:
        model_hash.update(link.weights.tobytes())
    return {"execution_id": str(uuid4()), "project_revision": report["project_revision"],
            "model_revision": _hash([report["project_revision"], model_hash.hexdigest()]),
            "data_revision": data_hash.hexdigest(), "sample_index": request.sample_index,
            "seed": request.seed, "trained": False,
            "outputs": {id: {"values": layers[id].values[0].tolist(), "shape": [sizes[id]],
                             "activation_applied": bool(layers[id].from_links),
                             "activation": next(layer.activation for layer in spec.layers if layer.id == id)}
                        for id in output_ids}, "layers": report["layers"], "links": report["links"]}
