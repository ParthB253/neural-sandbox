"""FastAPI app. For now it just holds one in-memory NetworkSpec and lets the
frontend read and replace it. No persistence, no bridge to core.Network yet --
that (spec -> real Layers/Links, plus a training stream) is the next step.
"""
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .core import Layer as _Layer  # Registers the layer node type.
from .datasets import DataSource as _DataSource  # Registers datasource node types.
from .nodes import Node, NodeInfo
from .schemas import DatasetSpec, FeedSpec, LayerSpec, LinkSpec, NetworkSpec, ProjectSpec
from .uploads import router as uploads_router

app = FastAPI(title="nn-sandbox")
app.include_router(uploads_router)

# Vite (5173) and CRA-style (3000) dev servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
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
_datasets: dict[str, DatasetSpec] = {}
_feeds: dict[str, FeedSpec] = {}


@app.get("/project")
def get_project() -> ProjectSpec:
    return ProjectSpec(network=_network, datasets=list(_datasets.values()), feeds=list(_feeds.values()))


@app.put("/project")
def put_project(project: ProjectSpec) -> ProjectSpec:
    """Commit a canvas edit atomically, including its connections."""
    global _network, _datasets, _feeds
    _network = project.network
    _datasets = {dataset.id: dataset for dataset in project.datasets}
    _feeds = {feed.id: feed for feed in project.feeds}
    return project


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/network")
def get_network() -> NetworkSpec:
    return _network


@app.put("/network")
def put_network(spec: NetworkSpec) -> NetworkSpec:
    global _network, _feeds
    _network = spec
    layer_ids = {layer.id for layer in spec.layers}
    _feeds = {feed_id: feed for feed_id, feed in _feeds.items() if feed.layer in layer_ids}
    return _network


@app.post("/network/layers", status_code=status.HTTP_201_CREATED)
def create_layer(layer: LayerSpec) -> LayerSpec:
    if any(existing.id == layer.id for existing in _network.layers):
        raise HTTPException(status_code=409, detail=f"Layer already exists: {layer.id}")
    _set_network([*_network.layers, layer], _network.links)
    return layer


@app.get("/network/layers/{layer_id}")
def get_layer(layer_id: str) -> LayerSpec:
    return _find_layer(layer_id)


@app.put("/network/layers/{layer_id}")
def put_layer(layer_id: str, layer: LayerSpec) -> LayerSpec:
    _require_matching_id(layer_id, layer.id)
    _find_layer(layer_id)
    layers = [layer if existing.id == layer_id else existing for existing in _network.layers]
    _set_network(layers, _network.links)
    return layer


@app.delete("/network/layers/{layer_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_layer(layer_id: str) -> Response:
    global _feeds
    _find_layer(layer_id)
    layers = [layer for layer in _network.layers if layer.id != layer_id]
    links = [
        link for link in _network.links
        if link.source != layer_id and link.target != layer_id
    ]
    _set_network(layers, links)
    _feeds = {feed_id: feed for feed_id, feed in _feeds.items() if feed.layer != layer_id}
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post("/network/links", status_code=status.HTTP_201_CREATED)
def create_link(link: LinkSpec) -> LinkSpec:
    if any(existing.id == link.id for existing in _network.links):
        raise HTTPException(status_code=409, detail=f"Link already exists: {link.id}")
    _set_network(_network.layers, [*_network.links, link])
    return link


@app.get("/network/links/{link_id}")
def get_link(link_id: str) -> LinkSpec:
    return _find_link(link_id)


@app.put("/network/links/{link_id}")
def put_link(link_id: str, link: LinkSpec) -> LinkSpec:
    _require_matching_id(link_id, link.id)
    _find_link(link_id)
    links = [link if existing.id == link_id else existing for existing in _network.links]
    _set_network(_network.layers, links)
    return link


@app.delete("/network/links/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_link(link_id: str) -> Response:
    _find_link(link_id)
    _set_network(_network.layers, [link for link in _network.links if link.id != link_id])
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/datasets")
def get_datasets() -> list[DatasetSpec]:
    return list(_datasets.values())


@app.post("/datasets", status_code=status.HTTP_201_CREATED)
def create_dataset(dataset: DatasetSpec) -> DatasetSpec:
    if dataset.id in _datasets:
        raise HTTPException(status_code=409, detail=f"Dataset already exists: {dataset.id}")
    _datasets[dataset.id] = dataset
    return dataset


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str) -> DatasetSpec:
    return _find_dataset(dataset_id)


@app.put("/datasets/{dataset_id}")
def put_dataset(dataset_id: str, dataset: DatasetSpec) -> DatasetSpec:
    _require_matching_id(dataset_id, dataset.id)
    _find_dataset(dataset_id)
    _datasets[dataset_id] = dataset
    return dataset


@app.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(dataset_id: str) -> Response:
    global _feeds
    _find_dataset(dataset_id)
    del _datasets[dataset_id]
    _feeds = {feed_id: feed for feed_id, feed in _feeds.items() if feed.dataset != dataset_id}
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/feeds")
def get_feeds() -> list[FeedSpec]:
    return list(_feeds.values())


@app.post("/feeds", status_code=status.HTTP_201_CREATED)
def create_feed(feed: FeedSpec) -> FeedSpec:
    if feed.id in _feeds:
        raise HTTPException(status_code=409, detail=f"Feed already exists: {feed.id}")
    _validate_feed(feed)
    _feeds[feed.id] = feed
    return feed


@app.get("/feeds/{feed_id}")
def get_feed(feed_id: str) -> FeedSpec:
    return _find_feed(feed_id)


@app.put("/feeds/{feed_id}")
def put_feed(feed_id: str, feed: FeedSpec) -> FeedSpec:
    _require_matching_id(feed_id, feed.id)
    _find_feed(feed_id)
    _validate_feed(feed)
    _feeds[feed_id] = feed
    return feed


@app.delete("/feeds/{feed_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_feed(feed_id: str) -> Response:
    _find_feed(feed_id)
    del _feeds[feed_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/node-types")
def get_node_types() -> list[NodeInfo]:
    """Describe every class that can be represented on the canvas."""
    return Node.node_types()


@app.get("/node-types/{node_type:path}")
def get_node_type(node_type: str) -> NodeInfo:
    """Describe one canvas-node type using metadata owned by its class."""
    try:
        return Node.info_for(node_type)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error.args[0])) from error


def _set_network(layers: list[LayerSpec], links: list[LinkSpec]) -> None:
    global _network
    try:
        _network = NetworkSpec(layers=layers, links=links)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _find_layer(layer_id: str) -> LayerSpec:
    for layer in _network.layers:
        if layer.id == layer_id:
            return layer
    raise HTTPException(status_code=404, detail=f"Unknown layer: {layer_id}")


def _find_link(link_id: str) -> LinkSpec:
    for link in _network.links:
        if link.id == link_id:
            return link
    raise HTTPException(status_code=404, detail=f"Unknown link: {link_id}")


def _find_dataset(dataset_id: str) -> DatasetSpec:
    try:
        return _datasets[dataset_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {dataset_id}") from error


def _find_feed(feed_id: str) -> FeedSpec:
    try:
        return _feeds[feed_id]
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown feed: {feed_id}") from error


def _validate_feed(feed: FeedSpec) -> None:
    _find_dataset(feed.dataset)
    _find_layer(feed.layer)


def _require_matching_id(path_id: str, body_id: str) -> None:
    if path_id != body_id:
        raise HTTPException(status_code=409, detail="The path ID and body ID must match")
