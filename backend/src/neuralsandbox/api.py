"""An in-memory canvas draft, explicitly saved networks, and shared file storage."""
from uuid import UUID, uuid4
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .core import Layer as _Layer  # Registers the layer node type.
from .datasets import DataSource as _DataSource  # Registers datasource node types.
from .nodes import Node, NodeInfo
from .schemas import DatasetSpec, FeedSpec, LayerSpec, LinkSpec, NetworkSpec, ProjectSpec
from .uploads import router as uploads_router
from . import network_store
from .execution import router as execution_router

app = FastAPI(title="NeuralSandbox")
app.include_router(uploads_router)
app.include_router(execution_router)

# Vite (5173) and CRA-style (3000) dev servers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup creates an empty draft. Saved networks are opened explicitly.
_network = NetworkSpec()
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


@app.get("/networks")
def list_networks() -> list[network_store.NetworkSummary]:
    return network_store.list_networks()


@app.get("/networks/{network_id}")
def read_saved_network(network_id: UUID) -> ProjectSpec:
    return network_store.read_network(network_id)


@app.put("/networks/{network_id}")
def save_network(network_id: UUID, project: ProjectSpec) -> ProjectSpec:
    if network_id != project.network.id:
        raise HTTPException(status_code=409, detail="The path ID and network ID must match")
    network_store.write_network(project)
    return put_project(project)


@app.post("/networks", status_code=status.HTTP_201_CREATED)
def save_network_as(project: ProjectSpec) -> ProjectSpec:
    copy = project.model_copy(deep=True)
    copy.network.id = uuid4()
    network_store.write_network(copy)
    return put_project(copy)


@app.post("/networks/{network_id}/open")
def open_network(network_id: UUID) -> ProjectSpec:
    return put_project(network_store.read_network(network_id))


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
        _network = NetworkSpec(id=_network.id, title=_network.title, layers=layers, links=links)
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
