from fastapi.testclient import TestClient

from neuralsandbox.api import app


client = TestClient(app)


def test_node_type_catalog_is_driven_by_registered_classes():
    response = client.get("/node-types")

    assert response.status_code == 200
    node_types = {item["type"]: item for item in response.json()}
    assert set(node_types) == {
        "dataset.csv",
        "dataset.idx",
        "dataset.image_folder",
        "dataset.json",
        "dataset.npy",
        "layer",
    }
    assert node_types["dataset.csv"]["category"] == "dataset"
    assert node_types["dataset.csv"]["outputs"][0]["dynamic"] is True
    assert node_types["layer"]["inputs"][0]["multiple"] is True


def test_get_one_node_type():
    response = client.get("/node-types/dataset.csv")

    assert response.status_code == 200
    assert response.json()["label"] == "CSV dataset"
    assert {field["name"] for field in response.json()["config"]} == {
        "path",
        "delimiter",
        "quotechar",
    }


def test_unknown_node_type_is_404():
    response = client.get("/node-types/dataset.unknown")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown node type: dataset.unknown"


def test_layer_and_link_crud_supports_canvas_edits():
    network = {
        "layers": [
            {"id": "crud-in", "size": 2, "activation": "identity", "pos": [0, 0]},
            {"id": "crud-out", "size": 1, "activation": "identity", "pos": [300, 0]},
        ],
        "links": [],
    }
    assert client.put("/network", json=network).status_code == 200

    added = {"id": "crud-hidden", "size": 3, "activation": "relu", "pos": [150, 20]}
    assert client.post("/network/layers", json=added).status_code == 201
    edited = {**added, "size": 4, "pos": [150, 40]}
    response = client.put("/network/layers/crud-hidden", json=edited)
    assert response.status_code == 200
    assert response.json()["size"] == 4

    first_link = {"id": "crud-in-hidden", "source": "crud-in", "target": "crud-hidden"}
    second_link = {"id": "crud-hidden-out", "source": "crud-hidden", "target": "crud-out"}
    assert client.post("/network/links", json=first_link).status_code == 201
    assert client.post("/network/links", json=second_link).status_code == 201

    response = client.delete("/network/layers/crud-hidden")
    assert response.status_code == 204
    saved = client.get("/network").json()
    assert {layer["id"] for layer in saved["layers"]} == {"crud-in", "crud-out"}
    assert saved["links"] == []


def test_network_rejects_unknown_link_endpoints_and_cycles():
    unknown_endpoint = {
        "layers": [{"id": "only", "size": 1}],
        "links": [{"id": "bad", "source": "only", "target": "missing"}],
    }
    assert client.put("/network", json=unknown_endpoint).status_code == 422

    cycle = {
        "layers": [{"id": "a", "size": 1}, {"id": "b", "size": 1}],
        "links": [
            {"id": "a-b", "source": "a", "target": "b"},
            {"id": "b-a", "source": "b", "target": "a"},
        ],
    }
    assert client.put("/network", json=cycle).status_code == 422


def test_dataset_and_feed_crud_and_cascading_delete(tmp_path):
    network = {
        "layers": [{"id": "feed-input", "size": 2, "activation": "identity"}],
        "links": [],
    }
    assert client.put("/network", json=network).status_code == 200
    dataset = {
        "id": "crud-dataset",
        "source": {"type": "npy", "path": str(tmp_path / "values.npy")},
        "transforms": [{"type": "flatten", "field": "value"}],
        "pos": [-200, 0],
    }
    assert client.post("/datasets", json=dataset).status_code == 201
    assert client.get("/datasets/crud-dataset").status_code == 200

    edited = {**dataset, "pos": [-250, 50]}
    assert client.put("/datasets/crud-dataset", json=edited).status_code == 200

    feed = {
        "id": "crud-feed",
        "dataset": "crud-dataset",
        "field": "value",
        "layer": "feed-input",
        "role": "input",
    }
    assert client.post("/feeds", json=feed).status_code == 201
    assert client.get("/feeds/crud-feed").json() == feed

    assert client.delete("/datasets/crud-dataset").status_code == 204
    assert client.get("/feeds/crud-feed").status_code == 404


def test_feed_requires_existing_nodes():
    feed = {
        "id": "orphan-feed",
        "dataset": "missing-dataset",
        "field": "value",
        "layer": "missing-layer",
        "role": "input",
    }
    assert client.post("/feeds", json=feed).status_code == 404
