"""Named network persistence without ownership or duplication of project files."""
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from neuralsandbox import api, network_store, uploads
from neuralsandbox.schemas import NetworkSpec

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(network_store, "NETWORK_DIR", tmp_path / "networks")
    monkeypatch.setattr(uploads, "UPLOAD_DIR", tmp_path / "files")
    monkeypatch.setattr(api, "_network", NetworkSpec())
    monkeypatch.setattr(api, "_datasets", {})
    monkeypatch.setattr(api, "_feeds", {})


def test_new_network_is_empty_and_has_identity_and_title():
    first = client.get("/project").json()
    second = NetworkSpec()
    assert UUID(first["network"]["id"]) != second.id
    assert first["network"]["title"] == "Untitled network"
    assert first["network"]["layers"] == []
    assert first["network"]["links"] == []
    assert first["datasets"] == first["feeds"] == []
    assert client.get("/networks").json() == []


def test_save_overwrite_save_as_and_open_survive_a_fresh_draft():
    project = client.get("/project").json()
    original_id = project["network"]["id"]
    project["network"]["title"] = "  Classifier  "
    project["network"]["layers"] = [{"id": "input", "size": 8, "activation": "relu", "pos": [40, 60]}]
    saved = client.put(f"/networks/{original_id}", json=project)
    assert saved.status_code == 200
    assert saved.json()["network"]["title"] == "Classifier"

    # Layer CRUD must not silently replace network identity or its title.
    assert client.post("/network/layers", json={"id": "output", "size": 2}).status_code == 201
    current = client.get("/project").json()
    assert current["network"]["id"] == original_id
    assert current["network"]["title"] == "Classifier"
    current["network"]["links"] = [{"id": "edge", "source": "input", "target": "output"}]
    assert client.put(f"/networks/{original_id}", json=current).status_code == 200
    assert len(client.get("/networks").json()) == 1

    current["network"]["title"] = "Classifier copy"
    copy = client.post("/networks", json=current)
    assert copy.status_code == 201
    copy_id = copy.json()["network"]["id"]
    assert copy_id != original_id
    assert client.get(f"/networks/{original_id}").json()["network"]["title"] == "Classifier"
    assert len(client.get("/networks").json()) == 2

    # Simulate a restart's empty in-memory draft, keeping files on disk.
    api._network = NetworkSpec()
    api._datasets = {}
    api._feeds = {}
    reconnected = TestClient(api.app)
    assert len(reconnected.get("/networks").json()) == 2
    opened = reconnected.post(f"/networks/{original_id}/open").json()
    assert opened["network"]["id"] == original_id
    assert opened["network"]["title"] == "Classifier"
    assert len(opened["network"]["layers"]) == 2
    assert len(opened["network"]["links"]) == 1
    assert reconnected.get("/project").json() == opened
    assert (network_store.NETWORK_DIR / f"{copy_id}.json").exists()


def test_shared_files_are_not_copied_and_saved_references_are_protected():
    stored = client.post("/files", files={"file": ("data.csv", b"value\n1\n")}).json()
    project = client.get("/project").json()
    project["datasets"] = [{"id": "data", "source": {"type": "csv", "path": stored["path"]}, "transforms": [{"type": "scale", "field": "value", "divisor": 255}], "pos": [-200, 40]}]
    project["network"]["layers"] = [{"id": "layer", "size": 1}]
    project["feeds"] = [{"id": "connection", "dataset": "data", "layer": "layer", "field": "value"}]
    first = client.put(f'/networks/{project["network"]["id"]}', json=project).json()
    copy = client.post("/networks", json=first).json()
    assert copy["datasets"] == first["datasets"]
    assert copy["feeds"] == first["feeds"]
    assert client.get("/files").json() == [stored]
    assert client.put("/project", json={}).status_code == 200
    assert client.delete(f'/files/{stored["id"]}').status_code == 409
    assert Path(stored["path"]).read_bytes() == b"value\n1\n"
    assert client.post(f'/networks/{copy["network"]["id"]}/open').json() == copy


def test_bad_save_does_not_replace_a_snapshot_or_active_draft():
    project = client.get("/project").json()
    network_id = project["network"]["id"]
    assert client.put(f"/networks/{network_id}", json=project).status_code == 200
    project["network"]["title"] = "   "
    assert client.put(f"/networks/{network_id}", json=project).status_code == 422
    assert client.get(f"/networks/{network_id}").json()["network"]["title"] == "Untitled network"
    project["network"]["title"] = "valid"
    assert client.put(f"/networks/{uuid4()}", json=project).status_code == 409
    assert client.post(f"/networks/{uuid4()}/open").status_code == 404
    assert client.get("/project").json()["network"]["title"] == "Untitled network"


def test_write_failure_preserves_saved_network_and_cleans_tempfile(monkeypatch):
    project = client.get("/project").json()
    network_id = project["network"]["id"]
    assert client.put(f"/networks/{network_id}", json=project).status_code == 200
    original = (network_store.NETWORK_DIR / f"{network_id}.json").read_bytes()

    def fail_replace(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(Path, "replace", fail_replace)
    project["network"]["title"] = "Do not partially save"
    assert client.put(f"/networks/{network_id}", json=project).status_code == 503
    assert (network_store.NETWORK_DIR / f"{network_id}.json").read_bytes() == original
    assert list(network_store.NETWORK_DIR.glob("*.tmp")) == []
    assert client.get("/project").json()["network"]["title"] == "Untitled network"
