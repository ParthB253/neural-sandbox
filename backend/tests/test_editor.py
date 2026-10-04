"""Editor contracts: saved file bytes and atomic topology changes."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neuralsandbox import api, uploads, network_store

client = TestClient(api.app)


@pytest.fixture(autouse=True)
def isolated_project(tmp_path, monkeypatch):
    monkeypatch.setattr(network_store, "NETWORK_DIR", tmp_path / "networks")
    monkeypatch.setattr(uploads, "UPLOAD_DIR", tmp_path / "files")
    monkeypatch.setattr(api, "_network", api.NetworkSpec())
    monkeypatch.setattr(api, "_datasets", {})
    monkeypatch.setattr(api, "_feeds", {})


def test_stored_upload_reusable_and_protected_while_used():
    response = client.post("/files", files={"file": ("../../example.csv", b"value\n1\n2\n", "text/csv")})
    assert response.status_code == 201
    stored = response.json()
    assert stored["name"] == "example.csv"
    assert Path(stored["path"]).read_bytes() == b"value\n1\n2\n"
    assert client.get("/files").json() == [stored]
    # A fresh client still sees files stored on disk.
    assert TestClient(api.app).get("/files").json() == [stored]
    project = {
        "network": {"layers": [{"id": "input", "size": 2}], "links": []},
        "datasets": [
            {"id": "data-a", "source": {"type": "csv", "path": stored["path"]}},
            {"id": "data-b", "source": {"type": "csv", "path": stored["path"]}},
        ],
        "feeds": [{"id": "connection", "dataset": "data-a", "layer": "input", "field": "value"}],
    }
    assert client.put("/project", json=project).status_code == 200
    assert client.delete(f'/files/{stored["id"]}').status_code == 409
    assert client.put("/project", json={}).status_code == 200
    assert client.delete(f'/files/{stored["id"]}').status_code == 204
    assert not Path(stored["path"]).exists()
    assert client.get("/files").json() == []


def test_invalid_project_does_not_partially_replace_saved_draft():
    valid = {"network": {"layers": [{"id": "a", "size": 2}], "links": []}}
    saved = client.put("/project", json=valid).json()
    invalid = {**valid, "feeds": [{"id": "bad", "dataset": "missing", "layer": "a", "field": "value"}]}
    assert client.put("/project", json=invalid).status_code == 422
    assert client.get("/project").json() == saved


@pytest.mark.parametrize("project", [
    {"network": {"layers": [{"id": "a", "size": 1}], "links": []}, "datasets": [{"id": "a", "source": {"type": "csv", "path": "a.csv"}}]},
    {"network": {"layers": [{"id": "a", "size": 1}, {"id": "b", "size": 1}], "links": [{"id": "one", "source": "a", "target": "b"}, {"id": "two", "source": "a", "target": "b"}]}},
    {"network": {"layers": [{"id": "a", "size": 1}, {"id": "b", "size": 1}], "links": [{"id": "one", "source": "a", "target": "b"}, {"id": "two", "source": "b", "target": "a"}]}},
])
def test_invalid_editor_topologies_are_rejected(project):
    assert client.put("/project", json=project).status_code == 422


def test_empty_and_oversized_files_leave_no_orphans(monkeypatch):
    monkeypatch.setattr(uploads, "MAX_BYTES", 4)
    assert client.post("/files", files={"file": ("empty.csv", b"")}).status_code == 422
    assert client.post("/files", files={"file": ("big.csv", b"12345")}).status_code == 413
    assert client.get("/files").json() == []
    assert list(uploads.UPLOAD_DIR.iterdir()) == []


def test_same_filename_preserves_both_uploads():
    first = client.post("/files", files={"file": ("data.csv", b"a\n1")}).json()
    second = client.post("/files", files={"file": ("data.csv", b"a\n2")}).json()
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == b"a\n1"
    assert Path(second["path"]).read_bytes() == b"a\n2"
