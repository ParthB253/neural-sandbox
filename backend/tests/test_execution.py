import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuralsandbox.api import app
from neuralsandbox.core import Layer
from neuralsandbox.datasets.factory import build_data_source
from neuralsandbox.datasets.feed import Feed
from neuralsandbox.schemas import DatasetSpec

client = TestClient(app)


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "images.npy"
    np.save(path, np.arange(2 * 28 * 28, dtype=np.float64).reshape(2, 28, 28))
    return {
        "network": {"id": "00000000-0000-0000-0000-000000000001", "layers": [
            {"id": "input", "size": 16, "size_mode": "auto", "activation": "identity"},
            {"id": "output", "size": 10, "activation": "softmax"},
        ], "links": [{"id": "weights", "source": "input", "target": "output"}]},
        "datasets": [{"id": "images", "source": {"type": "npy", "path": str(path)},
                      "transforms": [{"type": "flatten", "field": "value"},
                                     {"type": "scale", "field": "value", "divisor": 255}]}],
        "feeds": [{"id": "pixels", "dataset": "images", "field": "value", "layer": "input"}],
    }


def run(project, **kwargs):
    return client.post("/executions/forward", json={"project": project, **kwargs})


def test_source_feed_metadata_and_auto_mnist_shape(project):
    source = build_data_source(DatasetSpec.model_validate(project["datasets"][0]))
    feed = Feed(source, "value", None, "input")
    assert source.sample_count == 2
    assert source.schema["value"].sample_shape == (784,)
    assert feed.sample_shape == (784,)
    assert feed.feature_count == 784
    response = client.post("/project/analysis", json=project)
    assert response.status_code == 200
    result = response.json()
    assert result["issues"] == []
    assert result["layers"]["input"]["size"] == 784
    assert result["layers"]["output"]["size"] == 10
    assert result["links"]["weights"]["weight_shape"] == [784, 10]
    assert project["network"]["layers"][0]["size"] == 16


def test_forward_is_reproducible_and_does_not_mutate_draft(project):
    client.put("/project", json=project)
    before = client.get("/project").json()
    a, b = run(project).json(), run(project).json()
    assert a["outputs"] == b["outputs"]
    assert a["model_revision"] == b["model_revision"]
    assert a["execution_id"] != b["execution_id"]
    assert a["trained"] is False
    assert np.isclose(sum(a["outputs"]["output"]["values"]), 1)
    assert client.get("/project").json() == before
    changed_seed = run(project, seed=1).json()
    assert changed_seed["model_revision"] != a["model_revision"]
    assert changed_seed["outputs"] != a["outputs"]


def test_revisions_distinguish_data_model_and_presentation(project):
    first = run(project).json()
    moved = copy.deepcopy(project)
    moved["network"]["title"] = "Renamed"
    moved["network"]["layers"][0]["pos"] = [900, 400]
    assert run(moved).json()["project_revision"] == first["project_revision"]
    assert run(project, sample_index=1).json()["model_revision"] == first["model_revision"]
    path = project["datasets"][0]["source"]["path"]
    np.save(path, np.zeros((2, 28, 28)))
    changed = run(project).json()
    assert changed["data_revision"] != first["data_revision"]
    assert changed["project_revision"] == first["project_revision"]
    assert changed["model_revision"] == first["model_revision"]


@pytest.mark.parametrize("index", [-1, 2, 999])
def test_invalid_example_index(project, index):
    assert run(project, sample_index=index).status_code == 422


@pytest.mark.parametrize("problem", ["matrix", "manual", "field", "file", "activation", "auto", "nonfinite", "empty"])
def test_execution_reports_invalid_configuration(project, problem):
    if problem == "matrix":
        project["datasets"][0]["transforms"] = []
    elif problem == "manual":
        project["network"]["layers"][0]["size_mode"] = "manual"
    elif problem == "field":
        project["feeds"][0]["field"] = "missing"
    elif problem == "file":
        project["datasets"][0]["source"]["path"] += ".missing"
    elif problem == "activation":
        project["network"]["layers"][1]["activation"] = "unknown"
    elif problem == "auto":
        project["network"]["layers"][1]["size_mode"] = "auto"
    elif problem == "nonfinite":
        np.save(project["datasets"][0]["source"]["path"], np.full((2, 28, 28), np.nan))
    elif problem == "empty":
        np.save(project["datasets"][0]["source"]["path"], np.empty((0, 28, 28)))
    assert run(project).status_code == 422


def test_scalar_feed_and_bounds(project):
    np.save(project["datasets"][0]["source"]["path"], np.array([2., 3.]))
    project["datasets"][0]["transforms"] = []
    assert run(project).json()["layers"]["input"]["size"] == 1
    source = build_data_source(DatasetSpec.model_validate(project["datasets"][0]))
    layer = Layer(lambda x: x, lambda x: x, from_array=np.zeros(1), bias=np.zeros((1, 1)))
    feed = Feed(source, "value", layer, "input")
    assert feed.feed(1).tolist() == [[3.]]
    for index in (-1, 2):
        with pytest.raises(IndexError):
            feed.feed(index)


def test_diamond_dag_uses_sum_of_weighted_contributions(project):
    project["network"]["layers"].extend([
        {"id": "left", "size": 2, "activation": "identity"},
        {"id": "right", "size": 3, "activation": "identity"},
    ])
    project["network"]["links"] = [
        {"id": "il", "source": "input", "target": "left"},
        {"id": "ir", "source": "input", "target": "right"},
        {"id": "lo", "source": "left", "target": "output"},
        {"id": "ro", "source": "right", "target": "output"},
    ]
    result = run(project, seed=42)
    assert result.status_code == 200
    rng = np.random.default_rng(42)
    il, ir = rng.uniform(-.5, .5, (784, 2)), rng.uniform(-.5, .5, (784, 3))
    lo, ro = rng.uniform(-.5, .5, (2, 10)), rng.uniform(-.5, .5, (3, 10))
    ob, lb, rb = rng.uniform(-.5, .5, (1, 10)), rng.uniform(-.5, .5, (1, 2)), rng.uniform(-.5, .5, (1, 3))
    pixels = np.arange(784).reshape(1, -1) / 255
    logits = (pixels @ il + lb) @ lo + (pixels @ ir + rb) @ ro + ob
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    assert np.allclose(result.json()["outputs"]["output"]["values"], expected[0])


def test_auto_width_follows_replaced_source_and_manual_outputs_stay_fixed(project):
    np.save(project["datasets"][0]["source"]["path"], np.ones((2, 8, 8)))
    result = run(project).json()
    assert result["layers"]["input"]["size"] == 64
    assert result["links"]["weights"]["weight_shape"] == [64, 10]


@pytest.mark.parametrize("case", ["multiple", "mixed", "unaligned", "limit", "nonnumeric", "truncated"])
def test_execution_rejects_ambiguous_or_unusable_inputs(project, case, tmp_path):
    if case in ("multiple", "mixed", "unaligned"):
        path = tmp_path / "extra.npy"
        np.save(path, np.ones((3 if case == "unaligned" else 2, 784)))
        project["datasets"].append({"id": "extra", "source": {"type": "npy", "path": str(path)}})
        project["feeds"].append({"id": "extra-feed", "dataset": "extra", "field": "value", "layer": "input"})
        if case == "mixed":
            project["feeds"][-1]["layer"] = "output"
        elif case == "unaligned":
            project["network"]["layers"].append({"id": "other", "size": 784, "activation": "identity"})
            project["feeds"][-1]["layer"] = "other"
    elif case == "limit":
        project["network"]["layers"][1]["size"] = 3_000_000
    elif case == "nonnumeric":
        np.save(project["datasets"][0]["source"]["path"], np.array(["hello", "world"]))
        project["datasets"][0]["transforms"] = []
    elif case == "truncated":
        # A zero-dimensional array has no sample axis.
        np.save(project["datasets"][0]["source"]["path"], np.array(1))
    assert run(project).status_code == 422
