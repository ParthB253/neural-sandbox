from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuralsandbox.api import app
from neuralsandbox.datasets import Concatenate, build_data_source, build_feed
from neuralsandbox.datasets.data import DataTable
from neuralsandbox.schemas import ProjectSpec
from neuralsandbox.training_runtime import TrainingRuntime


def test_income_fields_resolve_to_29_features_and_reach_training(tmp_path: Path) -> None:
    fields = ["age", "education.num", "capital.gain", "capital.loss", "hours.per.week", "workclass", "occupation"]
    path = tmp_path / "income.csv"
    path.write_text(
        ",".join(fields + ["income"]) + "\n" + "\n".join(
            f"{20+i},8,0,0,40,w{i % 9},o{i % 15},{i % 2}" for i in range(30)
        ), encoding="utf-8",
    )
    project = ProjectSpec.model_validate({
        "network": {"layers": [
            {"id": "input", "size": 32, "size_mode": "auto", "activation": "identity"},
            {"id": "output", "size": 1, "activation": "sigmoid"},
        ], "links": [{"id": "weights", "source": "input", "target": "output"}]},
        "datasets": [{"id": "income", "source": {"type": "csv", "path": str(path)}}],
        "feeds": [
            {"id": "features", "dataset": "income", "layer": "input", "field": "age", "transforms": [
                {"type": "select", "fields": fields},
                *[{"type": "cast", "field": field, "dtype": "float32"} for field in fields[:5]],
                {"type": "scale", "field": "age", "divisor": 100},
                {"type": "one_hot", "field": "workclass"},
                {"type": "one_hot", "field": "occupation"},
            ]},
            {"id": "target", "dataset": "income", "layer": "output", "field": "income", "role": "target",
             "transforms": [{"type": "cast", "field": "income", "dtype": "float32"}]},
        ],
    })
    client = TestClient(app)
    before = client.post("/project/analysis", json=project.model_dump(mode="json")).json()
    assert before["layers"]["input"]["size"] == 1
    assert before["feeds"]["features"]["field"] == "age"
    assert before["feeds"]["features"]["fields"]["occupation"]["sample_shape"] == [15]

    changed = project.model_dump(mode="json")
    changed["feeds"][0]["transforms"].append({"type": "concatenate", "fields": fields, "field": "features"})
    changed["feeds"][0]["field"] = "features"
    fixed = ProjectSpec.model_validate(changed)
    result = client.post("/project/analysis", json=changed).json()
    assert result["issues"] == []
    assert result["layers"]["input"]["size"] == 29
    assert result["feeds"]["features"]["width"] == 29
    assert result["links"]["weights"]["weight_shape"] == [29, 1]
    feed = build_feed(fixed.feeds[0], build_data_source(fixed.datasets[0]))
    assert feed.feature_count == 29
    runtime = TrainingRuntime(fixed)
    values = runtime.inputs[runtime.layers["input"]]
    assert values.shape == (30, 29)
    assert values[0, 0] == pytest.approx(.2)
    assert values[0, 5] == 1
    assert values[0, 14] == 1
    assert project.feeds[0].field == "age"


def test_concatenate_preserves_order_and_rejects_unencoded_or_multidimensional_fields() -> None:
    original = DataTable({"number": np.array([2., 4.]), "vector": np.eye(2), "text": np.array(["a", "b"]), "image": np.zeros((2, 2, 2))})
    combined = Concatenate(("vector", "number"), "features").apply(original)
    np.testing.assert_array_equal(combined.fields["features"], [[1, 0, 2], [0, 1, 4]])
    assert "features" not in original.fields
    with pytest.raises(ValueError, match="not numeric"):
        Concatenate(("number", "text"), "features").apply(original)
    with pytest.raises(ValueError, match="Flatten"):
        Concatenate(("image",), "features").apply(original)
    with pytest.raises(ValueError, match="Unknown field"):
        Concatenate(("missing",), "features").apply(original)
