import json

import numpy as np
import pytest
from PIL import Image

from neuralsandbox.core import Layer
from neuralsandbox.datasets import (
    Cast,
    CsvSource,
    DataSource,
    DataTransform,
    Feed,
    Flatten,
    ImageFolderSource,
    JsonSource,
    NpySource,
    OneHot,
    Scale,
    SelectFields,
    build_data_source,
)
from neuralsandbox.schemas import DatasetSpec


def test_csv_source_builds_named_fields_and_applies_transforms(tmp_path):
    path = tmp_path / "animals.csv"
    path.write_text("pixels,label\n0,cat\n255,dog\n", encoding="utf-8")

    data = CsvSource(
        path,
        transforms=(
            Cast("pixels", "float64"),
            Scale("pixels", 255),
            OneHot("label", ("cat", "dog")),
        ),
    ).load()

    assert data.sample_count == 2
    assert data.schema["pixels"].sample_shape == ()
    assert data.schema["label"].sample_shape == (2,)
    assert np.allclose(data.fields["pixels"], [0.0, 1.0])
    assert np.array_equal(data.fields["label"], [[1.0, 0.0], [0.0, 1.0]])


def test_json_source_preserves_vector_fields(tmp_path):
    path = tmp_path / "examples.json"
    path.write_text(
        json.dumps([{"image": [[1, 2], [3, 4]], "label": 7}, {"image": [[5, 6], [7, 8]], "label": 3}]),
        encoding="utf-8",
    )

    data = JsonSource(path).load()

    assert data.schema["image"].sample_shape == (2, 2)
    assert data.fields["label"].tolist() == [7, 3]


def test_image_folder_source_reads_images_and_parent_labels(tmp_path):
    cat = tmp_path / "cat"
    dog = tmp_path / "dog"
    cat.mkdir()
    dog.mkdir()
    Image.fromarray(np.full((2, 3), 20, dtype=np.uint8)).save(cat / "one.png")
    Image.fromarray(np.full((2, 3), 40, dtype=np.uint8)).save(dog / "two.png")

    data = ImageFolderSource(tmp_path).load()

    assert data.schema["image"].sample_shape == (2, 3, 3)
    assert data.fields["label"].tolist() == ["cat", "dog"]
    assert data.fields["filename"].tolist() == ["cat/one.png", "dog/two.png"]


def test_flatten_preserves_examples_and_feed_selects_one(tmp_path):
    path = tmp_path / "images.npy"
    np.save(path, np.arange(8).reshape(2, 2, 2))
    source = NpySource(path, transforms=(Flatten("value"),))
    layer = Layer(lambda value: value, lambda value: np.ones_like(value), from_array=np.zeros(4))

    values = Feed(source, "value", layer, "input").feed(1)

    assert source.data.schema["value"].sample_shape == (4,)
    assert np.array_equal(values, [[4, 5, 6, 7]])
    assert np.array_equal(layer.values, values)


def test_feed_requires_a_flat_feature_field(tmp_path):
    path = tmp_path / "images.npy"
    np.save(path, np.zeros((2, 2, 2)))
    source = NpySource(path)
    layer = Layer(lambda value: value, lambda value: np.ones_like(value), from_array=np.zeros(4))

    with pytest.raises(ValueError, match="Flatten"):
        Feed(source, "value", layer, "input").feed()


def test_dataset_spec_builds_a_source_and_ordered_transform_pipeline(tmp_path):
    path = tmp_path / "animals.csv"
    path.write_text("pixels;label\n0;cat\n255;dog\n", encoding="utf-8")
    spec = DatasetSpec.model_validate({
        "id": "animals",
        "source": {"type": "csv", "path": str(path), "delimiter": ";"},
        "transforms": [
            {"type": "cast", "field": "pixels", "dtype": "float64"},
            {"type": "scale", "field": "pixels", "divisor": 255},
            {"type": "one_hot", "field": "label", "categories": ["cat", "dog"]},
        ],
        "pos": [20, 40],
    })

    source = build_data_source(spec)
    data = source.load()

    assert isinstance(source, CsvSource)
    assert np.allclose(data.fields["pixels"], [0.0, 1.0])
    assert np.array_equal(data.fields["label"], [[1.0, 0.0], [0.0, 1.0]])
    assert spec.model_dump(mode="json")["source"]["type"] == "csv"


def test_data_source_uses_the_shared_node_registry(tmp_path):
    path = tmp_path / "values.npy"
    np.save(path, np.asarray([[1.0, 2.0]]))

    source = DataSource.create("npy", path=path)

    assert isinstance(source, NpySource)
    assert "_registry" not in DataSource.__dict__


def test_transforms_are_built_through_their_registry():
    transform = DataTransform.create("select", fields=("pixels", "label"))

    assert isinstance(transform, SelectFields)
    assert transform.fields == ("pixels", "label")


def test_dataset_spec_rejects_a_zero_scale_divisor(tmp_path):
    with pytest.raises(ValueError, match="divisor"):
        DatasetSpec.model_validate({
            "id": "bad-scale",
            "source": {"type": "npy", "path": str(tmp_path / "values.npy")},
            "transforms": [{"type": "scale", "field": "value", "divisor": 0}],
        })
