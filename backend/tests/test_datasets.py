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
    build_feed,
)
from neuralsandbox.schemas import DatasetSpec, FeedSpec


def test_csv_strings_one_hot_without_mapping(tmp_path):
    path = tmp_path / "levels.csv"
    path.write_text("level\nmid\nlow\nhigh\nmid\n", encoding="utf-8")
    spec = FeedSpec.model_validate({
        "id": "levels", "dataset": "csv", "layer": "input", "field": "level",
        "transforms": [{"type": "one_hot", "field": "level"}],
    })
    layer = Layer(lambda value: value, lambda value: np.ones_like(value), from_array=np.zeros(3))
    feed = build_feed(spec, CsvSource(path), layer)
    data = feed.load()
    # Sorted unique categories: high -> 0, low -> 1, mid -> 2.
    assert data.fields["level"].tolist() == [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]]
    assert data.schema["level"].sample_shape == (3,)
    assert feed.feed(1).tolist() == [[0, 1, 0]]


@pytest.mark.parametrize("values", [[10, -2, 10, 7], ["z", "a", "z", "m"]])
def test_one_hot_infers_deterministic_unique_categories(values):
    from neuralsandbox.datasets.data import DataTable

    data = DataTable({"category": np.array(values)})
    encoded = OneHot("category").apply(data)
    assert encoded.fields["category"].tolist() == [[0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert data.fields["category"].tolist() == values


def test_csv_mapping_then_one_hot_round_trip(tmp_path):
    path = tmp_path / "levels.csv"
    path.write_text("level\nhigh\nlow\nmid\nlow\n", encoding="utf-8")
    spec = FeedSpec.model_validate({
        "id": "levels", "dataset": "csv", "layer": "input", "field": "level",
        "transforms": [
            {"type": "mapping", "field": "level", "mapping": {"low": 0, "mid": 1, "high": 2}},
            {"type": "one_hot", "field": "level"},
        ],
    })
    restored = FeedSpec.model_validate_json(spec.model_dump_json())
    data = build_feed(restored, CsvSource(path)).load()
    assert data.fields["level"].tolist() == [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]]


def test_mapping_preserves_shape_and_other_fields():
    from neuralsandbox.datasets import Mapping
    from neuralsandbox.datasets.data import DataTable

    data = DataTable({"words": np.array([["", "low"], ["low", ""]]), "other": np.array([1, 2])})
    result = Mapping("words", {"": -0.5, "low": 2.25}).apply(data)
    assert result.fields["words"].tolist() == [[-0.5, 2.25], [2.25, -0.5]]
    assert result.fields["other"] is data.fields["other"]
    assert data.fields["words"][0, 1] == "low"
    with pytest.raises(ValueError, match="Unmapped value.*words.*low"):
        Mapping("words", {"": 0}).apply(data)


@pytest.mark.parametrize("mapping", [{}, {"low": "0"}, {"low": True}, {"low": float("inf")}])
def test_mapping_rejects_invalid_pairs(mapping):
    from neuralsandbox.schemas import MappingSpec
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MappingSpec(field="level", mapping=mapping)


def test_csv_feed_applies_transforms_to_named_fields(tmp_path):
    path = tmp_path / "animals.csv"
    path.write_text("pixels,label\n0,cat\n255,dog\n", encoding="utf-8")

    data = Feed(
        CsvSource(path), "pixels", None, "input",
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
    source = NpySource(path)
    layer = Layer(lambda value: value, lambda value: np.ones_like(value), from_array=np.zeros(4))

    values = Feed(source, "value", layer, "input", (Flatten("value"),)).feed(1)

    assert source.data.schema["value"].sample_shape == (2, 2)
    assert np.array_equal(values, [[4, 5, 6, 7]])
    assert np.array_equal(layer.values, values)


def test_feed_requires_a_flat_feature_field(tmp_path):
    path = tmp_path / "images.npy"
    np.save(path, np.zeros((2, 2, 2)))
    source = NpySource(path)
    layer = Layer(lambda value: value, lambda value: np.ones_like(value), from_array=np.zeros(4))

    with pytest.raises(ValueError, match="Flatten"):
        Feed(source, "value", layer, "input").feed()


def test_feed_spec_builds_an_ordered_transform_pipeline(tmp_path):
    path = tmp_path / "animals.csv"
    path.write_text("pixels;label\n0;cat\n255;dog\n", encoding="utf-8")
    spec = DatasetSpec.model_validate({
        "id": "animals",
        "source": {"type": "csv", "path": str(path), "delimiter": ";"},
        "pos": [20, 40],
    })

    source = build_data_source(spec)
    feed_spec = FeedSpec.model_validate({
        "id": "pixels", "dataset": "animals", "layer": "input", "field": "pixels",
        "transforms": [
            {"type": "cast", "field": "pixels", "dtype": "float64"},
            {"type": "scale", "field": "pixels", "divisor": 255},
            {"type": "one_hot", "field": "label", "categories": ["cat", "dog"]},
        ],
    })
    data = build_feed(feed_spec, source).load()

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


def test_feed_spec_rejects_a_zero_scale_divisor(tmp_path):
    with pytest.raises(ValueError, match="divisor"):
        FeedSpec.model_validate({
            "id": "bad-scale", "dataset": "data", "layer": "input", "field": "value",
            "transforms": [{"type": "scale", "field": "value", "divisor": 0}],
        })
