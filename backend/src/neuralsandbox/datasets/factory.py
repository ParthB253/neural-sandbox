"""Build dataset runtime objects from their API wire format."""

from ..schemas import DatasetSpec, FeedSpec, TransformSpec
from .data_source import DataSource
from .transforms import DataTransform
from .feed import Feed
from ..core import Layer


def build_data_source(spec: DatasetSpec) -> DataSource:
    source_config = spec.source.model_dump(exclude={"type"})
    return DataSource.create(spec.source.type, **source_config)


def build_feed(spec: FeedSpec, source: DataSource, layer: Layer | None = None) -> Feed:
    transforms = tuple(_build_transform(transform) for transform in spec.transforms)
    return Feed(source, spec.field, layer, spec.role, transforms)


def _build_transform(spec: TransformSpec) -> DataTransform:
    transform_config = spec.model_dump(exclude={"type"})
    return DataTransform.create(spec.type, **transform_config)
