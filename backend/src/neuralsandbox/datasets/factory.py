"""Build dataset runtime objects from their API wire format."""

from ..schemas import DatasetSpec, TransformSpec
from .data_source import DataSource
from .transforms import DataTransform


def build_data_source(spec: DatasetSpec) -> DataSource:
    transforms = tuple(_build_transform(transform) for transform in spec.transforms)
    source_config = spec.source.model_dump(exclude={"type"})
    return DataSource.create(spec.source.type, transforms=transforms, **source_config)


def _build_transform(spec: TransformSpec) -> DataTransform:
    transform_config = spec.model_dump(exclude={"type"})
    return DataTransform.create(spec.type, **transform_config)
