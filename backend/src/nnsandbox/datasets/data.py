"""The format shared by all data sources and transforms."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSchema:
    """Metadata for one value per dataset example."""

    dtype: str
    sample_shape: tuple[int, ...]


@dataclass
class DataTable:
    """Aligned named arrays, with axis zero reserved for examples.

    A CSV's columns and an image folder's image/label pairs therefore have the
    same representation without making pandas part of the training runtime.
    """

    fields: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("A data source must produce at least one field")

        sample_count: int | None = None
        for name, values in self.fields.items():
            if not name:
                raise ValueError("Field names cannot be empty")
            if not isinstance(values, np.ndarray) or values.ndim == 0:
                raise ValueError(f"Field {name!r} must be an array with a sample axis")
            if sample_count is None:
                sample_count = len(values)
            elif len(values) != sample_count:
                raise ValueError("All fields must contain the same number of examples")

    @property
    def sample_count(self) -> int:
        return len(next(iter(self.fields.values())))

    @property
    def schema(self) -> dict[str, FieldSchema]:
        return {
            name: FieldSchema(str(values.dtype), values.shape[1:])
            for name, values in self.fields.items()
        }

    def replace_field(self, name: str, values: np.ndarray) -> "DataTable":
        fields = dict(self.fields)
        fields[name] = values
        return DataTable(fields)

    def select(self, names: tuple[str, ...]) -> "DataTable":
        missing = set(names) - self.fields.keys()
        if missing:
            raise ValueError(f"Unknown fields: {', '.join(sorted(missing))}")
        return DataTable({name: self.fields[name] for name in names})
