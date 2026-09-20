from abc import ABC, abstractmethod
from typing import Any, ClassVar, cast

import numpy as np
import numpy.typing as npt

from .data import DataTable


class DataTransform(ABC):
    """A serialisable-in-principle operation over named, aligned data fields."""

    transform_type: ClassVar[str]
    _transform_registry: ClassVar[dict[str, type["DataTransform"]]] = {}

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        transform_type = getattr(cls, "transform_type", None)
        if not transform_type:
            raise TypeError(f"{cls.__name__} must define a transform type")
        if transform_type in DataTransform._transform_registry:
            raise TypeError(f"Duplicate transform type: {transform_type}")
        DataTransform._transform_registry[transform_type] = cls

    @classmethod
    def create(cls, transform_type: str, **kwargs: Any) -> "DataTransform":
        try:
            transform_cls = cls._transform_registry[transform_type]
        except KeyError as error:
            raise ValueError(f"Unknown transform type: {transform_type}") from error
        concrete_transform = cast(type[DataTransform], transform_cls)
        return concrete_transform(**kwargs)

    @abstractmethod
    def apply(self, data: DataTable) -> DataTable:
        pass


class SelectFields(DataTransform):
    transform_type = "select"

    def __init__(self, fields: tuple[str, ...]) -> None:
        self.fields = fields

    def apply(self, data: DataTable) -> DataTable:
        return data.select(self.fields)


class Flatten(DataTransform):
    transform_type = "flatten"

    def __init__(self, field: str) -> None:
        self.field = field

    def apply(self, data: DataTable) -> DataTable:
        values = _field(data, self.field)
        # Axis zero identifies examples and must survive every field transform.
        return data.replace_field(self.field, values.reshape(len(values), -1))


class Cast(DataTransform):
    transform_type = "cast"

    def __init__(self, field: str, dtype: str | np.dtype) -> None:
        self.field = field
        self.dtype = np.dtype(dtype)

    def apply(self, data: DataTable) -> DataTable:
        return data.replace_field(self.field, _field(data, self.field).astype(self.dtype))


class Scale(DataTransform):
    transform_type = "scale"

    def __init__(self, field: str, divisor: float) -> None:
        if divisor == 0:
            raise ValueError("Scale divisor cannot be zero")
        self.field = field
        self.divisor = divisor

    def apply(self, data: DataTable) -> DataTable:
        return data.replace_field(self.field, _field(data, self.field) / self.divisor)


class OneHot(DataTransform):
    transform_type = "one_hot"

    def __init__(self, field: str, categories: tuple[object, ...] | None = None) -> None:
        self.field = field
        self.categories = categories

    def apply(self, data: DataTable) -> DataTable:
        values = _field(data, self.field)
        if values.ndim != 1:
            raise ValueError(f"OneHot requires a scalar field, got {values.shape}")

        categories = np.asarray(self.categories) if self.categories is not None else np.unique(values)
        if not len(categories):
            raise ValueError("OneHot requires at least one category")
        positions = {value: index for index, value in enumerate(categories.tolist())}
        try:
            indexes: npt.NDArray[np.intp] = np.asarray(
                [positions[value] for value in values.tolist()],
                dtype=np.intp,
            )
        except KeyError as error:
            raise ValueError(f"Unknown category for field {self.field!r}: {error.args[0]!r}") from error

        encoded = np.zeros((len(values), len(categories)), dtype=np.float64)
        rows: npt.NDArray[np.intp] = np.arange(len(values), dtype=np.intp)
        encoded[rows, indexes] = 1.0
        return data.replace_field(self.field, encoded)


def _field(data: DataTable, name: str) -> np.ndarray:
    try:
        return data.fields[name]
    except KeyError as error:
        raise ValueError(f"Unknown field: {name}") from error
