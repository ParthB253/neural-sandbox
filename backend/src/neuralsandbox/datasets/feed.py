from ..core import Layer
from .data_source import DataSource
from .data import DataTable, FieldSchema
from .transforms import DataTransform


class Feed:
    """Bind one transformed field to an input layer or a training target."""

    def __init__(self, source: DataSource, field: str, layer: Layer | None, role: str,
                 transforms: tuple[DataTransform, ...] = ()) -> None:
        if role not in {"input", "target"}:
            raise ValueError("Feed role must be 'input' or 'target'")
        self.source = source
        self.field = field
        self.layer = layer
        self.role = role
        self.transforms = transforms
        self.data: DataTable | None = None

    def load(self) -> DataTable:
        data = self.source.data if self.source.data is not None else self.source.load()
        for transform in self.transforms:
            data = transform.apply(data)
        self.data = data
        return data

    @property
    def schema(self) -> dict[str, FieldSchema]:
        return (self.data if self.data is not None else self.load()).schema

    @property
    def sample_shape(self) -> tuple[int, ...]:
        try:
            return self.schema[self.field].sample_shape
        except KeyError as error:
            raise ValueError(f"Unknown field: {self.field}") from error

    @property
    def feature_count(self) -> int:
        shape = self.sample_shape
        if len(shape) > 1:
            raise ValueError(f"Field {self.field!r} has sample shape {shape}; apply Flatten")
        return shape[0] if shape else 1

    def feed(self, sample_index: int = 0):
        if self.layer is None:
            raise ValueError("Feed must be bound to a runtime layer before execution")
        data = self.data if self.data is not None else self.load()
        try:
            values = data.fields[self.field]
        except KeyError as error:
            raise ValueError(f"Unknown field: {self.field}") from error
        if values.ndim not in (1, 2):
            raise ValueError(
                f"Field {self.field!r} has sample shape {values.shape[1:]}; "
                "apply a transform such as Flatten before feeding it to a layer"
            )
        if sample_index < 0:
            raise IndexError(f"Sample index out of range: {sample_index}")
        sample = values[sample_index : sample_index + 1]
        if len(sample) != 1:
            raise IndexError(f"Sample index out of range: {sample_index}")
        sample = sample.reshape(1, -1)
        if sample.shape[1] != self.layer.size():
            raise ValueError(f"Feed size {sample.shape[1]} does not match layer: {self.layer}")
        if self.role == "input":
            self.layer.values = sample
        return sample
