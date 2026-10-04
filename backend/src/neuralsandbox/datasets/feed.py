from ..core import Layer
from .data_source import DataSource


class Feed:
    """Bind one transformed field to an input layer or a training target."""

    def __init__(self, source: DataSource, field: str, layer: Layer, role: str) -> None:
        if role not in {"input", "target"}:
            raise ValueError("Feed role must be 'input' or 'target'")
        self.source = source
        self.field = field
        self.layer = layer
        self.role = role

    def feed(self, sample_index: int = 0):
        data = self.source.data or self.source.load()
        try:
            values = data.fields[self.field]
        except KeyError as error:
            raise ValueError(f"Unknown field: {self.field}") from error
        if values.ndim != 2:
            raise ValueError(
                f"Field {self.field!r} has sample shape {values.shape[1:]}; "
                "apply a transform such as Flatten before feeding it to a layer"
            )
        sample = values[sample_index : sample_index + 1]
        if len(sample) != 1:
            raise IndexError(f"Sample index out of range: {sample_index}")
        if sample.shape[1] != self.layer.size():
            raise ValueError(f"Feed size {sample.shape[1]} does not match layer: {self.layer}")
        if self.role == "input":
            self.layer.values = sample
        return sample
