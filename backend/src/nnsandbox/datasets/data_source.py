from abc import ABC, abstractmethod
import csv
import json
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from PIL import Image

from ..nodes import Node, NodeConfig, NodeInfo, NodePort
from .data import DataTable
from .transforms import DataTransform


class DataSource(Node, ABC):
    source_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs) -> None:
        source_type = getattr(cls, "source_type", None)
        if not source_type:
            raise TypeError(f"{cls.__name__} must define a source type")
        info = cls.node_info
        expected_node_type = f"dataset.{source_type}"
        if info is None or info.type != expected_node_type:
            raise TypeError(
                f"{cls.__name__}.node_info.type must be {expected_node_type!r}"
            )
        super().__init_subclass__(**kwargs)

    def __init__(self, transforms: tuple[DataTransform, ...] = ()) -> None:
        self.transforms = transforms
        self.data: DataTable | None = None

    @classmethod
    def create(cls, source_type: str, **kwargs: Any) -> "DataSource":
        try:
            source_cls = Node.class_for(f"dataset.{source_type}")
        except KeyError as error:
            raise ValueError(f"Unknown data source type: {source_type}") from error
        if not issubclass(source_cls, DataSource):
            raise TypeError(f"Registered node is not a data source: {source_type}")
        data_source_cls = cast(type[DataSource], source_cls)
        return data_source_cls(**kwargs)

    def load(self) -> DataTable:
        data = self._load()
        for transform in self.transforms:
            data = transform.apply(data)
        self.data = data
        return data

    @abstractmethod
    def _load(self) -> DataTable:
        """Read raw source data into aligned named arrays."""


class IdxSource(DataSource):
    source_type = "idx"
    node_info = NodeInfo(
        type="dataset.idx",
        label="IDX dataset",
        category="dataset",
        description="Load one array from an IDX file.",
        config=(NodeConfig("path", "path", description="Server-side IDX file"),),
        outputs=(NodePort("fields", "One output per loaded field", dynamic=True),),
    )
    DTYPE_TABLE = {
        0x08: np.dtype(">u1"),
        0x09: np.dtype(">i1"),
        0x0B: np.dtype(">i2"),
        0x0C: np.dtype(">i4"),
        0x0D: np.dtype(">f4"),
        0x0E: np.dtype(">f8"),
    }

    def __init__(self, path: str | Path, transforms: tuple[DataTransform, ...] = ()) -> None:
        super().__init__(transforms)
        self.path = Path(path)

    def _load(self) -> DataTable:
        raw = self.path.read_bytes()
        if len(raw) < 4 or raw[0:2] != b"\x00\x00":
            raise ValueError(f"{self.path} is not a valid IDX file")
        try:
            dtype = self.DTYPE_TABLE[raw[2]]
        except KeyError as error:
            raise ValueError(f"Unsupported IDX data type: {raw[2]:#x}") from error
        dimensions = raw[3]
        header_size = 4 + 4 * dimensions
        if dimensions == 0 or len(raw) < header_size:
            raise ValueError(f"{self.path} has an incomplete IDX header")
        shape = tuple(np.frombuffer(raw, dtype=">u4", count=dimensions, offset=4))
        values = np.frombuffer(raw, dtype=dtype, offset=header_size).reshape(shape)
        return DataTable({"value": values})


class NpySource(DataSource):
    source_type = "npy"
    node_info = NodeInfo(
        type="dataset.npy",
        label="NumPy dataset",
        category="dataset",
        description="Load one array from a .npy file.",
        config=(NodeConfig("path", "path", description="Server-side .npy file"),),
        outputs=(NodePort("fields", "One output per loaded field", dynamic=True),),
    )

    def __init__(self, path: str | Path, transforms: tuple[DataTransform, ...] = ()) -> None:
        super().__init__(transforms)
        self.path = Path(path)

    def _load(self) -> DataTable:
        values = np.load(self.path, allow_pickle=False)
        if not isinstance(values, np.ndarray):
            raise ValueError("NpySource only accepts a single NumPy array")
        return DataTable({"value": values})


class CsvSource(DataSource):
    source_type = "csv"
    node_info = NodeInfo(
        type="dataset.csv",
        label="CSV dataset",
        category="dataset",
        description="Load named fields from a CSV header and its rows.",
        config=(
            NodeConfig("path", "path", description="Server-side CSV file"),
            NodeConfig("delimiter", "string", False, ","),
            NodeConfig("quotechar", "string", False, '"'),
        ),
        outputs=(NodePort("fields", "One output per CSV column", dynamic=True),),
    )

    def __init__(self, path: str | Path, transforms: tuple[DataTransform, ...] = (), **csv_options) -> None:
        super().__init__(transforms)
        self.path = Path(path)
        self.csv_options = csv_options

    def _load(self) -> DataTable:
        with self.path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, **self.csv_options)
            if not reader.fieldnames:
                raise ValueError("CSV source requires a header row")
            rows = list(reader)

        return DataTable({name: np.asarray([row[name] for row in rows]) for name in reader.fieldnames})


class JsonSource(DataSource):
    source_type = "json"
    node_info = NodeInfo(
        type="dataset.json",
        label="JSON dataset",
        category="dataset",
        description="Load named fields from an array of JSON objects.",
        config=(NodeConfig("path", "path", description="Server-side JSON file"),),
        outputs=(NodePort("fields", "One output per object field", dynamic=True),),
    )

    def __init__(self, path: str | Path, transforms: tuple[DataTransform, ...] = ()) -> None:
        super().__init__(transforms)
        self.path = Path(path)

    def _load(self) -> DataTable:
        with self.path.open(encoding="utf-8") as file:
            records = json.load(file)
        if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
            raise ValueError("JSON source must be a list of object records")
        if not records:
            raise ValueError("JSON source cannot be empty because its fields are unknown")

        names = tuple(dict.fromkeys(name for record in records for name in record))
        return DataTable({name: np.asarray([record.get(name) for record in records]) for name in names})


class ImageFolderSource(DataSource):
    source_type = "image_folder"
    node_info = NodeInfo(
        type="dataset.image_folder",
        label="Image folder",
        category="dataset",
        description="Load images and derive labels from their parent folders.",
        config=(
            NodeConfig("path", "path", description="Server-side image directory"),
            NodeConfig("extensions", "string[]", False, (".bmp", ".jpeg", ".jpg", ".png")),
            NodeConfig("mode", "string", False, "RGB"),
        ),
        outputs=(NodePort("fields", "Image, label, and filename fields", dynamic=True),),
    )

    def __init__(
        self,
        path: str | Path,
        transforms: tuple[DataTransform, ...] = (),
        extensions: tuple[str, ...] = (".bmp", ".jpeg", ".jpg", ".png"),
        mode: str = "RGB",
    ) -> None:
        super().__init__(transforms)
        self.path = Path(path)
        self.extensions = tuple(extension.lower() for extension in extensions)
        self.mode = mode

    def _load(self) -> DataTable:
        files = sorted(path for path in self.path.rglob("*") if path.suffix.lower() in self.extensions)
        if not files:
            raise ValueError(f"No supported images found under {self.path}")

        images = []
        labels = []
        filenames = []
        for path in files:
            with Image.open(path) as image:
                images.append(np.asarray(image.convert(self.mode)))
            relative = path.relative_to(self.path)
            labels.append(relative.parts[0] if len(relative.parts) > 1 else "")
            filenames.append(str(relative))

        try:
            image_values = np.stack(images)
        except ValueError as error:
            raise ValueError("All images in an image folder must have the same dimensions") from error
        return DataTable({
            "image": image_values,
            "label": np.asarray(labels),
            "filename": np.asarray(filenames),
        })
