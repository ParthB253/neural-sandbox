"""Disk snapshots of network configuration; uploaded files remain shared resources."""
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from fastapi import HTTPException
from pydantic import BaseModel

from .schemas import ProjectSpec

NETWORK_DIR = Path(os.environ.get(
    "NEURALSANDBOX_NETWORK_DIR", Path(__file__).resolve().parents[3] / "workspaces" / "networks"
))


class NetworkSummary(BaseModel):
    id: UUID
    title: str
    updated_at: datetime
    layers: int
    data_sources: int


def read_network(network_id: UUID) -> ProjectSpec:
    path = NETWORK_DIR / f"{network_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Saved network not found")
    return ProjectSpec.model_validate_json(path.read_text())


def list_networks() -> list[NetworkSummary]:
    if not NETWORK_DIR.exists():
        return []
    result = []
    for path in NETWORK_DIR.glob("*.json"):
        project = ProjectSpec.model_validate_json(path.read_text())
        result.append(NetworkSummary(
            id=project.network.id,
            title=project.network.title,
            updated_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc),
            layers=len(project.network.layers),
            data_sources=len(project.datasets),
        ))
    return sorted(result, key=lambda item: item.updated_at, reverse=True)


def write_network(project: ProjectSpec) -> None:
    """Replace one complete snapshot atomically, retaining the old save on failure."""
    temporary = None
    try:
        NETWORK_DIR.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", dir=NETWORK_DIR, suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            file.write(project.model_dump_json(indent=2))
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(NETWORK_DIR / f"{project.network.id}.json")
    except OSError as error:
        raise HTTPException(status_code=503, detail="Could not save the network to disk. Please retry.") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def file_is_referenced(file_path: str) -> bool:
    """Protect shared files even when the network using them is not open."""
    if not NETWORK_DIR.exists():
        return False
    for path in NETWORK_DIR.glob("*.json"):
        project = ProjectSpec.model_validate_json(path.read_text())
        if any(str(dataset.source.path) == file_path for dataset in project.datasets):
            return True
    return False
