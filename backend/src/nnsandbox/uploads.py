"""Reusable project files. Keep bytes on disk; never trust an uploaded path."""
import json
import os
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, UploadFile, status
from pydantic import BaseModel

router = APIRouter()
UPLOAD_DIR = Path(os.environ.get("NNSANDBOX_UPLOAD_DIR", Path(__file__).resolve().parents[3] / "workspaces" / "files"))
MAX_BYTES = 100 * 1024 * 1024


class StoredFile(BaseModel):
    id: str
    name: str
    size: int
    path: str


def _metadata(file_id: str) -> Path:
    try:
        UUID(file_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Unknown file") from None
    return UPLOAD_DIR / f"{file_id}.json"


@router.get("/files")
def list_files() -> list[StoredFile]:
    if not UPLOAD_DIR.exists():
        return []
    return [StoredFile.model_validate_json(path.read_text()) for path in sorted(UPLOAD_DIR.glob("*.json"))]


@router.post("/files", status_code=status.HTTP_201_CREATED)
async def upload_file(file: UploadFile) -> StoredFile:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid4())
    name = (file.filename or "untitled").replace("\\", "/").split("/")[-1] or "untitled"
    if name in {".", ".."}:
        name = "untitled"
    # A separate directory preserves the original extension, including IDX names.
    folder = UPLOAD_DIR / file_id
    folder.mkdir()
    path = folder / name
    size = 0
    try:
        with path.open("wb") as dest:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail="Files must be 100 MB or smaller")
                dest.write(chunk)
        if size == 0:
            raise HTTPException(status_code=422, detail="Choose a non-empty file")
        stored = StoredFile(id=file_id, name=name, size=size, path=str(path.resolve()))
        _metadata(file_id).write_text(json.dumps(stored.model_dump()))
        return stored
    except Exception:
        path.unlink(missing_ok=True)
        folder.rmdir()
        raise
    finally:
        await file.close()


@router.delete("/files/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_file(file_id: str) -> None:
    from .api import get_project

    metadata = _metadata(file_id)
    if not metadata.exists():
        raise HTTPException(status_code=404, detail="Unknown file")
    stored = StoredFile.model_validate_json(metadata.read_text())
    if any(str(dataset.source.path) == stored.path for dataset in get_project().datasets):
        raise HTTPException(status_code=409, detail="Remove data source nodes using this file before deleting it")
    Path(stored.path).unlink(missing_ok=True)
    (UPLOAD_DIR / file_id).rmdir()
    metadata.unlink()
