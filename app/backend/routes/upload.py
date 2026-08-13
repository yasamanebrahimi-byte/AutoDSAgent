"""CSV upload endpoint."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.backend.config import settings
from app.backend.schemas.dataset import DatasetMetadata
from app.backend.services.dataset_service import (
    generate_dataset_metadata,
    load_csv,
    validate_csv_filename,
)
from app.backend.services.run_manager import RunManager
from app.tools.app_logging import get_logger, log_event


router = APIRouter(tags=["upload"])
run_manager = RunManager()
logger = get_logger(__name__)
UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("/upload", response_model=DatasetMetadata)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetMetadata:
    """Accept a CSV upload, preserve it, profile basic metadata, and return it."""

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="A CSV file is required.")

    try:
        validate_csv_filename(file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_id = run_manager.generate_run_id()

    try:
        paths = run_manager.create_run(run_id)
    except FileExistsError as exc:
        raise HTTPException(status_code=500, detail="Could not create a unique run.") from exc

    try:
        raw_path = paths.input / "raw_data.csv"
        await _save_upload_with_size_limit(
            file=file,
            raw_path=raw_path,
            max_size_mb=settings.max_upload_size_mb,
        )
        dataframe = load_csv(raw_path)
    except Exception as exc:
        run_manager.delete_run(run_id)
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=400, detail=f"Unable to read CSV file: {exc}") from exc
    finally:
        await file.close()

    metadata = generate_dataset_metadata(
        dataframe=dataframe,
        filename=file.filename,
        run_id=run_id,
    )
    run_manager.save_metadata(run_id, metadata.model_dump(mode="json"))
    log_event(
        logger,
        logging.INFO,
        "Dataset uploaded.",
        run_id=run_id,
        filename=file.filename,
        rows=metadata.rows,
        columns=metadata.columns,
    )

    return metadata


async def _save_upload_with_size_limit(
    file: UploadFile,
    raw_path: Path,
    max_size_mb: int,
) -> int:
    """Stream an upload to disk while enforcing the configured size limit."""

    max_bytes = int(max_size_mb) * 1024 * 1024
    bytes_written = 0

    with raw_path.open("wb") as output_file:
        while True:
            chunk = await file.read(UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            bytes_written += len(chunk)
            if bytes_written > max_bytes:
                output_file.close()
                if raw_path.exists():
                    raw_path.unlink()
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded CSV exceeds the {max_size_mb} MB limit.",
                )
            output_file.write(chunk)

    return bytes_written
