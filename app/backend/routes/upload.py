"""CSV upload endpoint."""

from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

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

    raw_path = paths.input / "raw_data.csv"
    try:
        with raw_path.open("wb") as output_file:
            shutil.copyfileobj(file.file, output_file)
    finally:
        await file.close()

    try:
        dataframe = load_csv(raw_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read CSV file: {exc}") from exc

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
