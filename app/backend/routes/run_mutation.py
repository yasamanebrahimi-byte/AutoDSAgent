"""HTTP mapping for per-run mutation lock conflicts."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from app.backend.services.run_manager import RunManager
from app.tools.run_lock import RunMutationConflictError, acquire_run_mutation_lock


@contextmanager
def locked_run_mutation(run_manager: RunManager, run_id: str) -> Iterator[None]:
    """Acquire a run lock or return a retryable HTTP conflict."""

    try:
        with acquire_run_mutation_lock(run_manager, run_id):
            yield
    except RunMutationConflictError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
