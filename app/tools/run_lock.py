"""Cross-process locking for mutations scoped to one analysis run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from filelock import FileLock, Timeout

from app.backend.services.run_manager import RunManager
from app.tools.file_utils import ensure_directory


class RunMutationConflictError(RuntimeError):
    """Raised when another request already owns a run's mutation lock."""


@contextmanager
def acquire_run_mutation_lock(
    run_manager: RunManager,
    run_id: str,
    timeout: float = 0,
) -> Iterator[None]:
    """Acquire a cross-thread and cross-process mutation lock for one run."""

    paths = run_manager.get_paths(run_id)
    if not paths.root.exists():
        yield
        return

    ensure_directory(paths.logs)
    lock = FileLock(paths.logs / ".mutation.lock", timeout=timeout)
    try:
        with lock:
            yield
    except Timeout as exc:
        raise RunMutationConflictError(
            f"Run '{run_id}' is already being modified by another request."
        ) from exc
