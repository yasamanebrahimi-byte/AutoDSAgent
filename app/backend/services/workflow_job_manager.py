"""In-process background job management for long-running workflows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from typing import Any
from uuid import uuid4

from app.workflows.workflow_state import utc_now_iso


WorkflowTask = Callable[[], Mapping[str, Any]]


class WorkflowJobManager:
    """Submit workflows to a bounded worker pool and expose pollable status."""

    def __init__(self, max_workers: int = 2, max_history: int = 200) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="autods-workflow",
        )
        self._max_history = max_history
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def submit(self, run_id: str, task: WorkflowTask) -> dict[str, Any]:
        """Queue one workflow task and return its initial job record."""

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "run_id": run_id,
            "status": "queued",
            "submitted_at": utc_now_iso(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "status_url": f"/runs/{run_id}/workflow/jobs/{job_id}",
            "state_url": f"/runs/{run_id}/workflow/state",
        }
        with self._lock:
            self._prune_history()
            self._jobs[job_id] = job
        self._executor.submit(self._execute, job_id, task)
        return deepcopy(job)

    def get(self, run_id: str, job_id: str) -> dict[str, Any]:
        """Return one job owned by the requested run."""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["run_id"] != run_id:
                raise KeyError(job_id)
            return deepcopy(job)

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting jobs and optionally wait for active work."""

        self._executor.shutdown(wait=wait, cancel_futures=False)

    def _execute(self, job_id: str, task: WorkflowTask) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "running"
            job["started_at"] = utc_now_iso()

        try:
            state = task()
        except Exception as exc:
            with self._lock:
                job = self._jobs[job_id]
                job["status"] = "failed"
                job["error"] = str(exc)
                job["completed_at"] = utc_now_iso()
            return

        terminal_status = str(state.get("status", "completed"))
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = "failed" if terminal_status == "failed" else "completed"
            if terminal_status == "failed":
                errors = state.get("errors") or []
                job["error"] = (
                    str(errors[-1].get("message"))
                    if errors and isinstance(errors[-1], dict)
                    else "Workflow execution failed."
                )
            job["completed_at"] = utc_now_iso()

    def _prune_history(self) -> None:
        if len(self._jobs) < self._max_history:
            return
        terminal_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job["status"] in {"completed", "failed"}
        ]
        terminal_ids.sort(key=lambda job_id: self._jobs[job_id]["submitted_at"])
        for job_id in terminal_ids[: max(1, len(self._jobs) - self._max_history + 1)]:
            self._jobs.pop(job_id, None)
