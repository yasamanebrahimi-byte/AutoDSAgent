"""Append-only trace logging for deterministic agent workflow events."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.backend.services.run_manager import RunManager
from app.tools.file_utils import load_json, save_json
from app.workflows.workflow_state import utc_now_iso


TRACE_FILENAME = "agent_trace.json"


class TraceLogger:
    """Persist auditable workflow events in chronological order."""

    def __init__(self, run_manager: RunManager | None = None) -> None:
        self.run_manager = run_manager or RunManager()

    def trace_path(self, run_id: str) -> Path:
        """Return the trace file path for one run."""

        return self.run_manager.get_paths(run_id).logs / TRACE_FILENAME

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        """Load trace events, returning an empty list when no trace exists."""

        path = self.trace_path(run_id)
        if not path.exists():
            return []
        payload = load_json(path)
        if isinstance(payload, list):
            return payload
        return list(payload.get("events", []))

    def reset(self, run_id: str) -> None:
        """Clear trace events for a new workflow start."""

        self._write_events(run_id, [])

    def append_event(
        self,
        run_id: str,
        agent: str,
        step: str | None,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one event to the trace log and return it."""

        event = {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "agent": agent,
            "step": step,
            "event_type": event_type,
            "message": message,
            "details": details or {},
        }
        events = self.load_events(run_id)
        events.append(event)
        self._write_events(run_id, events)
        return event

    def _write_events(self, run_id: str, events: list[dict[str, Any]]) -> None:
        path = self.trace_path(run_id)
        save_json(path, events)
