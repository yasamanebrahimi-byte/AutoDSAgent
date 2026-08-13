"""Shared interface and helpers for deterministic workflow agents."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from app.workflows.workflow_state import relative_to_run, set_artifact


class BaseAgent:
    """Base class for deterministic workflow agent wrappers."""

    name = "BaseAgent"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the agent against workflow state."""

        raise NotImplementedError

    def _copy_state(self, state: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(state)

    def _require_run_id(self, state: dict[str, Any]) -> str:
        run_id = state.get("run_id")
        if not run_id:
            raise ValueError(f"{self.name} requires state['run_id'].")
        return str(run_id)

    def _set_artifact(
        self,
        state: dict[str, Any],
        key: str,
        path: str | Path,
        run_root: str | Path,
    ) -> None:
        set_artifact(state, key, relative_to_run(path, run_root))
