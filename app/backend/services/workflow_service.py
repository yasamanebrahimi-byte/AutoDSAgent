"""Backend service for automated workflow orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.agents.orchestrator import OrchestratorAgent
from app.backend.schemas.workflow import (
    WorkflowApprovalRequest,
    WorkflowRetryRequest,
    WorkflowStartRequest,
)
from app.backend.services.run_manager import RunManager
from app.tools.app_logging import get_logger, log_event
from app.tools.trace_logger import TraceLogger
from app.workflows.analysis_graph import build_analysis_graph
from app.workflows.workflow_state import (
    create_initial_workflow_state,
    load_workflow_state,
    save_workflow_state,
    state_path_for_logs_dir,
)


class WorkflowService:
    """Start, inspect, approve, and retry deterministic workflows."""

    def __init__(
        self,
        run_manager: RunManager | None = None,
        trace_logger: TraceLogger | None = None,
        orchestrator: OrchestratorAgent | None = None,
    ) -> None:
        self.run_manager = run_manager or RunManager()
        self.trace_logger = trace_logger or TraceLogger(self.run_manager)
        self.orchestrator = orchestrator or OrchestratorAgent(
            build_analysis_graph(self.run_manager, self.trace_logger)
        )
        self.logger = get_logger(__name__)

    def start_workflow(
        self,
        run_id: str,
        request: WorkflowStartRequest,
    ) -> dict[str, Any]:
        """Initialize and run a workflow until completion, failure, or approval."""

        self._validate_run(run_id)
        self.trace_logger.reset(run_id)
        state = create_initial_workflow_state(
            run_id=run_id,
            target_column=request.target_column,
            task_type=request.task_type,
            require_cleaning_approval=request.require_cleaning_approval,
            require_modeling_approval=request.require_modeling_approval,
        )
        save_workflow_state(self.workflow_state_path(run_id), state)
        self.trace_logger.append_event(
            run_id=run_id,
            agent="OrchestratorAgent",
            step=None,
            event_type="workflow_started",
            message="Automated workflow started.",
            details={
                "target_column": request.target_column,
                "task_type": request.task_type,
                "require_cleaning_approval": request.require_cleaning_approval,
                "require_modeling_approval": request.require_modeling_approval,
            },
        )
        log_event(
            self.logger,
            logging.INFO,
            "Workflow started.",
            run_id=run_id,
            target_column=request.target_column or "none",
        )
        updated_state = self.orchestrator.run(state)
        log_event(
            self.logger,
            logging.INFO,
            "Workflow stopped.",
            run_id=run_id,
            status=updated_state.get("status"),
            current_step=updated_state.get("current_step") or "none",
        )
        return updated_state

    def get_state(self, run_id: str) -> dict[str, Any]:
        """Return current workflow state."""

        self._validate_run(run_id)
        path = self.workflow_state_path(run_id)
        if not path.exists():
            raise FileNotFoundError(path)
        return load_workflow_state(path)

    def get_trace(self, run_id: str) -> list[dict[str, Any]]:
        """Return current trace events."""

        self._validate_run(run_id)
        return self.trace_logger.load_events(run_id)

    def apply_approval(
        self,
        run_id: str,
        request: WorkflowApprovalRequest,
    ) -> dict[str, Any]:
        """Apply approval/rejection and continue if appropriate."""

        state = self.get_state(run_id)
        updated_state = self.orchestrator.apply_approval(
            state,
            request.step,
            request.action.value,
        )
        log_event(
            self.logger,
            logging.INFO,
            "Workflow approval applied.",
            run_id=run_id,
            step=request.step,
            action=request.action.value,
            status=updated_state.get("status"),
        )
        return updated_state

    def retry_step(
        self,
        run_id: str,
        request: WorkflowRetryRequest,
    ) -> dict[str, Any]:
        """Retry a failed workflow step."""

        state = self.get_state(run_id)
        updated_state = self.orchestrator.retry_step(state, request.step)
        log_event(
            self.logger,
            logging.INFO,
            "Workflow step retry requested.",
            run_id=run_id,
            step=request.step,
            status=updated_state.get("status"),
        )
        return updated_state

    def reset_workflow(self, run_id: str) -> dict[str, str]:
        """Safely reset workflow logs without touching raw or derived data artifacts."""

        self._validate_run(run_id)
        state_path = self.workflow_state_path(run_id)
        if state_path.exists():
            state_path.unlink()
        self.trace_logger.reset(run_id)
        log_event(
            self.logger,
            logging.INFO,
            "Workflow reset.",
            run_id=run_id,
        )
        return {"status": "reset", "run_id": run_id}

    def workflow_state_path(self, run_id: str) -> Path:
        """Return the workflow state path for one run."""

        return state_path_for_logs_dir(self.run_manager.get_paths(run_id).logs)

    def _validate_run(self, run_id: str) -> None:
        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)
        raw_path = paths.input / "raw_data.csv"
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)
