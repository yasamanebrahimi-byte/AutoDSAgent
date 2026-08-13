"""Deterministic analysis state machine.

The project can swap this for LangGraph or an LLM-backed agent runtime later.
For now, the graph is intentionally local, auditable, and dependency-free.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.agents.cleaning_agent import CleaningAgent
from app.agents.eda_agent import EDAAgent
from app.agents.modeling_agent import ModelingAgent
from app.agents.profiler_agent import ProfilerAgent
from app.agents.report_agent import ReportAgent
from app.backend.services.cleaning_service import CleaningService
from app.backend.services.eda_service import EDAService
from app.backend.services.evaluation_service import EvaluationService
from app.backend.services.modeling_service import ModelingService
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.report_service import ReportService
from app.backend.services.run_manager import RunManager
from app.tools.approval import cleaning_approval_decision, modeling_approval_decision
from app.tools.trace_logger import TraceLogger
from app.workflows.workflow_state import (
    all_steps_terminal,
    get_step_state,
    mark_step_completed,
    mark_step_failed,
    mark_step_running,
    mark_step_skipped,
    mark_step_waiting_for_approval,
    mark_workflow_completed,
    mark_workflow_running,
    next_pending_step,
    reset_step_for_retry,
    save_workflow_state,
    set_step_approval,
    state_path_for_logs_dir,
)
from app.workflows.workflow_steps import (
    CLEANING_PLAN_STEP,
    CLEANING_STEP,
    EDA_STEP,
    MODELING_STEP,
    PROFILE_STEP,
    REPORT_STEP,
)


class AnalysisGraph:
    """Simple deterministic state machine for the analysis workflow."""

    orchestrator_name = "OrchestratorAgent"

    def __init__(
        self,
        run_manager: RunManager | None = None,
        trace_logger: TraceLogger | None = None,
        profiler_agent: ProfilerAgent | None = None,
        cleaning_agent: CleaningAgent | None = None,
        eda_agent: EDAAgent | None = None,
        modeling_agent: ModelingAgent | None = None,
        report_agent: ReportAgent | None = None,
    ) -> None:
        self.run_manager = run_manager or RunManager()
        self.trace_logger = trace_logger or TraceLogger(self.run_manager)

        profiling_service = ProfilingService(self.run_manager)
        cleaning_service = CleaningService(
            run_manager=self.run_manager,
            profiling_service=profiling_service,
        )
        eda_service = EDAService(self.run_manager)
        evaluation_service = EvaluationService(self.run_manager)
        modeling_service = ModelingService(
            run_manager=self.run_manager,
            evaluation_service=evaluation_service,
        )
        report_service = ReportService(self.run_manager)

        self.profiler_agent = profiler_agent or ProfilerAgent(profiling_service)
        self.cleaning_agent = cleaning_agent or CleaningAgent(cleaning_service)
        self.eda_agent = eda_agent or EDAAgent(eda_service)
        self.modeling_agent = modeling_agent or ModelingAgent(modeling_service)
        self.report_agent = report_agent or ReportAgent(report_service)

    def run_until_pause_or_complete(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run pending steps until the workflow completes, fails, or needs approval."""

        state = deepcopy(state)
        run_id = str(state["run_id"])
        mark_workflow_running(state, state.get("current_step") or PROFILE_STEP)
        self._save_state(state)

        while True:
            if state.get("status") in {"failed", "waiting_for_approval"}:
                return state

            if all_steps_terminal(state):
                mark_workflow_completed(state)
                self._trace(
                    run_id,
                    self.orchestrator_name,
                    None,
                    "workflow_completed",
                    "Workflow completed.",
                )
                self._save_state(state)
                return state

            step = next_pending_step(state)
            if step is None:
                mark_workflow_completed(state)
                self._save_state(state)
                return state

            if step == CLEANING_STEP and self._pause_for_cleaning_approval(state):
                return state

            if step == MODELING_STEP:
                if self._skip_modeling_without_target(state):
                    continue
                if self._skip_modeling_without_cleaned_data(state):
                    continue
                if self._pause_for_modeling_approval(state):
                    return state

            state = self._run_step(state, step)

    def apply_approval(
        self,
        state: dict[str, Any],
        step: str,
        action: str,
    ) -> dict[str, Any]:
        """Apply a human approval action and continue when appropriate."""

        if step not in {CLEANING_STEP, MODELING_STEP}:
            raise ValueError(f"Step '{step}' does not support approval.")
        if action not in {"approve", "reject"}:
            raise ValueError("Approval action must be 'approve' or 'reject'.")

        state = deepcopy(state)
        run_id = str(state["run_id"])
        step_state = get_step_state(state, step)
        if (
            step_state.get("status") != "waiting_for_approval"
            or step_state.get("approval_status") != "pending"
        ):
            raise ValueError(f"Step '{step}' is not waiting for approval.")

        if action == "approve":
            set_step_approval(state, step, "approved")
            step_state["status"] = "pending"
            step_state["error"] = None
            mark_workflow_running(state, step)
            self._trace(
                run_id,
                self.orchestrator_name,
                step,
                "approval_granted",
                f"Approval granted for step '{step}'.",
            )
            self._save_state(state)
            return self.run_until_pause_or_complete(state)

        reason = f"Human reviewer rejected step '{step}'."
        set_step_approval(state, step, "rejected", reason=reason)
        mark_step_skipped(state, step, reason)
        state.setdefault("warnings", []).append(reason)
        self._trace(
            run_id,
            self.orchestrator_name,
            step,
            "approval_rejected",
            reason,
        )
        self._save_state(state)
        return self.run_until_pause_or_complete(state)

    def retry_step(self, state: dict[str, Any], step: str) -> dict[str, Any]:
        """Retry a failed step if attempts remain."""

        state = deepcopy(state)
        run_id = str(state["run_id"])
        reset_step_for_retry(state, step)
        self._trace(
            run_id,
            self.orchestrator_name,
            step,
            "step_retried",
            f"Retry requested for step '{step}'.",
            {"attempts_before_retry": get_step_state(state, step)["attempts"]},
        )
        self._save_state(state)
        return self.run_until_pause_or_complete(state)

    def _run_step(self, state: dict[str, Any], step: str) -> dict[str, Any]:
        run_id = str(state["run_id"])
        agent_name = self._agent_name_for_step(step)

        mark_step_running(state, step)
        self._trace(
            run_id,
            agent_name,
            step,
            "step_started",
            f"Started step '{step}'.",
            {"attempt": get_step_state(state, step)["attempts"]},
        )
        self._save_state(state)

        try:
            if step == PROFILE_STEP:
                updated_state = self.profiler_agent.run(state)
            elif step == CLEANING_PLAN_STEP:
                updated_state = self.cleaning_agent.generate_plan(state)
            elif step == CLEANING_STEP:
                updated_state = self.cleaning_agent.apply_cleaning(state)
            elif step == EDA_STEP:
                updated_state = self.eda_agent.run(state)
            elif step == MODELING_STEP:
                updated_state = self.modeling_agent.run(state)
            elif step == REPORT_STEP:
                updated_state = self.report_agent.run(state)
            else:
                raise ValueError(f"Unknown workflow step: {step}")
        except Exception as exc:
            error = str(exc)
            mark_step_failed(state, step, error)
            self._trace(
                run_id,
                agent_name,
                step,
                "step_failed",
                f"Step '{step}' failed: {error}",
                {
                    "attempts": get_step_state(state, step)["attempts"],
                    "max_attempts": get_step_state(state, step)["max_attempts"],
                },
            )
            self._save_state(state)
            return state

        mark_step_completed(updated_state, step)
        self._trace(
            run_id,
            agent_name,
            step,
            "step_completed",
            f"Completed step '{step}'.",
            {"attempts": get_step_state(updated_state, step)["attempts"]},
        )
        self._save_state(updated_state)
        return updated_state

    def _pause_for_cleaning_approval(self, state: dict[str, Any]) -> bool:
        step_state = get_step_state(state, CLEANING_STEP)
        if step_state.get("approval_status") in {"approved", "rejected", "not_required"}:
            return False

        run_id = str(state["run_id"])
        plan = self.cleaning_agent.cleaning_service.load_cleaning_plan(run_id)
        decision = cleaning_approval_decision(
            plan.model_dump(mode="json"),
            require_approval=bool(
                state.get("approval_settings", {}).get(
                    "require_cleaning_approval",
                    True,
                )
            ),
        )
        if not decision["required"]:
            set_step_approval(
                state,
                CLEANING_STEP,
                "not_required",
                reason="Cleaning approval was not required by the active rules.",
                details=decision,
            )
            self._save_state(state)
            return False

        reason = "; ".join(decision["reasons"])
        mark_step_waiting_for_approval(
            state,
            CLEANING_STEP,
            reason=reason,
            details=decision,
        )
        self._trace(
            run_id,
            self.orchestrator_name,
            CLEANING_STEP,
            "approval_required",
            "Cleaning approval is required before safe cleaning runs.",
            decision,
        )
        self._save_state(state)
        return True

    def _pause_for_modeling_approval(self, state: dict[str, Any]) -> bool:
        step_state = get_step_state(state, MODELING_STEP)
        if step_state.get("approval_status") in {"approved", "rejected", "not_required"}:
            return False

        run_id = str(state["run_id"])
        metadata = self._load_metadata(run_id)
        decision = modeling_approval_decision(
            state,
            metadata=metadata,
            require_approval=bool(
                state.get("approval_settings", {}).get(
                    "require_modeling_approval",
                    True,
                )
            ),
        )
        if not decision["required"]:
            set_step_approval(
                state,
                MODELING_STEP,
                "not_required",
                reason="Modeling approval was not required by the active rules.",
                details=decision,
            )
            self._save_state(state)
            return False

        reason = "; ".join(decision["reasons"])
        mark_step_waiting_for_approval(
            state,
            MODELING_STEP,
            reason=reason,
            details=decision,
        )
        self._trace(
            run_id,
            self.orchestrator_name,
            MODELING_STEP,
            "approval_required",
            "Modeling approval is required before training starts.",
            decision,
        )
        self._save_state(state)
        return True

    def _skip_modeling_without_target(self, state: dict[str, Any]) -> bool:
        if state.get("target_column"):
            return False

        reason = "Modeling was skipped because no target column was provided."
        mark_step_skipped(state, MODELING_STEP, reason)
        self._trace(
            str(state["run_id"]),
            self.orchestrator_name,
            MODELING_STEP,
            "step_skipped",
            reason,
        )
        self._save_state(state)
        return True

    def _skip_modeling_without_cleaned_data(self, state: dict[str, Any]) -> bool:
        run_id = str(state["run_id"])
        cleaned_data_path = self.run_manager.get_paths(run_id).intermediate / "cleaned_data.csv"
        if cleaned_data_path.exists():
            return False

        reason = "Modeling was skipped because cleaned_data.csv is unavailable."
        mark_step_skipped(state, MODELING_STEP, reason)
        state.setdefault("warnings", []).append(reason)
        self._trace(
            run_id,
            self.orchestrator_name,
            MODELING_STEP,
            "step_skipped",
            reason,
        )
        self._save_state(state)
        return True

    def _save_state(self, state: dict[str, Any]) -> None:
        run_id = str(state["run_id"])
        path = state_path_for_logs_dir(self.run_manager.get_paths(run_id).logs)
        save_workflow_state(path, state)

    def _trace(
        self,
        run_id: str,
        agent: str,
        step: str | None,
        event_type: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.trace_logger.append_event(
            run_id=run_id,
            agent=agent,
            step=step,
            event_type=event_type,
            message=message,
            details=details,
        )

    def _load_metadata(self, run_id: str) -> dict[str, Any] | None:
        try:
            return self.run_manager.load_metadata(run_id)
        except FileNotFoundError:
            return None

    def _agent_name_for_step(self, step: str) -> str:
        if step == PROFILE_STEP:
            return self.profiler_agent.name
        if step in {CLEANING_PLAN_STEP, CLEANING_STEP}:
            return self.cleaning_agent.name
        if step == EDA_STEP:
            return self.eda_agent.name
        if step == MODELING_STEP:
            return self.modeling_agent.name
        if step == REPORT_STEP:
            return self.report_agent.name
        return self.orchestrator_name


def build_analysis_graph(
    run_manager: RunManager | None = None,
    trace_logger: TraceLogger | None = None,
) -> AnalysisGraph:
    """Build the deterministic analysis graph."""

    return AnalysisGraph(run_manager=run_manager, trace_logger=trace_logger)
