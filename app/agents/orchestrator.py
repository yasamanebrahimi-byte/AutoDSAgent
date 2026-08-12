"""Coordinator for the deterministic analysis workflow."""

from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.workflows.analysis_graph import AnalysisGraph, build_analysis_graph


class OrchestratorAgent(BaseAgent):
    """Route work across deterministic specialist agents."""

    name = "OrchestratorAgent"

    def __init__(self, analysis_graph: AnalysisGraph | None = None) -> None:
        self.analysis_graph = analysis_graph or build_analysis_graph()

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Run the workflow until completion, failure, or approval pause."""

        return self.analysis_graph.run_until_pause_or_complete(state)

    def apply_approval(
        self,
        state: dict[str, Any],
        step: str,
        action: str,
    ) -> dict[str, Any]:
        """Apply approval or rejection to a waiting workflow step."""

        return self.analysis_graph.apply_approval(state, step, action)

    def retry_step(self, state: dict[str, Any], step: str) -> dict[str, Any]:
        """Retry a failed workflow step."""

        return self.analysis_graph.retry_step(state, step)
