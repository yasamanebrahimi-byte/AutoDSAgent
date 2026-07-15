"""Future coordinator for the AutoDS multi-agent analysis pipeline."""


class OrchestratorAgent:
    """Future agent responsible for routing work across specialist agents."""

    def run(self, state: dict) -> dict:
        raise NotImplementedError("OrchestratorAgent will be implemented in a future week.")
