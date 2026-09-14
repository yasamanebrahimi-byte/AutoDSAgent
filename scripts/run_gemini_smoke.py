"""Run provider smoke calls against a small synthetic local dataset only."""

from __future__ import annotations

import json

import pandas as pd

from app.deterministic import deterministic_recommendation, profile_dataframe
from app.llm import build_agents
from app.validation import freeze_supervised_split, training_profile_frame, build_execution_contract


def main() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [float(index) for index in range(30)],
            "feature_b": [float((index * 7) % 11) for index in range(30)],
            "target": [float(index * 0.5 + (index % 3)) for index in range(30)],
        }
    )
    split = freeze_supervised_split(frame, "target", "regression", test_size=0.2, random_state=42)
    training = training_profile_frame(
        frame,
        "target",
        "regression",
        test_size=0.2,
        random_state=42,
        split=split,
    )
    profile = profile_dataframe(training)
    contract = build_execution_contract(
        training,
        "target",
        "regression",
        test_size=0.2,
        random_state=42,
    )
    deterministic = deterministic_recommendation(
        training,
        "Predict target from the numeric features.",
        "target",
        task_type="regression",
    ).model_dump(mode="json")
    rows = []
    for model in ("gemini-3.8-flash", "gemini-3.5-flash-lite"):
        agent = build_agents(
            provider="google",
            model=model,
            generation_settings={
                "temperature": None,
                "top_p": None,
                "seed": None,
                "reasoning_effort": "medium",
            },
            respect_environment_model=False,
        )
        plan = agent.modeling_plan(
            profile,
            "Predict target from the numeric features.",
            "target",
            "regression",
            execution_contract=contract,
        )
        planner_provenance = dict(agent.last_request_provenance or {})
        agent.assert_effective_model(expected_model=model)
        resolution = agent.reconcile_modeling(
            "Predict target from the numeric features.",
            profile,
            plan,
            deterministic,
        )
        reconciler_provenance = dict(agent.last_request_provenance or {})
        agent.assert_effective_model(expected_model=model)
        rows.append(
            {
                "provider": "google",
                "model_requested": model,
                "fallback_used": False,
                "planner_schema": type(plan).__name__,
                "reconciler_schema": type(resolution).__name__,
                "planner_provenance": planner_provenance,
                "reconciler_provenance": reconciler_provenance,
                "token_accounting_available": any(
                    value is not None
                    for provenance in (planner_provenance, reconciler_provenance)
                    for value in (
                        provenance.get("input_tokens"),
                        provenance.get("output_tokens"),
                        provenance.get("thought_tokens"),
                    )
                ),
            }
        )
    print(json.dumps({"synthetic_only": True, "results": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
