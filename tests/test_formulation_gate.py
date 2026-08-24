from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import app.pipeline as pipeline
from app.deterministic import choose_target, deterministic_formulation
from app.schemas import (
    CleaningPlan,
    FormulationPlan,
    FormulationResolution,
    ModelingPlan,
    ModelingResolution,
    PreprocessingContract,
    ReportDraft,
)
from app.validation import InvariantViolation


def _frame(rows: int = 64) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": [float(index) for index in range(rows)],
            "noise": [float((index * 7) % 11) for index in range(rows)],
            "diagnosis": ["positive" if index % 2 else "negative" for index in range(rows)],
        }
    )


class CapturingAgents:
    instance: "CapturingAgents | None" = None
    formulation_task = "classification"
    formulation_target = "diagnosis"

    def __init__(self, api_key=None, model="test"):
        del api_key
        self.model = model
        self.formulation_inputs = None
        self.formulation_reconciliation_calls = 0
        self.modeling_reconciliation_calls = 0
        CapturingAgents.instance = self

    def formulate_problem(self, profile, question, target_constraint=None):
        self.formulation_inputs = {
            "profile": profile,
            "question": question,
            "target_constraint": target_constraint,
        }
        return FormulationPlan(
            target_column=self.formulation_target,
            task_type=self.formulation_task,
            reasoning="The formulation proposal uses the question and compact schema evidence only.",
            confidence=0.8,
        )

    def reconcile_formulation(
        self, question, profile, user_target_constraint, agent_formulation, deterministic_formulation
    ):
        del question, profile, user_target_constraint, agent_formulation, deterministic_formulation
        self.formulation_reconciliation_calls += 1
        return FormulationResolution(
            selected_target_column="diagnosis",
            selected_task_type="classification",
            checks=["target_checked", "task_checked"],
            justification="The deterministic evidence supports classification for the categorical diagnosis target.",
            confidence=0.9,
        )

    def modeling_plan(self, profile, question, target_hint, task_type):
        del profile, question, target_hint, task_type
        return ModelingPlan(
            recommended_method="linear",
            preprocessing=PreprocessingContract(numeric_scaling="standard"),
            reasoning="The modeling proposal uses only the frozen training profile and approved formulation context.",
            confidence=0.7,
        )

    def reconcile_modeling(self, question, profile, modeling_plan, deterministic):
        del question, profile, modeling_plan
        self.modeling_reconciliation_calls += 1
        return ModelingResolution(
            selected_method=deterministic["recommended_method"],
            selected_preprocessing=PreprocessingContract.model_validate(deterministic["preprocessing"]),
            checks=["method_checked", "preprocessing_checked"],
            justification="The deterministic model recommendation satisfies the training-only compatibility checks.",
            confidence=0.8,
        )

    def cleaning(self, profile, target_column):
        del profile, target_column
        return CleaningPlan(actions=[], reasoning="No structural cleaning is required for this fixture.")

    def eda(self, question, summary):
        del question, summary
        return ["The training-only summary was inspected."]

    def report(self, question, context):
        del question, context
        return ReportDraft(
            executive_summary="The gated fixture completed with an auditable formulation and modeling decision.",
            key_findings=["The test run preserved the formulation boundary."],
            modeling_interpretation="The approved model was evaluated with training-only cross-validation.",
            limitations=["This is a test fixture."],
            next_steps=["Review the persisted decision artifact."],
        )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    target_column: str | None = None,
    formulation_task: str = "classification",
    formulation_target: str = "diagnosis",
):
    frame = _frame()
    dataset = tmp_path / "fixture.csv"
    frame.to_csv(dataset, index=False)
    CapturingAgents.formulation_task = formulation_task
    CapturingAgents.formulation_target = formulation_target
    monkeypatch.setattr(pipeline, "OpenAIAgents", CapturingAgents)
    result = pipeline.run_analysis(
        dataset,
        "Classify diagnosis from the measured features.",
        target_column=target_column,
        output_dir=tmp_path / "runs",
        offline=False,
    )
    run_dir = Path(result["run_dir"])
    return CapturingAgents.instance, json.loads((run_dir / "decision.json").read_text())


def test_formulation_inputs_are_independent_and_modeling_schema_is_separate(tmp_path, monkeypatch):
    agents, decision = _run(tmp_path, monkeypatch)
    assert set(agents.formulation_inputs) == {"profile", "question", "target_constraint"}
    assert agents.formulation_inputs["target_constraint"] is None
    assert decision["formulation"]["deterministic"]
    assert "deterministic" not in agents.formulation_inputs
    assert "holdout" not in json.dumps(agents.formulation_inputs).casefold()
    assert "recommended_method" not in decision["formulation"]["agent_initial"]
    assert "target_column" not in decision["modeling_gate"]["agent_initial"]
    assert "task_type" not in decision["modeling_gate"]["agent_initial"]


def test_agreement_freezes_split_without_formulation_reconciliation(tmp_path, monkeypatch):
    agents, decision = _run(tmp_path, monkeypatch)
    assert decision["formulation"]["status"] == "agreement"
    assert decision["formulation"]["comparison"]["overall_agreement"] is True
    assert decision["formulation"]["reconciliation"] is None
    assert agents.formulation_reconciliation_calls == 0
    assert decision["split_frozen_after_formulation_gate"] is True
    assert decision["split_contract"]["strategy"] == "stratified"


def test_task_disagreement_reconciles_before_split(tmp_path, monkeypatch):
    agents, decision = _run(tmp_path, monkeypatch, formulation_task="regression")
    assert decision["formulation"]["comparison"]["task_agreement"] is False
    assert decision["formulation"]["status"] == "disagreement_resolved"
    assert agents.formulation_reconciliation_calls == 1
    assert decision["formulation"]["final"]["task_type"] == "classification"
    assert decision["split_contract"]["strategy"] == "stratified"


def test_explicit_user_target_is_hard_constraint(tmp_path, monkeypatch):
    agents, decision = _run(
        tmp_path,
        monkeypatch,
        target_column="diagnosis",
        formulation_target="patient_id",
    )
    assert agents.formulation_inputs["target_constraint"] == {
        "target_column": "diagnosis",
        "target_source": "user_supplied",
        "target_is_mutable": False,
    }
    assert decision["formulation"]["final"]["target_column"] == "diagnosis"
    assert decision["formulation"]["final"]["target_is_mutable"] is False
    assert agents.formulation_reconciliation_calls == 0


def test_deterministic_target_inference_fails_closed_without_a_match():
    frame = pd.DataFrame(
        {
            "customer_status": ["new", "old"] * 8,
            "future_outcome": [0, 1] * 8,
            "patient_id": list(range(16)),
        }
    )
    result = deterministic_formulation(frame, "Predict customer behavior.")
    assert result.status == "uncertain"
    assert result.target_column is None
    with pytest.raises(ValueError, match="defensible target"):
        choose_target(frame, "Predict customer behavior.")


def test_invalid_formulation_never_freezes_split(tmp_path, monkeypatch):
    frame = _frame()
    dataset = tmp_path / "invalid.csv"
    frame.to_csv(dataset, index=False)
    freeze_calls = []
    monkeypatch.setattr(pipeline, "freeze_supervised_split", lambda *args, **kwargs: freeze_calls.append(1))
    with pytest.raises(InvariantViolation):
        pipeline.run_analysis(
            dataset,
            "Classify diagnosis from the measured features.",
            target_column="not_a_column",
            output_dir=tmp_path / "runs",
            offline=True,
        )
    assert freeze_calls == []
