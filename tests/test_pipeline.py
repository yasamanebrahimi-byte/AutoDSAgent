import json
from pathlib import Path

import pandas as pd
import pytest

import app.pipeline as pipeline
from app.pipeline import run_analysis
from app.validation import DeterministicRecommendationUnavailable


ROOT = Path(__file__).resolve().parents[1]


def test_offline_run_persists_the_validation_gate_and_report(tmp_path: Path):
    result = run_analysis(
        ROOT / "examples" / "sample_data" / "diabetes_progression.csv",
        "Estimate disease progression from the patient measurements.",
        target_column="disease_progression",
        output_dir=tmp_path,
        offline=True,
    )
    run_dir = Path(result["run_dir"])
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert decision["validation"]["selected_method"]
    assert decision["validation"]["status"] in {"agreement", "disagreement_resolved"}
    deterministic_validation = decision["validation"]["deterministic_validation"]
    assert deterministic_validation["overall_status"] == "passed"
    assert {check["code"] for check in deterministic_validation["checks"]} >= {
        "target_exists",
        "target_absent_from_feature_matrix",
        "split_has_nonempty_partitions",
    }
    assert deterministic_validation["validated_target_column"] == "disease_progression"
    assert decision["gate_completed_before_training"] is True
    assert set(decision["agent_sources"]) >= {"modeling", "cleaning", "eda", "report"}
    assert manifest["api_used"] is False
    assert (run_dir / "report.md").exists()
    assert (run_dir / "reproduce_analysis.py").exists()
    assert "deterministic_recommendation" in (run_dir / "reproduce_analysis.py").read_text(
        encoding="utf-8"
    )
    assert "validate_training_plan" in (run_dir / "reproduce_analysis.py").read_text(
        encoding="utf-8"
    )


def test_deterministic_recommender_failure_fails_closed_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = tmp_path / "classification.csv"
    frame = pd.DataFrame(
        {
            "signal": list(range(40)),
            "target": ["yes", "no"] * 20,
        }
    )
    frame.to_csv(dataset, index=False)
    fit_calls: list[object] = []
    reconciliation_calls: list[object] = []

    def fail_deterministic(*args, **kwargs):
        raise RuntimeError("deterministic policy crashed")

    def record_fit(*args, **kwargs):
        fit_calls.append((args, kwargs))
        raise AssertionError("model fitting must not be reached")

    def record_reconciliation(*args, **kwargs):
        reconciliation_calls.append((args, kwargs))
        raise AssertionError("reconciliation must not be reached")

    monkeypatch.setattr(pipeline, "deterministic_recommendation", fail_deterministic)
    monkeypatch.setattr(pipeline, "fit_selected_model", record_fit)
    monkeypatch.setattr(pipeline.OpenAIAgents, "reconcile", record_reconciliation)

    with pytest.raises(DeterministicRecommendationUnavailable):
        run_analysis(
            dataset,
            "classify target",
            target_column="target",
            output_dir=tmp_path / "runs",
            offline=True,
        )

    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    assert decision["agent_plan"] is not None
    assert decision["deterministic_recommendation"] is None
    assert decision["validation"] is None
    assert decision["validation_gate_status"] == "not_completed"
    assert decision["gate_completed_before_training"] is False
    assert decision["model_training_occurred"] is False
    assert decision["failure"]["code"] == "deterministic_recommendation_unavailable"
    assert decision["failure"]["original_error_type"] == "RuntimeError"
    assert decision["failure"]["original_error_message"] == "deterministic policy crashed"
    assert "reconciliation" not in decision["agent_sources"]
    assert decision["failure"]["code"] not in {"agreement", "disagreement_resolved"}
    assert fit_calls == []
    assert reconciliation_calls == []
    assert manifest["validation_status"] == "failed"
    assert manifest["failure"]["code"] == "deterministic_recommendation_unavailable"
    assert not (run_dir / "model" / "selected_model.joblib").exists()
