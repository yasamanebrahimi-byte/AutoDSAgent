import json
from pathlib import Path

from app.pipeline import run_analysis


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
