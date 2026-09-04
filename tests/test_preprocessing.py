import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from app.modeling import fit_selected_model
from app.preprocessing import compare_preprocessing_plans
from app.schemas import PreprocessingContract
from app.validation import InvariantViolation, modeling_arrays, validate_training_plan


def _contract(**overrides) -> PreprocessingContract:
    values = {
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "numeric_scaling": "none",
        "categorical_encoding": "one_hot",
        "categorical_unknown_handling": "ignore",
        "identifier_handling": "exclude",
        "high_cardinality_handling": "exclude",
        "unsupported_text_handling": "exclude",
        "datetime_handling": "exclude",
        "infinity_handling": "replace_with_missing",
        "fit_inside_pipeline": True,
    }
    values.update(overrides)
    return PreprocessingContract(**values)


def _frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "segment": np.where(np.arange(rows) % 2, "b", "a"),
            "target": np.where(np.arange(rows) % 2, "yes", "no"),
        }
    )
    frame.loc[[2, 10, 25], "signal"] = np.nan
    frame.loc[[4, 11], "segment"] = None
    return frame


def test_contract_is_strict_and_pipeline_fitting_is_required():
    with pytest.raises(ValidationError):
        PreprocessingContract.model_validate(["schema_aware_encoding", "training_only_imputation"])
    with pytest.raises(ValidationError):
        PreprocessingContract.model_validate(["invent_a_transformer"])
    with pytest.raises(ValidationError):
        PreprocessingContract(fit_inside_pipeline=False)


def test_ordering_and_irrelevant_differences_do_not_trigger_reconciliation():
    frame = _frame()
    numeric_frame = frame.drop(columns=["segment"])
    base = _contract()
    numeric_only = base.model_copy(update={"categorical_imputation": "none", "categorical_encoding": "none"})
    requirements = validate_training_plan(
        numeric_frame,
        "target",
        "classification",
        "tree_ensemble",
    ).preprocessing_requirements
    comparison = compare_preprocessing_plans(base, numeric_only, requirements)
    assert comparison["status"] == "agreement"
    assert comparison["immaterial_differences"]



def test_missing_imputation_and_categorical_encoding_are_mandatory():
    frame = _frame()
    no_imputation = _contract(numeric_imputation="none", categorical_imputation="none")
    result = validate_training_plan(
        frame,
        "target",
        "classification",
        "tree_ensemble",
        preprocessing=no_imputation,
    )
    assert result.status == "failed"
    assert {check.code for check in result.failed_checks} >= {
        "numeric_missing_values_are_handled",
        "categorical_missing_values_are_handled",
    }

    no_encoder = _contract(categorical_encoding="none")
    result = validate_training_plan(
        frame,
        "target",
        "classification",
        "tree_ensemble",
        preprocessing=no_encoder,
    )
    assert result.status == "failed"
    assert "categorical_features_use_safe_encoding" in {
        check.code for check in result.failed_checks
    }


def test_linear_scaling_is_required_but_tree_models_are_not_forced_to_scale():
    frame = _frame()
    linear = validate_training_plan(
        frame,
        "target",
        "classification",
        "regularized_linear",
        preprocessing=_contract(numeric_scaling="none"),
    )
    assert "linear_numeric_features_use_approved_scaling_policy" in {
        check.code for check in linear.failed_checks
    }
    tree = validate_training_plan(
        frame,
        "target",
        "classification",
        "tree_ensemble",
        preprocessing=_contract(numeric_scaling="none"),
    )
    assert tree.status == "passed"


def test_executed_pipeline_matches_contract_and_handles_unknown_categories(tmp_path: Path):
    contract = _contract()
    result = fit_selected_model(
        _frame(),
        target_column="target",
        task_type="classification",
        method="tree_ensemble",
        output_dir=tmp_path,
        preprocessing=contract,
    )
    assert result["approved_preprocessing"] == contract.model_dump(mode="json")
    assert result["executed_preprocessing"]["contract"] == result["approved_preprocessing"]
    pipeline = joblib.load(tmp_path / "selected_model.joblib")
    categorical = next(
        transformer
        for name, transformer, _ in pipeline.named_steps["preprocessor"].transformers
        if name == "categorical"
    )
    encoder = categorical.named_steps["encoder"]
    assert encoder.handle_unknown == "ignore"
    predictions = pipeline.predict(pd.DataFrame({"signal": [0.1], "segment": ["unseen"]}))
    assert len(predictions) == 1


def test_imputation_is_fitted_on_training_partition_only(tmp_path: Path):
    frame = _frame()
    contract = _contract()
    validation = validate_training_plan(
        frame,
        "target",
        "classification",
        "tree_ensemble",
        preprocessing=contract,
    )
    validation.raise_if_failed()
    features, target = modeling_arrays(frame, validation)
    X_train, _, _, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )
    result = fit_selected_model(
        frame,
        target_column="target",
        task_type="classification",
        method="tree_ensemble",
        output_dir=tmp_path,
        preprocessing=contract,
    )
    pipeline = joblib.load(result["model_path"])
    numeric = next(
        transformer
        for name, transformer, _ in pipeline.named_steps["preprocessor"].transformers_
        if name == "numeric"
    )
    assert numeric.named_steps["imputer"].statistics_[0] == pytest.approx(
        X_train["signal"].median()
    )


def test_invalid_preprocessing_contract_fails_closed(tmp_path: Path):
    with pytest.raises(ValidationError):
        PreprocessingContract(categorical_encoding="target_mean")
    result = validate_training_plan(
        _frame(),
        "target",
        "classification",
        "tree_ensemble",
        preprocessing=_contract(numeric_imputation="none"),
    )
    assert result.status == "failed"
    with pytest.raises(InvariantViolation):
        result.raise_if_failed()
    model_dir = tmp_path / "invalid_model"
    with pytest.raises(InvariantViolation):
        fit_selected_model(
            _frame(),
            target_column="target",
            task_type="classification",
            method="tree_ensemble",
            output_dir=model_dir,
            preprocessing=_contract(numeric_imputation="none"),
        )
    assert not (model_dir / "selected_model.joblib").exists()


def test_offline_artifacts_persist_all_plans_and_reproduction_contract(tmp_path: Path):
    dataset = tmp_path / "classification.csv"
    _frame(100).to_csv(dataset, index=False)
    from app.pipeline import run_analysis

    result = run_analysis(
        dataset,
        "classify target",
        target_column="target",
        output_dir=tmp_path / "runs",
        offline=True,
    )
    run_dir = Path(result["run_dir"])
    decision = json.loads((run_dir / "decision.json").read_text(encoding="utf-8"))
    modeling = json.loads((run_dir / "modeling.json").read_text(encoding="utf-8"))
    assert decision["modeling_plan"]["preprocessing"]
    assert decision["deterministic_recommendation"]["preprocessing"]
    assert decision["validation"]["approved_preprocessing"]
    assert decision["validation"]["preprocessing_comparison"]
    assert modeling["approved_preprocessing"] == decision["validation"]["approved_preprocessing"]
    assert "APPROVED_PREPROCESSING" in (run_dir / "reproduce_analysis.py").read_text(
        encoding="utf-8"
    )
    assert (run_dir / "report.md").read_text(encoding="utf-8").find("Preprocessing contract") >= 0
