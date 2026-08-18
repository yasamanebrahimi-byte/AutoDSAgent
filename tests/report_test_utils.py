from __future__ import annotations

from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    file_sha256,
    fingerprint_payload,
    write_artifact_lineage,
)
from app.tools.file_utils import save_json
from app.workflows.workflow_state import create_initial_workflow_state


def create_report_run(
    tmp_path,
    run_id: str = "report-test",
    include_cleaning: bool = True,
    include_eda: bool = True,
    include_modeling: bool = True,
    include_workflow: bool = True,
):
    manager = RunManager(runs_dir=tmp_path)
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(
        "customer_id,age,income,segment,churn\n"
        "1,30,60000,A,yes\n"
        "2,,70000,B,no\n"
        "3,35,80000,A,yes\n"
        "4,40,90000,C,no\n",
        encoding="utf-8",
    )
    source_fingerprint = file_sha256(raw_path)
    generation_id = f"{run_id}-generation"
    target_column = "churn"

    metadata_path = save_json(
        paths.intermediate / "metadata.json",
        metadata_payload(run_id, source_fingerprint),
    )
    _write_lineage(
        metadata_path,
        paths.root,
        run_id,
        "metadata",
        None,
        source_fingerprint,
        None,
        None,
        {"source_data": source_fingerprint},
    )
    profile_path = save_json(paths.intermediate / "profile.json", profile_payload(run_id))
    _write_lineage(
        profile_path,
        paths.root,
        run_id,
        "profile",
        generation_id,
        source_fingerprint,
        target_column,
        None,
        {"source_data": source_fingerprint},
    )
    profile_fingerprint = file_sha256(profile_path)
    analysis_input = {
        "dataset_used": "raw",
        "path": "input/raw_data.csv",
        "fingerprint": source_fingerprint,
        "source_fingerprint": source_fingerprint,
        "selection_reason": "fixture_raw",
    }

    if include_cleaning:
        cleaning_plan_path = save_json(
            paths.intermediate / "cleaning_plan.json",
            cleaning_plan_payload(run_id),
        )
        _write_lineage(
            cleaning_plan_path,
            paths.root,
            run_id,
            "cleaning_plan",
            generation_id,
            source_fingerprint,
            target_column,
            None,
            {"source_data": source_fingerprint, "profile": profile_fingerprint},
        )
        cleaned_path = paths.intermediate / "cleaned_data.csv"
        cleaned_path.write_text(
            "customer_id,age,income,segment,churn\n"
            "1,30,60000,A,yes\n"
            "2,35,70000,B,no\n"
            "3,35,80000,A,yes\n"
            "4,40,90000,C,no\n",
            encoding="utf-8",
        )
        cleaned_fingerprint = file_sha256(cleaned_path)
        _write_lineage(
            cleaned_path,
            paths.root,
            run_id,
            "cleaned_data",
            generation_id,
            source_fingerprint,
            target_column,
            None,
            {
                "source_data": source_fingerprint,
                "profile": profile_fingerprint,
                "cleaning_plan": file_sha256(cleaning_plan_path),
            },
        )
        cleaning_summary_path = save_json(
            paths.intermediate / "cleaning_summary.json",
            cleaning_summary_payload(run_id),
        )
        _write_lineage(
            cleaning_summary_path,
            paths.root,
            run_id,
            "cleaning_summary",
            generation_id,
            source_fingerprint,
            target_column,
            None,
            {
                "source_data": source_fingerprint,
                "cleaning_plan": file_sha256(cleaning_plan_path),
                "cleaned_data": cleaned_fingerprint,
            },
        )
        analysis_input = {
            "dataset_used": "cleaned",
            "path": "intermediate/cleaned_data.csv",
            "fingerprint": cleaned_fingerprint,
            "source_fingerprint": source_fingerprint,
            "selection_reason": "fixture_cleaned",
        }

    if include_eda:
        eda_summary_path = save_json(
            paths.intermediate / "eda_summary.json",
            eda_summary_payload(run_id),
        )
        eda_findings_path = save_json(
            paths.intermediate / "eda_findings.json",
            eda_findings_payload(),
        )
        upstream = {
            "source_data": source_fingerprint,
            analysis_input["dataset_used"]: analysis_input["fingerprint"],
        }
        _write_lineage(
            eda_summary_path,
            paths.root,
            run_id,
            "eda_summary",
            generation_id,
            source_fingerprint,
            target_column,
            None,
            upstream,
        )
        _write_lineage(
            eda_findings_path,
            paths.root,
            run_id,
            "eda_findings",
            generation_id,
            source_fingerprint,
            target_column,
            None,
            upstream,
        )

    if include_modeling:
        modeling_upstream = {
            "source_data": source_fingerprint,
            analysis_input["dataset_used"]: analysis_input["fingerprint"],
        }
        modeling_summary_path = save_json(
            paths.intermediate / "modeling_summary.json",
            modeling_summary_payload(run_id),
        )
        evaluation_summary_path = save_json(
            paths.intermediate / "evaluation_summary.json",
            evaluation_summary_payload(run_id),
        )
        model_results_path = save_json(
            paths.models / "model_results.json",
            model_results_payload(run_id),
        )
        for artifact_path, artifact_type in (
            (modeling_summary_path, "modeling_summary"),
            (evaluation_summary_path, "evaluation_summary"),
            (model_results_path, "model_results"),
        ):
            _write_lineage(
                artifact_path,
                paths.root,
                run_id,
                artifact_type,
                generation_id,
                source_fingerprint,
                target_column,
                "classification",
                modeling_upstream,
            )

    if include_workflow:
        state = create_initial_workflow_state(
            run_id=run_id,
            target_column=target_column,
            task_type="classification" if include_modeling else None,
            require_cleaning_approval=False,
            require_modeling_approval=False,
            generation_id=generation_id,
            source_fingerprint=source_fingerprint,
        )
        state["analysis_input"] = analysis_input
        for step_name, step_state in state["steps"].items():
            step_state["status"] = "completed"
            step_state["attempts"] = 1
        if not include_modeling:
            state["steps"]["modeling"]["status"] = "skipped"
            state["steps"]["modeling"]["outputs"]["skip_reason"] = (
                "Modeling was skipped because no target column was provided."
            )
        save_json(paths.logs / "workflow_state.json", state)
        save_json(
            paths.logs / "agent_trace.json",
            [
                {
                    "timestamp": "2026-08-12T00:00:00+00:00",
                    "run_id": run_id,
                    "agent": "ReportAgent",
                    "step": "report",
                    "event_type": "step_completed",
                    "message": "Completed step 'report'.",
                    "details": {},
                }
            ],
        )

    return manager, paths, run_id


def _write_lineage(
    artifact_path,
    run_root,
    run_id: str,
    artifact_type: str,
    generation_id: str | None,
    source_fingerprint: str,
    target_column: str | None,
    task_type: str | None,
    upstream_fingerprints: dict,
) -> None:
    config = {
        "artifact_type": artifact_type,
        "source_fingerprint": source_fingerprint,
        "target_column": target_column,
        "task_type": task_type,
        "upstream_fingerprints": upstream_fingerprints,
    }
    write_artifact_lineage(
        artifact_path,
        run_root=run_root,
        run_id=run_id,
        artifact_type=artifact_type,
        generation_id=generation_id,
        source_fingerprint=source_fingerprint,
        target_column=target_column,
        task_type=task_type,
        config_fingerprint=fingerprint_payload(config),
        upstream_fingerprints=upstream_fingerprints,
        relevant_config=config,
    )


def metadata_payload(run_id: str, source_fingerprint: str | None = None) -> dict:
    return {
        "run_id": run_id,
        "filename": "customers.csv",
        "rows": 4,
        "columns": 5,
        "column_names": ["customer_id", "age", "income", "segment", "churn"],
        "dtypes": {
            "customer_id": "int64",
            "age": "float64",
            "income": "int64",
            "segment": "object",
            "churn": "object",
        },
        "missing_values": {"customer_id": 0, "age": 1, "income": 0, "segment": 0, "churn": 0},
        "duplicate_rows": 0,
        "preview": [],
        "source_fingerprint": source_fingerprint,
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def profile_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "rows": 4,
        "columns": 5,
        "total_missing_values": 1,
        "duplicate_rows": 0,
        "memory_usage_bytes": 1024,
        "column_type_counts": {
            "numeric": 2,
            "categorical": 2,
            "boolean": 0,
            "datetime": 0,
            "text": 0,
            "id": 1,
            "unknown": 0,
        },
        "is_empty": False,
        "has_duplicate_rows": False,
        "has_missing_values": True,
        "column_profiles": [
            {
                "column_name": "customer_id",
                "pandas_dtype": "int64",
                "semantic_type": "id",
                "missing_values": 0,
                "missing_percentage": 0,
                "unique_values": 4,
                "unique_percentage": 100,
                "sample_values": [1, 2, 3],
                "is_constant": False,
                "is_high_cardinality": False,
                "is_id": True,
                "is_datetime": False,
                "is_numeric": False,
                "is_categorical": False,
                "is_boolean": False,
                "is_text_like": False,
            },
            {
                "column_name": "age",
                "pandas_dtype": "float64",
                "semantic_type": "numeric",
                "missing_values": 1,
                "missing_percentage": 25,
                "unique_values": 3,
                "unique_percentage": 75,
                "sample_values": [30, 35, 40],
                "is_constant": False,
                "is_high_cardinality": False,
                "is_id": False,
                "is_datetime": False,
                "is_numeric": True,
                "is_categorical": False,
                "is_boolean": False,
                "is_text_like": False,
                "numeric_stats": {"mean": 35, "median": 35, "possible_outlier_count": 0},
            },
            {
                "column_name": "segment",
                "pandas_dtype": "object",
                "semantic_type": "categorical",
                "missing_values": 0,
                "missing_percentage": 0,
                "unique_values": 3,
                "unique_percentage": 75,
                "sample_values": ["A", "B", "C"],
                "is_constant": False,
                "is_high_cardinality": False,
                "is_id": False,
                "is_datetime": False,
                "is_numeric": False,
                "is_categorical": True,
                "is_boolean": False,
                "is_text_like": False,
            },
        ],
        "data_quality_issues": [
            {
                "severity": "warning",
                "issue_type": "missing_values",
                "column": "age",
                "message": "Column has missing values.",
                "recommendation": "Review missingness.",
            }
        ],
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def cleaning_plan_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "duplicate_row_handling": {
            "action_type": "drop_duplicates",
            "column": None,
            "strategy": "remove_exact_duplicates",
            "reason": "Remove exact duplicates only.",
            "apply": True,
            "details": {},
        },
        "missing_value_strategies": [
            {
                "action_type": "impute",
                "column": "age",
                "strategy": "median",
                "reason": "Numeric missing values were low enough for median imputation.",
                "apply": True,
                "details": {},
            }
        ],
        "columns_recommended_for_dropping": [],
        "columns_recommended_for_keeping": ["customer_id"],
        "type_conversion_recommendations": [],
        "encoding_recommendations": [],
        "warnings_requiring_review": [],
        "actions": [],
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def cleaning_summary_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "original_shape": [4, 5],
        "cleaned_shape": [4, 5],
        "duplicate_rows_removed": 0,
        "columns_dropped": [],
        "missing_values_before": 1,
        "missing_values_after": 0,
        "imputation_strategies_used": {"age": "median"},
        "type_conversions_applied": {},
        "warnings": [],
    }


def eda_summary_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "dataset_used": "cleaned",
        "dataset_path": "runs/report-test/intermediate/cleaned_data.csv",
        "target_column": "churn",
        "rows": 4,
        "columns": 5,
        "numeric_columns": ["age", "income"],
        "categorical_columns": ["segment", "churn"],
        "boolean_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "id_columns": ["customer_id"],
        "missing_values_remaining": {"age": 0},
        "duplicate_rows_remaining": 0,
        "generated_plots": [
            {
                "path": "plots/eda/missing_values.png",
                "label": "Missing Values",
                "category": "missing_values",
            }
        ],
        "key_statistics": {
            "target": {
                "target_column": "churn",
                "unique_values": 2,
                "findings": ["The target has two classes."],
            }
        },
        "warnings": [],
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def eda_findings_payload() -> dict:
    return {
        "univariate_findings": ["Column `income` varies across customers."],
        "bivariate_findings": ["`age` and `income` have a positive association."],
        "target_findings": ["The target `churn` has both yes and no classes."],
        "correlation_findings": ["`age` and `income` are associated, not causally linked."],
        "data_quality_notes": ["ID-like columns were excluded from automatic plots."],
        "recommended_next_steps": ["Validate churn definition with a domain expert."],
    }


def modeling_summary_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "dataset_path": "runs/report-test/intermediate/cleaned_data.csv",
        "target_column": "churn",
        "task_type": "classification",
        "rows_used": 4,
        "columns_used": 5,
        "features_used": ["age", "income", "segment"],
        "features_excluded": ["customer_id"],
        "excluded_feature_reasons": {"customer_id": "ID-like column"},
        "train_rows": 3,
        "test_rows": 1,
        "models_attempted": ["baseline_most_frequent", "logistic_regression"],
        "models_succeeded": ["baseline_most_frequent", "logistic_regression"],
        "models_failed": [],
        "best_candidate_name": "logistic_regression",
        "best_candidate_metrics": {"cv_macro_f1_mean": 1.0},
        "selected_model_name": "logistic_regression",
        "selected_model_role": "candidate",
        "candidate_beats_baseline": True,
        "selection_outcome": "candidate_selected_candidate_outperformed_baseline",
        "best_model_name": "logistic_regression",
        "baseline_model_name": "baseline_most_frequent",
        "actual_test_size": 0.25,
        "cv_folds": 2,
        "cv_strategy": "stratified_kfold",
        "task_inference_reason": "low-cardinality categorical string target -> classification",
        "classification_validation": {"class_count": 2, "min_class_count": 2},
        "primary_metric": "macro_f1",
        "warnings": [],
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def evaluation_summary_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "target_column": "churn",
        "task_type": "classification",
        "primary_metric": "macro_f1",
        "baseline_model_name": "baseline_most_frequent",
        "best_candidate_name": "logistic_regression",
        "best_candidate_metrics": {"cv_macro_f1_mean": 1.0},
        "selected_model_name": "logistic_regression",
        "selected_model_role": "candidate",
        "selected_model_cv_metrics": {"cv_macro_f1_mean": 1.0},
        "selected_model_holdout_metrics": {"macro_f1": 1.0},
        "candidate_beats_baseline": True,
        "selection_outcome": "candidate_selected_candidate_outperformed_baseline",
        "best_model_name": "logistic_regression",
        "baseline_metrics": {"cv_macro_f1_mean": 0.3333},
        "best_model_metrics": {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "macro_f1": 1.0,
            "weighted_f1": 1.0,
            "balanced_accuracy": 1.0,
        },
        "all_model_metrics": {
            "baseline_most_frequent": {"cv_macro_f1_mean": 0.3333},
            "logistic_regression": {"cv_macro_f1_mean": 1.0},
        },
        "candidate_cv_results": {
            "logistic_regression": {"cv_macro_f1_mean": 1.0},
        },
        "cv_model_metrics": {
            "baseline_most_frequent": {"cv_macro_f1_mean": 0.3333},
            "logistic_regression": {"cv_macro_f1_mean": 1.0},
        },
        "final_test_metrics": {"macro_f1": 1.0},
        "holdout_metrics": {"macro_f1": 1.0},
        "cv_folds": 2,
        "cv_strategy": "stratified_kfold",
        "selection_metric": "macro_f1",
        "selection_direction": "higher",
        "selection_tiebreaker": "original_result_order",
        "test_evaluated_model_names": ["logistic_regression"],
        "baseline_comparison": {
            "baseline_metric_value": 0.3333,
            "best_candidate_metric_value": 1.0,
            "selected_metric_value": 1.0,
            "candidate_beats_baseline": True,
            "candidate_tied_baseline": False,
            "absolute_improvement": 0.6667,
            "percent_improvement": 200.0,
            "interpretation": "The best candidate `logistic_regression` improved on the baseline with a higher mean CV macro_f1 and was selected.",
        },
        "generated_plots": [
            {
                "path": "plots/evaluation/model_comparison.png",
                "label": "CV Model Comparison",
                "category": "evaluation_model_comparison",
            }
        ],
        "warnings": [],
        "created_at": "2026-08-12T00:00:00+00:00",
    }


def model_results_payload(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "target_column": "churn",
        "task_type": "classification",
        "primary_metric": "macro_f1",
        "selection_direction": "higher",
        "selection_tiebreaker": "original_result_order",
        "baseline_model_name": "baseline_most_frequent",
        "baseline_metrics": {"cv_macro_f1_mean": 0.3333},
        "best_candidate_name": "logistic_regression",
        "best_candidate_metrics": {"cv_macro_f1_mean": 1.0},
        "selected_model_name": "logistic_regression",
        "selected_model_role": "candidate",
        "selected_model_cv_metrics": {"cv_macro_f1_mean": 1.0},
        "selected_model_holdout_metrics": {"macro_f1": 1.0},
        "candidate_beats_baseline": True,
        "selection_outcome": "candidate_selected_candidate_outperformed_baseline",
        "best_model_name": "logistic_regression",
        "cv_folds": 2,
        "cv_strategy": "stratified_kfold",
        "selection_metric": "macro_f1",
        "candidate_cv_results": {
            "logistic_regression": {"cv_macro_f1_mean": 1.0},
        },
        "cv_model_metrics": {
            "baseline_most_frequent": {"cv_macro_f1_mean": 0.3333},
            "logistic_regression": {"cv_macro_f1_mean": 1.0},
        },
        "final_test_metrics": {"macro_f1": 1.0},
        "holdout_metrics": {"macro_f1": 1.0},
        "test_evaluated_model_names": ["logistic_regression"],
        "results": [
            {
                "model_name": "baseline_most_frequent",
                "role": "baseline",
                "status": "succeeded",
                "metrics": {"cv_macro_f1_mean": 0.3333},
                "cv_metrics": {"cv_macro_f1_mean": 0.3333},
                "holdout_metrics": {},
                "primary_metric_value": 0.3333,
                "fold_count": 2,
                "selection_metric": "macro_f1",
                "error": None,
            },
            {
                "model_name": "logistic_regression",
                "role": "candidate",
                "status": "succeeded",
                "metrics": {"cv_macro_f1_mean": 1.0},
                "cv_metrics": {"cv_macro_f1_mean": 1.0},
                "holdout_metrics": {"macro_f1": 1.0},
                "primary_metric_value": 1.0,
                "fold_count": 2,
                "selection_metric": "macro_f1",
                "error": None,
            },
        ],
        "failed_models": [],
        "created_at": "2026-08-12T00:00:00+00:00",
    }
