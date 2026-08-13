import pandas as pd

from app.backend.services.cleaning_service import CleaningService
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager
from app.tools.cleaning import CleaningConfig, generate_cleaning_plan_payload
from app.tools.data_quality import detect_data_quality_issues


def test_cleaning_plan_and_safe_cleaning_artifacts_are_saved(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "cleaning-test"
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(
        "customer_id,age,segment,constant\n"
        "1,30,A,same\n"
        "2,,,same\n"
        "2,,,same\n"
        "3,40,B,same\n"
        "4,50,C,same\n",
        encoding="utf-8",
    )

    profiling_service = ProfilingService(run_manager=manager)
    cleaning_service = CleaningService(
        run_manager=manager,
        profiling_service=profiling_service,
    )

    plan = cleaning_service.generate_cleaning_plan(run_id)
    assert cleaning_service.cleaning_plan_path(run_id).exists()
    assert plan.duplicate_row_handling.apply is True
    assert "constant" in plan.columns_recommended_for_dropping

    summary = cleaning_service.apply_cleaning(run_id)
    cleaned_path = cleaning_service.cleaned_data_path(run_id)
    summary_path = cleaning_service.cleaning_summary_path(run_id)

    assert cleaned_path.exists()
    assert summary_path.exists()
    assert summary.duplicate_rows_removed == 1
    assert summary.original_shape == [5, 4]
    assert summary.cleaned_shape == [4, 3]
    assert summary.columns_dropped == ["constant"]
    assert summary.imputation_strategies_used["age"] == "median"
    assert summary.imputation_strategies_used["segment"] == "Unknown"
    assert summary.missing_values_after == 0

    cleaned = pd.read_csv(cleaned_path)
    assert "constant" not in cleaned.columns
    assert cleaned["age"].isna().sum() == 0
    assert "Unknown" in set(cleaned["segment"])

    raw = pd.read_csv(raw_path)
    assert raw.shape == (5, 4)
    assert "constant" in raw.columns


def test_numeric_target_missing_values_are_preserved(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "numeric-target-cleaning-test"
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(
        "customer_id,feature,revenue,segment\n"
        "1,10,100.5,A\n"
        "2,,200.0,B\n"
        "3,30,,\n"
        "4,40,400.0,A\n"
        "5,50,500.0,B\n",
        encoding="utf-8",
    )

    service = CleaningService(
        run_manager=manager,
        profiling_service=ProfilingService(manager),
    )

    plan = service.generate_cleaning_plan(run_id, target_column="revenue")
    summary = service.apply_cleaning(run_id, target_column="revenue")
    cleaned = pd.read_csv(service.cleaned_data_path(run_id))
    raw = pd.read_csv(raw_path)

    assert plan.target_column == "revenue"
    target_action = next(
        action for action in plan.missing_value_strategies if action.column == "revenue"
    )
    assert target_action.strategy == "preserve_missing_target"
    assert target_action.apply is False
    assert "revenue" not in summary.imputation_strategies_used
    assert summary.target_missing_values_before == 1
    assert summary.target_missing_values_after == 1
    assert cleaned["revenue"].isna().sum() == 1
    assert cleaned["feature"].isna().sum() == 0
    assert raw["revenue"].isna().sum() == 1


def test_categorical_target_missing_values_are_preserved(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "categorical-target-cleaning-test"
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(
        "customer_id,feature,churn,segment\n"
        "1,10,yes,A\n"
        "2,,no,B\n"
        "3,30,,\n"
        "4,40,yes,A\n"
        "5,50,no,B\n",
        encoding="utf-8",
    )

    service = CleaningService(
        run_manager=manager,
        profiling_service=ProfilingService(manager),
    )

    summary = service.apply_cleaning(run_id, target_column="churn")
    cleaned = pd.read_csv(service.cleaned_data_path(run_id))

    assert "churn" not in summary.imputation_strategies_used
    assert summary.target_missing_values_before == 1
    assert summary.target_missing_values_after == 1
    assert cleaned["churn"].isna().sum() == 1
    assert cleaned["feature"].isna().sum() == 0
    assert "Unknown" in set(cleaned["segment"])


def test_high_missingness_threshold_is_consistent_at_eighty_percent():
    profile = {
        "run_id": "threshold-test",
        "columns": 3,
        "duplicate_rows": 0,
        "column_profiles": [
            _column_profile("missing_799", 79.9),
            _column_profile("missing_800", 80.0),
            _column_profile("missing_801", 80.1),
        ],
        "data_quality_issues": [],
    }

    plan = generate_cleaning_plan_payload(profile, CleaningConfig())
    by_column = {action["column"]: action for action in plan["missing_value_strategies"]}

    assert by_column["missing_799"]["strategy"] == "median_imputation"
    assert by_column["missing_799"]["apply"] is True
    assert by_column["missing_800"]["strategy"] == "review_high_missingness"
    assert by_column["missing_800"]["apply"] is False
    assert by_column["missing_801"]["strategy"] == "review_high_missingness"
    assert by_column["missing_801"]["apply"] is False
    assert "missing_799" not in plan["columns_recommended_for_dropping"]
    assert "missing_800" in plan["columns_recommended_for_dropping"]
    assert "missing_801" in plan["columns_recommended_for_dropping"]

    issues = detect_data_quality_issues(
        pd.DataFrame({"a": range(1000)}),
        profile["column_profiles"],
    )
    severities = {issue["column"]: issue["severity"] for issue in issues if issue["column"]}
    assert severities["missing_799"] == "warning"
    assert severities["missing_800"] == "critical"
    assert severities["missing_801"] == "critical"


def test_valid_datetime_column_is_normalized(tmp_path):
    service, run_id = _cleaning_service_for_csv(
        tmp_path,
        "event_date,value\n"
        "2026-01-01,10\n"
        "2026-01-02,20\n"
        "2026-01-03,30\n",
    )

    summary = service.apply_cleaning(run_id)
    cleaned = pd.read_csv(service.cleaned_data_path(run_id))

    assert summary.type_conversions_applied["event_date"] == "parsed_datetime_to_iso_string"
    assert summary.datetime_parse_failures == {}
    assert cleaned["event_date"].tolist() == [
        "2026-01-01T00:00:00",
        "2026-01-02T00:00:00",
        "2026-01-03T00:00:00",
    ]


def test_mixed_validity_datetime_column_is_not_destroyed(tmp_path):
    service, run_id = _cleaning_service_for_csv(
        tmp_path,
        "event_date,value\n"
        "2026-01-01,10\n"
        "2026-01-02,20\n"
        "not-a-date,30\n"
        "2026-01-04,40\n"
        "2026-01-05,50\n",
    )

    summary = service.apply_cleaning(run_id)
    cleaned = pd.read_csv(service.cleaned_data_path(run_id))

    assert "event_date" not in summary.type_conversions_applied
    assert summary.datetime_parse_failures["event_date"]["failed_values"] == 1
    assert summary.datetime_parse_failures["event_date"]["examples"] == ["not-a-date"]
    assert "not-a-date" in cleaned["event_date"].tolist()


def test_invalid_datetime_like_name_without_parse_signal_is_left_alone(tmp_path):
    service, run_id = _cleaning_service_for_csv(
        tmp_path,
        "event_date,value\n"
        "not-a-date,10\n"
        "still-not,20\n"
        "unknown,30\n",
    )

    summary = service.apply_cleaning(run_id)
    cleaned = pd.read_csv(service.cleaned_data_path(run_id))

    assert summary.type_conversions_applied == {}
    assert summary.datetime_parse_failures == {}
    assert cleaned["event_date"].tolist() == ["not-a-date", "still-not", "unknown"]


def _column_profile(column_name: str, missing_percentage: float) -> dict[str, object]:
    return {
        "column_name": column_name,
        "semantic_type": "numeric",
        "missing_percentage": missing_percentage,
        "is_constant": False,
    }


def _cleaning_service_for_csv(tmp_path, csv_text: str) -> tuple[CleaningService, str]:
    manager = RunManager(runs_dir=tmp_path)
    run_id = "datetime-cleaning-test"
    paths = manager.create_run(run_id)
    (paths.input / "raw_data.csv").write_text(csv_text, encoding="utf-8")
    return CleaningService(
        run_manager=manager,
        profiling_service=ProfilingService(manager),
    ), run_id
