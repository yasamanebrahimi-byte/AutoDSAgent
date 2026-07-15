import pandas as pd

from app.backend.services.cleaning_service import CleaningService
from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager


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
