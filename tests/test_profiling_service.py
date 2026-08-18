from app.backend.services.profiling_service import ProfilingService
from app.backend.services.run_manager import RunManager


def test_profile_is_generated_and_saved(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "profile-test"
    paths = manager.create_run(run_id)
    (paths.input / "raw_data.csv").write_text(
        "customer_id,age,segment,constant\n"
        "1,30,A,same\n"
        "2,,,same\n"
        "2,,,same\n"
        "3,40,B,same\n"
        "4,50,C,same\n",
        encoding="utf-8",
    )

    service = ProfilingService(run_manager=manager)
    profile = service.generate_profile(run_id, target_column="segment")

    assert profile.rows == 5
    assert profile.columns == 4
    assert profile.total_missing_values == 4
    assert profile.duplicate_rows == 1
    assert len(profile.column_profiles) == 4
    assert service.profile_path(run_id).exists()

    issue_types = {issue.issue_type for issue in profile.data_quality_issues}
    assert "duplicate_rows" in issue_types
    assert "constant_column" in issue_types
    assert "likely_id_column" in issue_types
    assert "target_not_selected" not in issue_types

    profile_by_column = {
        column.column_name: column
        for column in profile.column_profiles
    }
    assert profile_by_column["customer_id"].semantic_type == "id"
    assert profile_by_column["age"].semantic_type == "numeric"
    assert profile_by_column["segment"].semantic_type == "categorical"
    assert profile_by_column["constant"].is_constant is True
