import json
from pathlib import Path

from app.backend.schemas.eda import EDARequest
from app.backend.services.eda_service import EDAService, RAW_DATASET_WARNING
from app.backend.services.run_manager import RunManager


def test_eda_is_generated_from_cleaned_data_and_artifacts_are_saved(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "eda-cleaned-test"
    paths = manager.create_run(run_id)
    (paths.input / "raw_data.csv").write_text(
        "customer_id,age,income,city,churn\n"
        "1,30,60000,NY,yes\n"
        "2,,70000,LA,no\n"
        "3,35,80000,NY,yes\n"
        "4,40,90000,SF,yes\n"
        "5,45,100000,NY,yes\n",
        encoding="utf-8",
    )
    (paths.intermediate / "cleaned_data.csv").write_text(
        "customer_id,age,income,city,churn\n"
        "1,30,60000,NY,yes\n"
        "2,32,70000,LA,no\n"
        "3,35,80000,NY,yes\n"
        "4,40,90000,SF,yes\n"
        "5,45,100000,NY,yes\n",
        encoding="utf-8",
    )

    service = EDAService(run_manager=manager)
    response = service.generate_eda(
        run_id,
        EDARequest(target_column="churn"),
    )

    assert response.summary.dataset_used == "cleaned"
    assert response.summary.dataset_path == "intermediate/cleaned_data.csv"
    assert not Path(response.summary.dataset_path).is_absolute()
    assert service.eda_summary_path(run_id).exists()
    assert service.eda_findings_path(run_id).exists()
    assert service.eda_report_path(run_id).exists()
    assert response.summary.generated_plots
    assert any(
        plot.path.startswith("plots/numeric_distributions/")
        for plot in response.summary.generated_plots
    )
    assert any(
        plot.category == "target_relationship"
        for plot in response.summary.generated_plots
    )

    saved_summary = json.loads(service.eda_summary_path(run_id).read_text(encoding="utf-8"))
    saved_findings = json.loads(service.eda_findings_path(run_id).read_text(encoding="utf-8"))
    assert saved_summary["dataset_used"] == "cleaned"
    assert "recommended_next_steps" in saved_findings


def test_eda_falls_back_to_raw_data_when_cleaned_data_is_missing(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "eda-raw-fallback-test"
    paths = manager.create_run(run_id)
    (paths.input / "raw_data.csv").write_text(
        "age,income,city\n"
        "30,60000,NY\n"
        "35,70000,LA\n"
        "40,80000,NY\n"
        "45,90000,SF\n",
        encoding="utf-8",
    )

    service = EDAService(run_manager=manager)
    response = service.generate_eda(run_id)

    assert response.summary.dataset_used == "raw"
    assert response.summary.dataset_path == "input/raw_data.csv"
    assert not Path(response.summary.dataset_path).is_absolute()
    assert RAW_DATASET_WARNING in response.summary.warnings
    assert service.eda_summary_path(run_id).exists()
    assert service.eda_findings_path(run_id).exists()
    assert service.eda_report_path(run_id).exists()
    assert response.summary.generated_plots


def test_eda_rerun_clears_obsolete_target_plots(tmp_path):
    manager = RunManager(runs_dir=tmp_path)
    run_id = "eda-rerun-test"
    paths = manager.create_run(run_id)
    (paths.input / "raw_data.csv").write_text(
        "age,income,churn\n"
        "30,60000,yes\n"
        "35,70000,no\n"
        "40,80000,yes\n"
        "45,90000,no\n"
        "50,100000,yes\n",
        encoding="utf-8",
    )
    service = EDAService(run_manager=manager)

    first = service.generate_eda(run_id, EDARequest(target_column="churn"))
    assert any(
        plot.path.startswith("plots/target_relationships/")
        for plot in first.summary.generated_plots
    )

    second = service.generate_eda(run_id, EDARequest(target_column=None))

    assert not any(
        plot.path.startswith("plots/target_relationships/")
        for plot in second.summary.generated_plots
    )
    assert not (paths.plots / "target_relationships").exists()
