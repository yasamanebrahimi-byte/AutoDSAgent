from __future__ import annotations

import pandas as pd

from app.backend.services.dataset_service import generate_dataset_metadata
from app.backend.services.run_manager import RunManager
from app.backend.services.workflow_service import WorkflowService


def create_workflow_service(tmp_path, run_id: str = "workflow-test"):
    manager = RunManager(runs_dir=tmp_path)
    paths = manager.create_run(run_id)
    raw_path = paths.input / "raw_data.csv"
    raw_path.write_text(_workflow_csv(), encoding="utf-8")

    dataframe = pd.read_csv(raw_path)
    metadata = generate_dataset_metadata(
        dataframe=dataframe,
        filename="workflow.csv",
        run_id=run_id,
    )
    manager.save_metadata(run_id, metadata.model_dump(mode="json"))

    return WorkflowService(run_manager=manager), manager, paths, run_id


def _workflow_csv() -> str:
    rows = ["customer_id,feature_a,feature_b,segment,target,constant"]
    for index in range(1, 41):
        feature_a = "" if index in {5, 6} else f"{20 + index * 1.5:.1f}"
        feature_b = index % 7
        segment = "" if index in {7, 8} else ["A", "B", "C"][index % 3]
        target = "yes" if index % 2 == 0 else "no"
        rows.append(
            f"{index},{feature_a},{feature_b},{segment},{target},same"
        )
    rows.append("2,23.0,2,C,yes,same")
    return "\n".join(rows) + "\n"
