from pathlib import Path

import numpy as np
import pandas as pd

from app.modeling import fit_selected_model


def test_training_uses_a_pipeline_and_writes_model(tmp_path: Path):
    rng = np.random.default_rng(42)
    rows = 80
    frame = pd.DataFrame(
        {
            "signal": rng.normal(size=rows),
            "segment": np.where(np.arange(rows) % 2, "a", "b"),
            "target": np.where(np.arange(rows) % 2, "yes", "no"),
        }
    )
    frame.loc[[2, 10, 25], "signal"] = np.nan

    result = fit_selected_model(
        frame,
        target_column="target",
        task_type="classification",
        method="tree_ensemble",
        output_dir=tmp_path,
    )

    assert result["selected_model"] == "random_forest"
    assert result["cv_folds"] >= 2
    assert "macro_f1" in result["holdout_metrics"]
    assert (tmp_path / "selected_model.joblib").exists()

