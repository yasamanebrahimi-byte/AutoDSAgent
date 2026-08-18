from fastapi.testclient import TestClient

from app.backend.main import app
from app.frontend.streamlit_app import _model_comparison_dataframe


def test_fastapi_app_imports_and_health_endpoint_works():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "autods-agent-backend",
    }


def test_frontend_model_comparison_renders_new_selection_shape():
    dataframe = _model_comparison_dataframe(
        {
            "results": [
                {
                    "model_name": "baseline_median",
                    "role": "baseline",
                    "status": "succeeded",
                    "cv_metrics": {"cv_rmse_mean": 1.0},
                    "primary_metric_value": 1.0,
                },
                {
                    "model_name": "ridge",
                    "role": "candidate",
                    "status": "succeeded",
                    "cv_metrics": {"cv_rmse_mean": 5.0},
                    "primary_metric_value": 5.0,
                },
            ]
        },
        {
            "selected_model_name": "baseline_median",
            "best_candidate_name": "ridge",
        },
    )

    labels = dict(zip(dataframe["Model"], dataframe["Selection"]))

    assert labels["baseline_median"] == "selected"
    assert labels["ridge"] == "best candidate"


def test_frontend_model_comparison_keeps_legacy_fallback():
    dataframe = _model_comparison_dataframe(
        {},
        {
            "best_model_name": "ridge",
            "candidate_cv_results": {
                "ridge": {"cv_rmse_mean": 1.0},
            },
        },
    )

    assert dataframe.iloc[0]["Model"] == "ridge"
    assert dataframe.iloc[0]["Selection"] == "selected, best candidate"
