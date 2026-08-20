from pathlib import Path

import pandas as pd

from app.deterministic import deterministic_recommendation, profile_dataframe


ROOT = Path(__file__).resolve().parents[1]


def test_profiles_and_recommends_classification():
    data = pd.read_csv(ROOT / "examples" / "sample_data" / "breast_cancer_wisconsin.csv")
    profile = profile_dataframe(data)
    recommendation = deterministic_recommendation(
        data,
        "classify diagnosis from the measured features",
        target_hint="diagnosis",
    )

    assert profile["rows"] == 569
    assert recommendation.target_column == "diagnosis"
    assert recommendation.task_type == "classification"
    assert recommendation.recommended_method in {
        "linear",
        "regularized_linear",
        "tree_ensemble",
        "boosted_tree",
    }


def test_recommends_regression_for_diabetes_target():
    data = pd.read_csv(ROOT / "examples" / "sample_data" / "diabetes_progression.csv")
    recommendation = deterministic_recommendation(
        data,
        "estimate disease progression",
        target_hint="disease_progression",
    )

    assert recommendation.task_type == "regression"
    assert recommendation.target_column == "disease_progression"

