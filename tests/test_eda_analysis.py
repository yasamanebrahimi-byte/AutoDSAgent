import pandas as pd

from app.tools.eda_analysis import (
    analyze_target_distribution,
    compute_correlation_summary,
    detect_outlier_patterns,
    summarize_categorical_columns,
    summarize_numeric_columns,
)


def test_numeric_statistics_are_computed_correctly():
    dataframe = pd.DataFrame({"value": [1, 2, 3, 4]})
    schema = {"value": "numeric"}

    summary = summarize_numeric_columns(dataframe, schema)

    assert summary["value"]["mean"] == 2.5
    assert summary["value"]["median"] == 2.5
    assert summary["value"]["minimum"] == 1.0
    assert summary["value"]["maximum"] == 4.0


def test_categorical_top_values_are_computed_correctly():
    dataframe = pd.DataFrame({"city": ["NY", "LA", "NY", "SF", "NY"]})
    schema = {"city": "categorical"}

    summary = summarize_categorical_columns(dataframe, schema)

    assert summary["city"]["unique_values"] == 3
    assert summary["city"]["top_values"][0] == {"value": "NY", "count": 3}


def test_outlier_detection_works_on_known_case():
    dataframe = pd.DataFrame({"value": [10, 11, 12, 13, 100]})

    findings = detect_outlier_patterns(dataframe, ["value"])

    assert len(findings) == 1
    assert "possible outliers" in findings[0]


def test_correlation_summary_identifies_strong_correlations():
    dataframe = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
            "noise": [2, 1, 2, 1, 2],
        }
    )

    summary = compute_correlation_summary(dataframe, ["x", "y", "noise"])

    assert summary["strong_pairs"][0]["column_a"] == "x"
    assert summary["strong_pairs"][0]["column_b"] == "y"
    assert summary["strong_pairs"][0]["correlation"] == 1.0


def test_target_imbalance_is_detected_for_classification_target():
    dataframe = pd.DataFrame({"churn": ["yes"] * 9 + ["no"]})
    schema = {"churn": "categorical"}

    analysis = analyze_target_distribution(dataframe, "churn", schema)

    assert analysis["is_imbalanced"] is True
    assert analysis["majority_class_percentage"] == 90.0
    assert "appears imbalanced" in analysis["findings"][0]
