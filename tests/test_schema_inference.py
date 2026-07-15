import pandas as pd

from app.tools.schema_inference import infer_semantic_type


def test_numeric_column_detected_as_numeric():
    assert infer_semantic_type(pd.Series([10, 20, 30]), "amount") == "numeric"


def test_categorical_column_detected_as_categorical():
    series = pd.Series(["red", "blue", "green", "red", "blue", "green"])

    assert infer_semantic_type(series, "color") == "categorical"


def test_boolean_column_detected_as_boolean():
    series = pd.Series(["yes", "no", "yes", "no"])

    assert infer_semantic_type(series, "subscribed") == "boolean"


def test_datetime_like_column_detected_as_datetime():
    series = pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"])

    assert infer_semantic_type(series, "signup_date") == "datetime"


def test_id_like_column_detected_as_id():
    series = pd.Series([101, 102, 103])

    assert infer_semantic_type(series, "customer_id") == "id"


def test_text_like_column_detected_as_text():
    series = pd.Series(
        [
            "This is a long free-text response with enough words to look textual.",
            "Another long customer note that should not be treated as a category.",
            "A third long natural language comment for schema inference.",
        ]
    )

    assert infer_semantic_type(series, "customer_notes") == "text"
