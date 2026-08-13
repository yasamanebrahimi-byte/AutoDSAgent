from uuid import uuid4

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


def test_two_value_string_category_is_not_boolean():
    assert infer_semantic_type(pd.Series(["red", "blue"] * 10), "color") == "categorical"
    assert infer_semantic_type(pd.Series(["small", "large"] * 10), "size") == "categorical"
    assert infer_semantic_type(pd.Series(["cat", "dog"] * 10), "animal") == "categorical"


def test_standard_boolean_token_pairs_are_boolean():
    examples = [
        ["true", "false", "TRUE", "FALSE"],
        ["yes", "no", "Yes", "No"],
        ["y", "n", "Y", "N"],
        ["1", "0", "1", "0"],
        ["on", "off", "ON", "OFF"],
    ]

    for values in examples:
        assert infer_semantic_type(pd.Series(values), "flag") == "boolean"


def test_numeric_boolean_tokens_with_missing_values_are_boolean():
    series = pd.Series([1, 0, None, 1, 0])

    assert infer_semantic_type(series, "enabled") == "boolean"


def test_datetime_like_column_detected_as_datetime():
    series = pd.Series(["2026-01-01", "2026-01-02", "2026-01-03"])

    assert infer_semantic_type(series, "signup_date") == "datetime"


def test_id_like_column_detected_as_id():
    series = pd.Series([101, 102, 103])

    assert infer_semantic_type(series, "customer_id") == "id"


def test_unique_random_floats_are_numeric_not_ids():
    series = pd.Series([index * 0.137 + 0.001 for index in range(100)])

    assert infer_semantic_type(series, "random_float_feature") == "numeric"


def test_unique_regression_target_values_are_numeric_not_ids():
    series = pd.Series([100_000.0 + index * 1234.56 for index in range(100)])

    assert infer_semantic_type(series, "revenue") == "numeric"


def test_sequential_integer_id_named_id_is_id():
    series = pd.Series(range(1, 101))

    assert infer_semantic_type(series, "id") == "id"


def test_uuid_strings_are_ids():
    series = pd.Series([str(uuid4()) for _ in range(25)])

    assert infer_semantic_type(series, "event_uuid") == "id"


def test_high_cardinality_strings_without_id_semantics_are_categorical():
    series = pd.Series([f"segment-{index:03d}" for index in range(100)])

    assert infer_semantic_type(series, "segment") == "categorical"


def test_duplicated_numeric_feature_remains_numeric():
    series = pd.Series([1.5, 1.5, 2.0, 2.0, 3.5, 3.5])

    assert infer_semantic_type(series, "measurement") == "numeric"


def test_small_dataset_integer_values_without_id_name_remain_numeric():
    series = pd.Series([10, 20, 30])

    assert infer_semantic_type(series, "score") == "numeric"


def test_missing_values_do_not_prevent_id_detection_with_id_name():
    series = pd.Series([1, 2, None, 4, 5])

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
