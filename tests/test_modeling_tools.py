import pytest
import pandas as pd

from app.tools.preprocessing import infer_task_type, prepare_modeling_data


def test_numeric_continuous_target_is_inferred_as_regression():
    dataframe = pd.DataFrame(
        {
            "feature": [float(index % 7) + index * 0.1 for index in range(60)],
            "target": [100.0 + index * 2.75 for index in range(60)],
        }
    )

    assert infer_task_type(dataframe, "target") == "regression"


def test_numeric_low_cardinality_target_is_inferred_as_classification():
    dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "target": [0, 1, 2] * 20,
        }
    )

    assert infer_task_type(dataframe, "target") == "classification"


def test_categorical_target_is_inferred_as_classification():
    dataframe = pd.DataFrame(
        {
            "feature": range(30),
            "target": ["yes", "no", "yes"] * 10,
        }
    )

    assert infer_task_type(dataframe, "target") == "classification"


def test_boolean_target_is_inferred_as_classification():
    dataframe = pd.DataFrame(
        {
            "feature": range(30),
            "target": [True, False, True] * 10,
        }
    )

    assert infer_task_type(dataframe, "target") == "classification"


def test_multiclass_text_target_is_inferred_as_classification():
    dataframe = pd.DataFrame(
        {
            "feature": range(45),
            "target": ["low", "medium", "high"] * 15,
        }
    )

    assert infer_task_type(dataframe, "target") == "classification"


def test_numeric_string_continuous_target_is_inferred_as_regression():
    dataframe = pd.DataFrame(
        {
            "feature": range(30),
            "target": [f"{100 + index * 1.25:.2f}" for index in range(30)],
        }
    )

    assert infer_task_type(dataframe, "target") == "regression"


def test_high_cardinality_text_target_is_rejected():
    dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "target": [f"free-text-label-{index}" for index in range(60)],
        }
    )

    with pytest.raises(ValueError, match="too many unique text values"):
        infer_task_type(dataframe, "target")


def test_id_like_target_is_rejected():
    dataframe = pd.DataFrame(
        {
            "feature": range(30),
            "customer_id": [f"CUST-{index:04d}" for index in range(30)],
        }
    )

    with pytest.raises(ValueError, match="identifier"):
        infer_task_type(dataframe, "customer_id")


def test_explicit_classification_rejects_continuous_numeric_target():
    dataframe = pd.DataFrame(
        {
            "feature": range(60),
            "target": [100.0 + index * 2.75 for index in range(60)],
        }
    )

    with pytest.raises(ValueError, match="continuous/high-cardinality"):
        infer_task_type(dataframe, "target", requested_task_type="classification")


def test_rare_class_target_fails_before_splitting():
    dataframe = pd.DataFrame(
        {
            "feature": range(12),
            "target": ["rare"] * 2 + ["common"] * 10,
        }
    )

    with pytest.raises(ValueError, match="rare class"):
        prepare_modeling_data(
            dataframe,
            target_column="target",
            task_type="classification",
            random_state=42,
        )


def test_task_inference_reason_is_persisted_in_preprocessing_metadata():
    dataframe = pd.DataFrame(
        {
            "feature": [index % 7 for index in range(30)],
            "target": [0, 1, 2] * 10,
        }
    )

    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        random_state=42,
    )

    assert prepared.task_type == "classification"
    assert prepared.task_inference_reason == (
        "low-cardinality discrete numeric target -> classification"
    )


def test_constant_target_raises_useful_error():
    dataframe = pd.DataFrame(
        {
            "feature": range(10),
            "target": ["same"] * 10,
        }
    )

    with pytest.raises(ValueError, match="constant"):
        infer_task_type(dataframe, "target")


def test_missing_target_column_raises_useful_error():
    dataframe = pd.DataFrame({"feature": range(10)})

    with pytest.raises(ValueError, match="was not found"):
        infer_task_type(dataframe, "target")
