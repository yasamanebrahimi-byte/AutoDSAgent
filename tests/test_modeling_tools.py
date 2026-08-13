import pytest
import pandas as pd

from app.tools.preprocessing import infer_task_type


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
