import pandas as pd
from scipy import sparse

from app.tools.preprocessing import prepare_modeling_data


def test_preprocessing_handles_mixed_features_and_excludes_ids_and_target():
    dataframe = pd.DataFrame(
        {
            "customer_id": list(range(1, 41)),
            "age": [25 + index % 12 for index in range(40)],
            "spend": [20.5 + (index % 9) * 3.25 for index in range(40)],
            "city": ["NY", "LA", "SF", None] * 10,
            "is_member": [True, False, True, None] * 10,
            "signup_date": [f"2025-01-{(index % 28) + 1:02d}" for index in range(40)],
            "notes": [
                f"long free text value {index} that should be treated as future text modeling"
                for index in range(40)
            ],
            "churn": ["yes", "no"] * 20,
        }
    )

    prepared = prepare_modeling_data(
        dataframe=dataframe,
        target_column="churn",
        task_type="classification",
        test_size=0.25,
        random_state=42,
    )

    assert prepared.train_rows == 30
    assert prepared.test_rows == 10
    assert "churn" not in prepared.features_used
    assert prepared.excluded_feature_reasons["churn"] == "target column"
    assert prepared.excluded_feature_reasons["customer_id"] == "likely identifier"
    assert prepared.excluded_feature_reasons["notes"] == "free-text modeling is future work"
    assert "age" in prepared.numeric_features
    assert "spend" in prepared.numeric_features
    assert "city" in prepared.categorical_features
    assert "is_member" in prepared.boolean_features
    assert "signup_date__year" in prepared.numeric_features

    fitted = prepared.preprocessor.fit(prepared.X_train, prepared.y_train)
    transformed = fitted.transform(prepared.X_train)
    assert transformed.shape[0] == prepared.train_rows
    assert transformed.shape[1] >= 4


def test_preprocessing_excludes_rows_with_missing_target_values():
    dataframe = pd.DataFrame(
        {
            "feature": [10, 20, 30, 40, 50, 60, 70],
            "segment": ["A", "B", "A", "B", "A", "B", "A"],
            "target": [1.5, None, 3.0, 4.5, None, 6.0, 7.5],
        }
    )

    prepared = prepare_modeling_data(
        dataframe=dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.25,
        random_state=42,
    )

    assert prepared.rows_used == 5
    assert prepared.train_rows + prepared.test_rows == 5
    assert prepared.y_train.isna().sum() == 0
    assert prepared.y_test.isna().sum() == 0


def test_preprocessing_uses_numeric_looking_string_features_as_numeric():
    dataframe = pd.DataFrame(
        {
            "numeric_text": ["1.2", "2.4", "3.1", None, "5.5", "6.2"] * 5,
            "target": [float(index) for index in range(30)],
        }
    )

    prepared = prepare_modeling_data(
        dataframe=dataframe,
        target_column="target",
        task_type="regression",
        random_state=42,
    )

    assert "numeric_text" in prepared.numeric_features
    assert "numeric_text" not in prepared.categorical_features


def test_high_cardinality_categorical_feature_is_excluded_before_one_hot():
    dataframe = pd.DataFrame(
        {
            "category": [f"category-{index:03d}" for index in range(80)],
            "signal": [index % 7 for index in range(80)],
            "target": [float(index) for index in range(80)],
        }
    )

    prepared = prepare_modeling_data(
        dataframe=dataframe,
        target_column="target",
        task_type="regression",
        random_state=42,
    )

    assert prepared.excluded_feature_reasons["category"] == (
        "high-cardinality categorical feature"
    )
    assert "category" not in prepared.features_used


def test_sparse_one_hot_output_is_preserved_for_compatible_pipelines():
    dataframe = pd.DataFrame(
        {
            "category": [f"group_{index % 20}" for index in range(80)],
            "target": [float(index % 11) for index in range(80)],
        }
    )

    prepared = prepare_modeling_data(
        dataframe=dataframe,
        target_column="target",
        task_type="regression",
        random_state=42,
    )
    transformed = prepared.preprocessor.fit_transform(prepared.X_train, prepared.y_train)

    assert sparse.issparse(transformed)
