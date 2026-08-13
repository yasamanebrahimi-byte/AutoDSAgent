import pandas as pd
import pytest
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


def test_numeric_imputer_statistic_is_learned_from_training_partition_only():
    dataframe = pd.DataFrame(
        {
            "feature": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                1000,
                1001,
                1002,
                1003,
                1004,
                1005,
                1006,
                1007,
                1008,
                1009,
            ],
            "target": [float(index) * 2.0 for index in range(20)],
        }
    )
    dataframe.loc[[0, 5, 10], "feature"] = None

    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )
    fitted = prepared.preprocessor.fit(prepared.X_train, prepared.y_train)

    train_median = pd.to_numeric(prepared.X_train["feature"], errors="coerce").median()
    full_median = pd.to_numeric(dataframe["feature"], errors="coerce").median()
    learned_median = fitted.named_transformers_["numeric"].named_steps[
        "imputer"
    ].statistics_[0]

    assert train_median != full_median
    assert learned_median == pytest.approx(train_median)

    mutated = dataframe.copy()
    mutated.loc[prepared.X_test.index, "feature"] = -9999
    mutated_prepared = prepare_modeling_data(
        mutated,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )
    mutated_fitted = mutated_prepared.preprocessor.fit(
        mutated_prepared.X_train,
        mutated_prepared.y_train,
    )
    mutated_learned_median = mutated_fitted.named_transformers_[
        "numeric"
    ].named_steps["imputer"].statistics_[0]

    assert mutated_prepared.X_train.index.tolist() == prepared.X_train.index.tolist()
    assert mutated_learned_median == pytest.approx(learned_median)


def test_category_like_preprocessing_is_learned_from_training_partition_only():
    seed_dataframe = pd.DataFrame(
        {
            "flag": ["yes", "no"] * 10,
            "segment": ["A", "B"] * 10,
            "numeric_signal": [float(index) for index in range(20)],
            "target": [float(index) * 1.5 for index in range(20)],
        }
    )
    seed_prepared = prepare_modeling_data(
        seed_dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )

    dataframe = seed_dataframe.copy()
    train_indices = seed_prepared.X_train.index.tolist()
    test_indices = seed_prepared.X_test.index.tolist()
    dataframe["flag"] = None
    dataframe.loc[train_indices[:8], "flag"] = "no"
    dataframe.loc[train_indices[8:10], "flag"] = "yes"
    dataframe.loc[test_indices, "flag"] = "yes"
    dataframe.loc[train_indices, "segment"] = ["A", "B"] * 6
    dataframe.loc[test_indices, "segment"] = "TEST_ONLY_LEVEL"

    prepared = prepare_modeling_data(
        dataframe,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )
    fitted = prepared.preprocessor.fit(prepared.X_train, prepared.y_train)

    train_mode = prepared.X_train["flag"].mode(dropna=True).iloc[0]
    full_mode = dataframe["flag"].mode(dropna=True).iloc[0]
    learned_mode = fitted.named_transformers_["boolean"].named_steps[
        "imputer"
    ].statistics_[0]
    learned_categories = fitted.named_transformers_["categorical"].named_steps[
        "encoder"
    ].categories_[0].tolist()

    assert train_mode == "no"
    assert full_mode == "yes"
    assert learned_mode == train_mode
    assert "TEST_ONLY_LEVEL" not in learned_categories

    mutated = dataframe.copy()
    mutated.loc[test_indices, "flag"] = "no"
    mutated.loc[test_indices, "segment"] = "DIFFERENT_TEST_ONLY_LEVEL"
    mutated_prepared = prepare_modeling_data(
        mutated,
        target_column="target",
        task_type="regression",
        test_size=0.4,
        random_state=42,
    )
    mutated_fitted = mutated_prepared.preprocessor.fit(
        mutated_prepared.X_train,
        mutated_prepared.y_train,
    )
    mutated_learned_mode = mutated_fitted.named_transformers_[
        "boolean"
    ].named_steps["imputer"].statistics_[0]
    mutated_categories = mutated_fitted.named_transformers_[
        "categorical"
    ].named_steps["encoder"].categories_[0].tolist()

    assert mutated_prepared.X_train.index.tolist() == prepared.X_train.index.tolist()
    assert mutated_prepared.features_used == prepared.features_used
    assert mutated_prepared.categorical_features == prepared.categorical_features
    assert mutated_prepared.boolean_features == prepared.boolean_features
    assert mutated_learned_mode == learned_mode
    assert mutated_categories == learned_categories
    assert "DIFFERENT_TEST_ONLY_LEVEL" not in mutated_categories
