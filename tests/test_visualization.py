import pandas as pd

from app.tools.visualization import (
    create_categorical_bar_chart,
    create_correlation_heatmap,
    create_missing_values_plot,
    create_numeric_histogram,
)


def test_missing_values_plot_is_created_when_missing_values_exist(tmp_path):
    dataframe = pd.DataFrame({"age": [10, None, 30], "city": ["NY", "LA", None]})

    path = create_missing_values_plot(dataframe, tmp_path / "plots")

    assert path is not None
    assert path.exists()
    assert path.name == "missing_values.png"


def test_numeric_histogram_is_created_for_numeric_column(tmp_path):
    dataframe = pd.DataFrame({"age": [10, 20, 30, 40, 50]})
    output_dir = tmp_path / "plots" / "numeric_distributions"

    path = create_numeric_histogram(dataframe, "age", output_dir)

    assert path is not None
    assert path.exists()
    assert path.parent == output_dir
    assert path.name == "age_histogram.png"


def test_categorical_bar_chart_is_created_for_categorical_column(tmp_path):
    dataframe = pd.DataFrame({"city": ["NY", "LA", "NY", "SF", "NY"]})
    output_dir = tmp_path / "plots" / "categorical_distributions"

    path = create_categorical_bar_chart(dataframe, "city", output_dir)

    assert path is not None
    assert path.exists()
    assert path.parent == output_dir
    assert path.name == "city_bar.png"


def test_correlation_heatmap_is_created_with_enough_numeric_columns(tmp_path):
    dataframe = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60],
            "income": [40, 60, 80, 100, 120],
        }
    )

    path = create_correlation_heatmap(dataframe, ["age", "income"], tmp_path / "plots")

    assert path is not None
    assert path.exists()
    assert path.name == "correlation_heatmap.png"
