"""Matplotlib visualization helpers for deterministic EDA artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from app.tools.file_utils import ensure_directory


def safe_filename(value: str, suffix: str = "") -> str:
    """Create a readable, deterministic, collision-resistant filename stem."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    normalized = normalized.strip("._-") or "column"
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    max_base_length = max(20, 80 - len(suffix) - len(digest) - 1)
    if len(normalized) > max_base_length:
        normalized = normalized[:max_base_length].rstrip("._-") or "column"
    return f"{normalized}_{digest}{suffix}"


def create_missing_values_plot(
    dataframe: pd.DataFrame,
    output_dir: str | Path,
    max_columns: int = 30,
) -> Path | None:
    """Create a missing-values bar chart if missing values remain."""

    missing = dataframe.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(max_columns)
    if missing.empty:
        return None

    output_path = ensure_directory(output_dir) / "missing_values.png"
    labels = [_truncate_label(str(column)) for column in missing.index]

    fig, ax = plt.subplots(figsize=(max(8, len(missing) * 0.5), 5))
    ax.bar(labels, missing.values, color="#4C78A8")
    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing values")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_numeric_histogram(
    dataframe: pd.DataFrame,
    column: str,
    output_dir: str | Path,
) -> Path | None:
    """Create a histogram for one numeric column."""

    numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if numeric.empty or numeric.nunique(dropna=True) < 2:
        return None

    output_path = ensure_directory(output_dir) / f"{safe_filename(column, '_histogram')}.png"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(numeric, bins=_histogram_bins(numeric), color="#4C78A8", edgecolor="white")
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_categorical_bar_chart(
    dataframe: pd.DataFrame,
    column: str,
    output_dir: str | Path,
    top_n: int = 10,
) -> Path | None:
    """Create a top-category bar chart for one categorical column."""

    counts = dataframe[column].value_counts(dropna=False).head(top_n)
    if counts.empty or len(counts) < 1:
        return None

    output_path = ensure_directory(output_dir) / f"{safe_filename(column, '_bar')}.png"
    labels = [_truncate_label(_display_value(value), max_length=24) for value in counts.index]

    fig, ax = plt.subplots(figsize=(max(7, len(counts) * 0.7), 4.5))
    ax.bar(labels, counts.values, color="#59A14F")
    ax.set_title(f"Top Categories for {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_correlation_heatmap(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    output_dir: str | Path,
    max_columns: int = 12,
) -> Path | None:
    """Create a simple correlation heatmap for numeric columns."""

    selected_columns = [
        column
        for column in numeric_columns
        if column in dataframe.columns
        and pd.to_numeric(dataframe[column], errors="coerce").nunique(dropna=True) >= 2
    ][:max_columns]
    if len(selected_columns) < 2:
        return None

    numeric_frame = dataframe[selected_columns].apply(pd.to_numeric, errors="coerce")
    correlation = numeric_frame.corr()
    if correlation.dropna(how="all").empty:
        return None

    output_path = ensure_directory(output_dir) / "correlation_heatmap.png"

    fig, ax = plt.subplots(figsize=(max(7, len(selected_columns) * 0.65), 6))
    image = ax.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_title("Numeric Correlation Heatmap")
    ax.set_xticks(range(len(selected_columns)))
    ax.set_yticks(range(len(selected_columns)))
    ax.set_xticklabels([_truncate_label(column, 16) for column in selected_columns])
    ax.set_yticklabels([_truncate_label(column, 16) for column in selected_columns])
    ax.tick_params(axis="x", rotation=45)

    if len(selected_columns) <= 10:
        for row_index in range(len(selected_columns)):
            for column_index in range(len(selected_columns)):
                value = correlation.iloc[row_index, column_index]
                if pd.notna(value):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=8,
                    )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_target_distribution_plot(
    dataframe: pd.DataFrame,
    target_column: str,
    semantic_type: str,
    output_dir: str | Path,
) -> Path | None:
    """Create a target distribution plot for numeric or categorical targets."""

    if target_column not in dataframe.columns:
        return None

    if semantic_type == "numeric":
        return _create_named_histogram(
            dataframe=dataframe,
            column=target_column,
            output_dir=output_dir,
            filename="target_distribution.png",
            title=f"Target Distribution: {target_column}",
        )

    return _create_named_bar_chart(
        dataframe=dataframe,
        column=target_column,
        output_dir=output_dir,
        filename="target_distribution.png",
        title=f"Target Class Distribution: {target_column}",
        top_n=20,
    )


def create_numeric_target_scatter_plot(
    dataframe: pd.DataFrame,
    feature_column: str,
    target_column: str,
    output_dir: str | Path,
) -> Path | None:
    """Create a scatter plot for a numeric feature against a numeric target."""

    values = pd.DataFrame(
        {
            feature_column: pd.to_numeric(dataframe[feature_column], errors="coerce"),
            target_column: pd.to_numeric(dataframe[target_column], errors="coerce"),
        }
    ).dropna()

    if len(values) < 2:
        return None

    output_path = (
        ensure_directory(output_dir)
        / f"{safe_filename(feature_column)}_vs_{safe_filename(target_column)}_scatter.png"
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(values[feature_column], values[target_column], alpha=0.7, color="#4C78A8")
    ax.set_title(f"{feature_column} vs {target_column}")
    ax.set_xlabel(feature_column)
    ax.set_ylabel(target_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_numeric_feature_by_categorical_target_boxplot(
    dataframe: pd.DataFrame,
    feature_column: str,
    target_column: str,
    output_dir: str | Path,
    max_classes: int = 10,
) -> Path | None:
    """Create a boxplot for a numeric feature grouped by a categorical target."""

    values = dataframe[[feature_column, target_column]].copy()
    values[feature_column] = pd.to_numeric(values[feature_column], errors="coerce")
    values = values.dropna()
    if values.empty:
        return None

    top_classes = values[target_column].value_counts().head(max_classes).index.tolist()
    grouped_values = [
        values.loc[values[target_column] == target_value, feature_column].dropna()
        for target_value in top_classes
    ]
    grouped_values = [group for group in grouped_values if not group.empty]
    if not grouped_values:
        return None

    output_path = (
        ensure_directory(output_dir)
        / f"{safe_filename(feature_column)}_by_{safe_filename(target_column)}_boxplot.png"
    )
    fig, ax = plt.subplots(figsize=(max(7, len(grouped_values) * 0.8), 4.5))
    labels = [
        _truncate_label(_display_value(value), 18) for value in top_classes[: len(grouped_values)]
    ]
    try:
        ax.boxplot(grouped_values, tick_labels=labels)
    except TypeError:
        ax.boxplot(grouped_values, labels=labels)
    ax.set_title(f"{feature_column} by {target_column}")
    ax.set_xlabel(target_column)
    ax.set_ylabel(feature_column)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_grouped_bar_chart(
    dataframe: pd.DataFrame,
    feature_column: str,
    target_column: str,
    output_dir: str | Path,
    top_n: int = 8,
) -> Path | None:
    """Create a grouped count chart for a categorical feature and target."""

    values = dataframe[[feature_column, target_column]].dropna()
    if values.empty:
        return None

    top_features = values[feature_column].value_counts().head(top_n).index.tolist()
    filtered = values[values[feature_column].isin(top_features)]
    table = pd.crosstab(filtered[feature_column], filtered[target_column])
    if table.empty or table.shape[1] > 10:
        return None

    output_path = (
        ensure_directory(output_dir)
        / f"{safe_filename(feature_column)}_by_{safe_filename(target_column)}_bar.png"
    )
    fig, ax = plt.subplots(figsize=(max(7, len(table.index) * 0.8), 4.5))
    table.plot(kind="bar", ax=ax)
    ax.set_title(f"{feature_column} by {target_column}")
    ax.set_xlabel(feature_column)
    ax.set_ylabel("Count")
    ax.set_xticklabels([_truncate_label(_display_value(value), 18) for value in table.index])
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title=target_column)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _create_named_histogram(
    dataframe: pd.DataFrame,
    column: str,
    output_dir: str | Path,
    filename: str,
    title: str,
) -> Path | None:
    numeric = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if numeric.empty or numeric.nunique(dropna=True) < 2:
        return None

    output_path = ensure_directory(output_dir) / filename
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(numeric, bins=_histogram_bins(numeric), color="#4C78A8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _create_named_bar_chart(
    dataframe: pd.DataFrame,
    column: str,
    output_dir: str | Path,
    filename: str,
    title: str,
    top_n: int,
) -> Path | None:
    counts = dataframe[column].value_counts(dropna=False).head(top_n)
    if counts.empty:
        return None

    output_path = ensure_directory(output_dir) / filename
    labels = [_truncate_label(_display_value(value), 24) for value in counts.index]
    fig, ax = plt.subplots(figsize=(max(7, len(counts) * 0.7), 4.5))
    ax.bar(labels, counts.values, color="#59A14F")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _histogram_bins(series: pd.Series) -> int:
    return min(30, max(5, int(series.nunique(dropna=True) ** 0.5) + 1))


def _display_value(value: Any) -> str:
    if pd.isna(value):
        return "Missing"
    return str(value)


def _truncate_label(value: str, max_length: int = 20) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."
