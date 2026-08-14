"""Exploratory data analysis service."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.backend.schemas.eda import EDAFindings, EDAPlotInfo, EDARequest, EDAResponse, EDASummary
from app.backend.services.run_manager import RunManager
from app.tools.artifact_lineage import (
    fingerprint_payload,
    invalidate_downstream_artifacts,
    lineage_context,
    select_analysis_input,
    validate_artifact_for_state,
    write_artifact_lineage,
)
from app.tools.app_logging import get_logger, log_event
from app.tools.data_loader import load_csv
from app.tools.eda_analysis import (
    analyze_target_distribution,
    compute_correlation_summary,
    detect_outlier_patterns,
    detect_skewness_patterns,
    generate_recommended_next_steps,
    summarize_categorical_columns,
    summarize_numeric_columns,
)
from app.tools.file_utils import ensure_directory, load_json, save_json, write_text_atomic
from app.tools.schema_inference import infer_schema
from app.tools.statistics_utils import safe_correlation, to_json_safe
from app.tools.visualization import (
    create_categorical_bar_chart,
    create_correlation_heatmap,
    create_grouped_bar_chart,
    create_missing_values_plot,
    create_numeric_feature_by_categorical_target_boxplot,
    create_numeric_histogram,
    create_numeric_target_scatter_plot,
    create_target_distribution_plot,
)
from app.workflows.workflow_steps import EDA_STEP


RAW_DATASET_WARNING = (
    "Cleaned dataset was not found. EDA was generated from the raw uploaded dataset."
)
TARGET_SKIPPED_NOTE = (
    "Target-specific analysis was skipped because no target column was provided."
)


class EDAService:
    """Generate and load deterministic EDA artifacts for analysis runs."""

    def __init__(self, run_manager: RunManager | None = None) -> None:
        self.run_manager = run_manager or RunManager()
        self.logger = get_logger(__name__)

    def eda_summary_path(self, run_id: str) -> Path:
        """Return the EDA summary JSON artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "eda_summary.json"

    def eda_findings_path(self, run_id: str) -> Path:
        """Return the EDA findings JSON artifact path."""

        return self.run_manager.get_paths(run_id).intermediate / "eda_findings.json"

    def eda_report_path(self, run_id: str) -> Path:
        """Return the deterministic Markdown EDA report path."""

        return self.run_manager.get_paths(run_id).reports / "eda_summary.md"

    def generate_eda(
        self,
        run_id: str,
        request: EDARequest | None = None,
    ) -> EDAResponse:
        """Generate, save, and return EDA artifacts for one run."""

        options = request or EDARequest()
        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        invalidate_downstream_artifacts(self.run_manager, run_id, EDA_STEP)
        dataset_selection = self._select_dataset(run_id, options.target_column)
        dataset_path = dataset_selection.path
        dataset_used = dataset_selection.dataset_used
        warnings = list(dataset_selection.warnings)
        dataframe = load_csv(dataset_path)
        schema = infer_schema(dataframe)

        if options.target_column:
            if options.target_column not in dataframe.columns:
                raise ValueError(
                    f"Target column '{options.target_column}' was not found in the dataset."
                )
            target_column = options.target_column
        else:
            target_column = None
            warnings.append(TARGET_SKIPPED_NOTE)

        columns_by_type = self._columns_by_type(dataframe, schema)
        useful_numeric_columns = self._useful_numeric_columns(dataframe, schema)
        useful_categorical_columns = self._useful_categorical_columns(dataframe, schema)

        eda_plots_dir = paths.plots / "eda"
        self._reset_plot_directory(eda_plots_dir, paths.root)
        generated_plots = self._generate_plots(
            dataframe=dataframe,
            run_root=paths.root,
            plots_dir=eda_plots_dir,
            schema=schema,
            useful_numeric_columns=useful_numeric_columns,
            useful_categorical_columns=useful_categorical_columns,
            target_column=target_column,
            options=options,
        )

        correlation_summary = compute_correlation_summary(dataframe, useful_numeric_columns)
        target_analysis = (
            analyze_target_distribution(dataframe, target_column, schema)
            if target_column
            else None
        )

        findings_payload = self._build_findings(
            dataframe=dataframe,
            schema=schema,
            useful_numeric_columns=useful_numeric_columns,
            useful_categorical_columns=useful_categorical_columns,
            correlation_summary=correlation_summary,
            target_analysis=target_analysis,
            target_column=target_column,
        )

        summary_payload: dict[str, Any] = {
            "run_id": run_id,
            "dataset_used": dataset_used,
            "dataset_path": dataset_path.relative_to(paths.root).as_posix(),
            "dataset_fingerprint": dataset_selection.fingerprint,
            "source_fingerprint": dataset_selection.source_fingerprint,
            "target_column": target_column,
            "rows": int(dataframe.shape[0]),
            "columns": int(dataframe.shape[1]),
            "numeric_columns": columns_by_type["numeric"],
            "categorical_columns": columns_by_type["categorical"],
            "boolean_columns": columns_by_type["boolean"],
            "datetime_columns": columns_by_type["datetime"],
            "text_columns": columns_by_type["text"],
            "id_columns": columns_by_type["id"],
            "missing_values_remaining": {
                str(column): int(count)
                for column, count in dataframe.isna().sum().items()
            },
            "duplicate_rows_remaining": int(dataframe.duplicated().sum()),
            "generated_plots": [
                plot.model_dump(mode="json") for plot in generated_plots
            ],
            "key_statistics": {
                "numeric": summarize_numeric_columns(dataframe, schema),
                "categorical": summarize_categorical_columns(dataframe, schema),
                "correlations": correlation_summary,
                "target": target_analysis,
            },
            "warnings": warnings,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        summary = EDASummary(**to_json_safe(summary_payload))
        findings_payload["recommended_next_steps"] = generate_recommended_next_steps(
            profile=None,
            eda_summary=summary.model_dump(mode="json"),
            findings=findings_payload,
        )
        findings = EDAFindings(**to_json_safe(findings_payload))

        summary_path = save_json(self.eda_summary_path(run_id), summary.model_dump(mode="json"))
        findings_path = save_json(self.eda_findings_path(run_id), findings.model_dump(mode="json"))
        self._save_markdown_report(
            path=self.eda_report_path(run_id),
            summary=summary,
            findings=findings,
        )
        context = lineage_context(self.run_manager, run_id)
        upstream_key = dataset_selection.dataset_used
        config_payload = {
            "artifact_family": "eda",
            "source_fingerprint": dataset_selection.source_fingerprint,
            "analysis_input": dataset_selection.dataset_used,
            "analysis_input_fingerprint": dataset_selection.fingerprint,
            "target_column": target_column,
            "max_numeric_plots": options.max_numeric_plots,
            "max_categorical_plots": options.max_categorical_plots,
            "max_target_relationship_plots": options.max_target_relationship_plots,
        }
        config_fingerprint = fingerprint_payload(config_payload)
        upstream_fingerprints = {
            "source_data": dataset_selection.source_fingerprint,
            upstream_key: dataset_selection.fingerprint,
        }
        for artifact_path, artifact_type in (
            (summary_path, "eda_summary"),
            (findings_path, "eda_findings"),
            (self.eda_report_path(run_id), "eda_report"),
        ):
            write_artifact_lineage(
                artifact_path,
                run_root=paths.root,
                run_id=run_id,
                artifact_type=artifact_type,
                generation_id=context["generation_id"],
                source_fingerprint=dataset_selection.source_fingerprint,
                target_column=target_column,
                config_fingerprint=config_fingerprint,
                upstream_fingerprints=upstream_fingerprints,
                relevant_config=config_payload,
            )
        log_event(
            self.logger,
            logging.INFO,
            "EDA generated.",
            run_id=run_id,
            dataset_used=summary.dataset_used,
            plots=len(summary.generated_plots),
            target_column=summary.target_column or "none",
        )

        return EDAResponse(summary=summary, findings=findings)

    def load_eda(self, run_id: str) -> EDAResponse:
        """Load saved EDA summary and findings for one run."""

        summary_path = self.eda_summary_path(run_id)
        findings_path = self.eda_findings_path(run_id)
        if not summary_path.exists() or not findings_path.exists():
            missing_path = summary_path if not summary_path.exists() else findings_path
            raise FileNotFoundError(missing_path)
        state = lineage_context(self.run_manager, run_id)["state"]
        for artifact_path, artifact_type in (
            (summary_path, "eda_summary"),
            (findings_path, "eda_findings"),
        ):
            validation = validate_artifact_for_state(
                artifact_path,
                artifact_type=artifact_type,
                state=state,
            )
            if not validation.is_current:
                raise ValueError(f"EDA artifact is stale: {validation.reason}.")

        return EDAResponse(
            summary=EDASummary(**load_json(summary_path)),
            findings=EDAFindings(**load_json(findings_path)),
        )

    def list_plots(self, run_id: str) -> list[EDAPlotInfo]:
        """Return generated plot files for a run."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        if not paths.plots.exists():
            return []

        plots = [
            self._plot_info_from_path(path, paths.root)
            for path in sorted(paths.plots.rglob("*.png"))
            if path.is_file()
        ]
        return plots

    def resolve_plot_path(self, run_id: str, plot_path: str) -> Path:
        """Resolve a requested plot path while keeping it inside the run plots folder."""

        paths = self.run_manager.get_paths(run_id)
        if not paths.root.exists():
            raise FileNotFoundError(paths.root)

        normalized_plot_path = plot_path.replace("\\", "/")
        if normalized_plot_path.startswith("plots/"):
            normalized_plot_path = normalized_plot_path.removeprefix("plots/")

        candidate = (paths.plots / normalized_plot_path).resolve()
        plots_root = paths.plots.resolve()
        if plots_root != candidate and plots_root not in candidate.parents:
            raise ValueError("Plot path must resolve inside the run plots directory.")

        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(candidate)

        return candidate

    def _select_dataset(self, run_id: str, target_column: str | None):
        selection = select_analysis_input(
            self.run_manager,
            run_id,
            target_column=target_column,
            require_cleaned=False,
        )
        if selection.dataset_used == "raw":
            selection.warnings.insert(0, RAW_DATASET_WARNING)
        return selection

    def _reset_plot_directory(self, plots_dir: Path, run_root: Path) -> None:
        plots_root = plots_dir.resolve()
        resolved_run_root = run_root.resolve()
        if plots_root != resolved_run_root and resolved_run_root in plots_root.parents:
            if plots_root.exists():
                shutil.rmtree(plots_root)
        ensure_directory(plots_root)

    def _generate_plots(
        self,
        dataframe: pd.DataFrame,
        run_root: Path,
        plots_dir: Path,
        schema: dict[str, str],
        useful_numeric_columns: list[str],
        useful_categorical_columns: list[str],
        target_column: str | None,
        options: EDARequest,
    ) -> list[EDAPlotInfo]:
        generated: list[EDAPlotInfo] = []

        missing_path = create_missing_values_plot(dataframe, plots_dir)
        self._append_plot(generated, missing_path, run_root, "Missing Values", "missing_values")

        numeric_dir = ensure_directory(plots_dir / "numeric_distributions")
        for column in useful_numeric_columns[: options.max_numeric_plots]:
            path = create_numeric_histogram(dataframe, column, numeric_dir)
            self._append_plot(
                generated,
                path,
                run_root,
                f"{column} Histogram",
                "numeric_distribution",
            )

        categorical_dir = ensure_directory(plots_dir / "categorical_distributions")
        for column in useful_categorical_columns[: options.max_categorical_plots]:
            path = create_categorical_bar_chart(dataframe, column, categorical_dir)
            self._append_plot(
                generated,
                path,
                run_root,
                f"{column} Category Counts",
                "categorical_distribution",
            )

        heatmap_path = create_correlation_heatmap(dataframe, useful_numeric_columns, plots_dir)
        self._append_plot(
            generated,
            heatmap_path,
            run_root,
            "Correlation Heatmap",
            "correlation_heatmap",
        )

        if target_column:
            target_dir = ensure_directory(plots_dir / "target_relationships")
            target_type = schema.get(target_column, "unknown")
            target_plot = create_target_distribution_plot(
                dataframe,
                target_column,
                target_type,
                target_dir,
            )
            self._append_plot(
                generated,
                target_plot,
                run_root,
                f"{target_column} Target Distribution",
                "target_relationship",
            )

            relationship_paths = self._generate_target_relationship_plots(
                dataframe=dataframe,
                schema=schema,
                useful_numeric_columns=useful_numeric_columns,
                useful_categorical_columns=useful_categorical_columns,
                target_column=target_column,
                target_dir=target_dir,
                max_plots=options.max_target_relationship_plots,
            )
            for path, label in relationship_paths:
                self._append_plot(generated, path, run_root, label, "target_relationship")

        return generated

    def _generate_target_relationship_plots(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
        useful_numeric_columns: list[str],
        useful_categorical_columns: list[str],
        target_column: str,
        target_dir: Path,
        max_plots: int,
    ) -> list[tuple[Path | None, str]]:
        if max_plots <= 0:
            return []

        target_type = schema.get(target_column, "unknown")
        paths: list[tuple[Path | None, str]] = []

        if target_type == "numeric":
            feature_correlations = []
            for column in useful_numeric_columns:
                if column == target_column:
                    continue
                correlation = safe_correlation(dataframe[column], dataframe[target_column])
                if correlation is not None:
                    feature_correlations.append((column, abs(correlation)))

            feature_correlations.sort(key=lambda item: item[1], reverse=True)
            selected_features = [column for column, _ in feature_correlations[:max_plots]]
            for column in selected_features:
                paths.append(
                    (
                        create_numeric_target_scatter_plot(
                            dataframe,
                            column,
                            target_column,
                            target_dir,
                        ),
                        f"{column} vs {target_column}",
                    )
                )
            return paths

        target_unique = int(dataframe[target_column].nunique(dropna=True))
        if target_unique > 10:
            return paths

        selected_numeric = [
            column
            for column in useful_numeric_columns
            if column != target_column
        ][:max_plots]
        for column in selected_numeric:
            paths.append(
                (
                    create_numeric_feature_by_categorical_target_boxplot(
                        dataframe,
                        column,
                        target_column,
                        target_dir,
                    ),
                    f"{column} by {target_column}",
                )
            )

        remaining_slots = max_plots - len(paths)
        selected_categorical = [
            column
            for column in useful_categorical_columns
            if column != target_column
        ][:remaining_slots]
        for column in selected_categorical:
            paths.append(
                (
                    create_grouped_bar_chart(
                        dataframe,
                        column,
                        target_column,
                        target_dir,
                    ),
                    f"{column} by {target_column}",
                )
            )

        return paths

    def _build_findings(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
        useful_numeric_columns: list[str],
        useful_categorical_columns: list[str],
        correlation_summary: dict[str, Any],
        target_analysis: dict[str, Any] | None,
        target_column: str | None,
    ) -> dict[str, list[str]]:
        univariate_findings = detect_skewness_patterns(dataframe, useful_numeric_columns)
        univariate_findings.extend(detect_outlier_patterns(dataframe, useful_numeric_columns))
        univariate_findings.extend(
            self._categorical_cardinality_findings(
                dataframe,
                useful_categorical_columns,
            )
        )

        correlation_findings = [
            f"Columns `{pair['column_a']}` and `{pair['column_b']}` have a "
            f"{pair['strength']} {pair['direction']} correlation ({pair['correlation']:.2f})."
            for pair in correlation_summary.get("pairs", [])
        ]

        target_findings = []
        if target_analysis is not None:
            target_findings.extend(target_analysis.get("findings", []))
            target_findings.extend(
                self._target_relationship_findings(
                    dataframe,
                    schema,
                    useful_numeric_columns,
                    target_column,
                )
            )

        data_quality_notes = self._data_quality_notes(dataframe, schema)

        return {
            "univariate_findings": univariate_findings,
            "bivariate_findings": correlation_findings[:10],
            "target_findings": target_findings,
            "correlation_findings": correlation_findings,
            "data_quality_notes": data_quality_notes,
            "recommended_next_steps": [],
        }

    def _target_relationship_findings(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
        useful_numeric_columns: list[str],
        target_column: str | None,
    ) -> list[str]:
        if not target_column:
            return []

        target_type = schema.get(target_column, "unknown")
        if target_type != "numeric":
            return []

        findings: list[str] = []
        correlations = []
        for column in useful_numeric_columns:
            if column == target_column:
                continue
            correlation = safe_correlation(dataframe[column], dataframe[target_column])
            if correlation is not None and abs(correlation) >= 0.5:
                correlations.append((column, correlation))

        correlations.sort(key=lambda item: abs(item[1]), reverse=True)
        for column, correlation in correlations[:5]:
            direction = "positive" if correlation > 0 else "negative"
            findings.append(
                f"Feature `{column}` has a {direction} correlation with target "
                f"`{target_column}` ({correlation:.2f})."
            )

        return findings

    def _data_quality_notes(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
    ) -> list[str]:
        notes: list[str] = []
        missing = dataframe.isna().sum()
        missing_columns = [str(column) for column, count in missing.items() if int(count) > 0]
        if missing_columns:
            notes.append("Several columns still contain missing values after cleaning or upload.")

        duplicate_rows = int(dataframe.duplicated().sum())
        if duplicate_rows:
            notes.append(f"{duplicate_rows} duplicate rows remain in the EDA dataset.")

        for column in dataframe.columns:
            column_name = str(column)
            if schema.get(column_name) not in {"categorical", "text"}:
                continue
            non_null = int(dataframe[column_name].notna().sum())
            if not non_null:
                continue
            unique_values = int(dataframe[column_name].nunique(dropna=True))
            unique_ratio = unique_values / non_null
            if unique_values >= 50 or (unique_values >= 20 and unique_ratio >= 0.8):
                notes.append(
                    f"Column `{column_name}` has high cardinality and may need special handling before modeling."
                )

        id_columns = [
            str(column)
            for column in dataframe.columns
            if schema.get(str(column)) == "id"
        ]
        if id_columns:
            notes.append(
                "ID-like columns were excluded from automatic distribution plots."
            )

        return self._deduplicate(notes)

    def _categorical_cardinality_findings(
        self,
        dataframe: pd.DataFrame,
        categorical_columns: list[str],
    ) -> list[str]:
        findings: list[str] = []
        for column in categorical_columns:
            unique_values = int(dataframe[column].nunique(dropna=True))
            non_null = int(dataframe[column].notna().sum())
            if not non_null:
                continue
            unique_ratio = unique_values / non_null
            if unique_values >= 20 and unique_ratio >= 0.5:
                findings.append(
                    f"Column `{column}` has relatively high cardinality for a categorical field."
                )
        return findings

    def _columns_by_type(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
    ) -> dict[str, list[str]]:
        semantic_types = ["numeric", "categorical", "boolean", "datetime", "text", "id"]
        return {
            semantic_type: [
                str(column)
                for column in dataframe.columns
                if schema.get(str(column), "unknown") == semantic_type
            ]
            for semantic_type in semantic_types
        }

    def _useful_numeric_columns(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
    ) -> list[str]:
        columns: list[str] = []
        for column in dataframe.columns:
            column_name = str(column)
            if schema.get(column_name) != "numeric":
                continue
            unique_values = int(pd.to_numeric(dataframe[column_name], errors="coerce").nunique(dropna=True))
            if unique_values < 3:
                continue
            columns.append(column_name)
        return columns

    def _useful_categorical_columns(
        self,
        dataframe: pd.DataFrame,
        schema: dict[str, str],
    ) -> list[str]:
        columns: list[str] = []
        row_count = max(int(dataframe.shape[0]), 1)
        for column in dataframe.columns:
            column_name = str(column)
            if schema.get(column_name) not in {"categorical", "boolean"}:
                continue

            unique_values = int(dataframe[column_name].nunique(dropna=True))
            if unique_values < 1:
                continue

            unique_ratio = unique_values / row_count
            if unique_values > 50 and unique_ratio > 0.5:
                continue

            columns.append(column_name)
        return columns

    def _plot_info_from_path(self, path: Path, run_root: Path) -> EDAPlotInfo:
        relative_path = path.relative_to(run_root).as_posix()
        category = self._plot_category(relative_path)
        return EDAPlotInfo(
            path=relative_path,
            label=self._plot_label(path),
            category=category,
        )

    def _append_plot(
        self,
        generated: list[EDAPlotInfo],
        path: Path | None,
        run_root: Path,
        label: str,
        category: str,
    ) -> None:
        if path is None:
            return
        generated.append(
            EDAPlotInfo(
                path=path.relative_to(run_root).as_posix(),
                label=label,
                category=category,
            )
        )

    def _plot_category(self, relative_path: str) -> str:
        if "numeric_distributions/" in relative_path:
            return "numeric_distribution"
        if "categorical_distributions/" in relative_path:
            return "categorical_distribution"
        if "target_relationships/" in relative_path:
            return "target_relationship"
        if relative_path.endswith("missing_values.png"):
            return "missing_values"
        if relative_path.endswith("correlation_heatmap.png"):
            return "correlation_heatmap"
        return "plot"

    def _plot_label(self, path: Path) -> str:
        return path.stem.replace("_", " ").title()

    def _save_markdown_report(
        self,
        path: Path,
        summary: EDASummary,
        findings: EDAFindings,
    ) -> None:
        ensure_directory(path.parent)
        lines = [
            f"# Exploratory Data Analysis Summary: {summary.run_id}",
            "",
            f"- Dataset used: {summary.dataset_used}",
            f"- Dataset path: `{summary.dataset_path}`",
            f"- Dataset shape: {summary.rows} rows x {summary.columns} columns",
            f"- Target column: {summary.target_column or 'None provided'}",
            "",
            "## Column Type Summary",
            "",
            f"- Numeric columns: {len(summary.numeric_columns)}",
            f"- Categorical columns: {len(summary.categorical_columns)}",
            f"- Boolean columns: {len(summary.boolean_columns)}",
            f"- Datetime columns: {len(summary.datetime_columns)}",
            f"- Text columns: {len(summary.text_columns)}",
            f"- ID-like columns: {len(summary.id_columns)}",
            "",
            "## Data Quality Notes",
            "",
            *self._markdown_list(findings.data_quality_notes),
            "",
            "## Key Univariate Findings",
            "",
            *self._markdown_list(findings.univariate_findings),
            "",
            "## Bivariate And Correlation Findings",
            "",
            *self._markdown_list(findings.correlation_findings),
            "",
            "## Target-Specific Findings",
            "",
            *self._markdown_list(findings.target_findings),
            "",
            "## Generated Plots",
            "",
            *self._markdown_list(
                [f"{plot.label}: `{plot.path}`" for plot in summary.generated_plots]
            ),
            "",
            "## Recommended Next Steps",
            "",
            *self._markdown_list(findings.recommended_next_steps),
            "",
            "## Limitations",
            "",
            "- EDA findings are deterministic heuristics, not causal claims.",
            "- EDA does not by itself prove causality or validate production model readiness.",
        ]
        write_text_atomic(path, "\n".join(lines) + "\n")

    def _markdown_list(self, values: list[str]) -> list[str]:
        if not values:
            return ["- None."]
        return [f"- {value}" for value in values]

    def _deduplicate(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduplicated: list[str] = []
        for value in values:
            if value not in seen:
                seen.add(value)
                deduplicated.append(value)
        return deduplicated
