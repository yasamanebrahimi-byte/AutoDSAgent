"""Reusable Markdown section builders for final reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


Artifacts = Mapping[str, Any]


@dataclass(frozen=True)
class ReportSection:
    """One generated or explicitly skipped report section."""

    name: str
    title: str
    markdown: str
    generated: bool = True
    skipped_reason: str | None = None
    warnings: tuple[str, ...] = ()


def build_title_section(
    run_id: str,
    metadata: Mapping[str, Any] | None = None,
    report_status: str = "partial",
) -> ReportSection:
    """Build the final report title block."""

    filename = _value(metadata, "filename") if metadata else None
    subtitle = f"Dataset: `{filename}`" if filename else f"Run: `{run_id}`"
    markdown = "\n".join(
        [
            "# AutoDS Agent Final Analysis Report",
            "",
            subtitle,
            f"Report status: **{report_status}**",
        ]
    )
    return ReportSection("title", "Title", markdown)


def build_run_metadata_section(run_id: str, artifacts: Artifacts) -> ReportSection:
    """Build run metadata details."""

    metadata = _artifact(artifacts, "metadata")
    workflow_state = _artifact(artifacts, "workflow_state")
    if not metadata and not workflow_state:
        return _skipped(
            "run_metadata",
            "Run Metadata",
            "metadata.json and workflow_state.json are unavailable.",
        )

    rows = [
        ("Run ID", run_id),
        ("Uploaded file", _value(metadata, "filename")),
        ("Rows", _value(metadata, "rows")),
        ("Columns", _value(metadata, "columns")),
        ("Created at", _value(metadata, "created_at")),
        ("Workflow status", _value(workflow_state, "status")),
        ("Workflow version", _value(workflow_state, "workflow_version")),
        ("Selected target", _value(workflow_state, "target_column")),
    ]
    markdown = _section(
        "Run Metadata",
        [
            _markdown_table(["Field", "Value"], rows),
        ],
    )
    return ReportSection("run_metadata", "Run Metadata", markdown)


def build_executive_summary_section(artifacts: Artifacts) -> ReportSection:
    """Build a concise business-readable summary section."""

    metadata = _artifact(artifacts, "metadata")
    profile = _artifact(artifacts, "profile")
    cleaning_summary = _artifact(artifacts, "cleaning_summary")
    eda_findings = _artifact(artifacts, "eda_findings")
    modeling_summary = _artifact(artifacts, "modeling_summary")
    evaluation_summary = _artifact(artifacts, "evaluation_summary")

    if not any([metadata, profile, cleaning_summary, eda_findings, modeling_summary]):
        return _skipped(
            "executive_summary",
            "Executive Summary",
            "no analysis artifacts were available.",
        )

    filename = _value(metadata, "filename") or "the uploaded dataset"
    rows = _value(profile, "rows") or _value(metadata, "rows")
    columns = _value(profile, "columns") or _value(metadata, "columns")
    actions = ["profiled the dataset"]
    if cleaning_summary:
        actions.append("applied conservative cleaning")
    if _artifact(artifacts, "eda_summary"):
        actions.append("generated exploratory analysis")
    if modeling_summary:
        actions.append("trained and evaluated baseline candidate models")

    bullets = [
        f"AutoDS Agent analyzed `{filename}`"
        + (f" with {rows} rows and {columns} columns." if rows and columns else "."),
        f"The system {', '.join(actions[:-1]) + ', and ' + actions[-1] if len(actions) > 1 else actions[0]}.",
    ]

    finding = _first_nonempty(
        _as_list(eda_findings.get("target_findings") if eda_findings else []),
        _as_list(eda_findings.get("univariate_findings") if eda_findings else []),
        _as_list(eda_findings.get("data_quality_notes") if eda_findings else []),
    )
    if finding:
        bullets.append(f"Main finding: {finding}")
    else:
        bullets.append("Main finding: no deterministic EDA finding was available in the saved artifacts.")

    if modeling_summary and evaluation_summary:
        primary_metric = _value(modeling_summary, "primary_metric") or _value(
            evaluation_summary,
            "primary_metric",
        )
        best_model = _value(modeling_summary, "best_model_name") or _value(
            evaluation_summary,
            "best_model_name",
        )
        metric_value = _metric_value(
            evaluation_summary.get("best_model_metrics", {}),
            str(primary_metric) if primary_metric else None,
        )
        suffix = f" ({primary_metric}: {metric_value})" if metric_value is not None else ""
        bullets.append(f"Modeling result: `{best_model}` was selected as the best model{suffix}.")
    else:
        bullets.append("Modeling result: modeling artifacts were not available, so this report does not claim model performance.")

    warnings = _collect_artifact_warnings(artifacts)
    if warnings:
        bullets.append(f"Important warning: {warnings[0]}")

    next_step = _first_nonempty(
        _as_list(eda_findings.get("recommended_next_steps") if eda_findings else []),
        ["Review the partial report and generate missing analysis artifacts before making decisions."],
    )
    bullets.append(f"Recommended next step: {next_step}")

    markdown = _section("Executive Summary", _markdown_list(bullets))
    return ReportSection("executive_summary", "Executive Summary", markdown)


def build_dataset_overview_section(artifacts: Artifacts) -> ReportSection:
    """Build dataset shape, column, and type overview."""

    metadata = _artifact(artifacts, "metadata")
    profile = _artifact(artifacts, "profile")
    if not metadata and not profile:
        return _skipped(
            "dataset_overview",
            "Dataset Overview",
            "metadata.json and profile.json are unavailable.",
        )

    rows = _value(profile, "rows") or _value(metadata, "rows")
    columns = _value(profile, "columns") or _value(metadata, "columns")
    duplicate_rows = _value(profile, "duplicate_rows") or _value(metadata, "duplicate_rows")
    missing_values = _value(profile, "total_missing_values")
    column_names = _as_list(_value(metadata, "column_names"))
    column_type_counts = _value(profile, "column_type_counts") or {}

    parts = [
        _markdown_table(
            ["Metric", "Value"],
            [
                ("Rows", rows),
                ("Columns", columns),
                ("Duplicate rows", duplicate_rows),
                ("Total missing values", missing_values),
            ],
        )
    ]
    if column_type_counts:
        parts.extend(
            [
                "",
                "Column type summary:",
                _markdown_table(
                    ["Semantic Type", "Count"],
                    sorted(column_type_counts.items()),
                ),
            ]
        )
    if column_names:
        parts.extend(
            [
                "",
                "Columns included in the uploaded dataset:",
                ", ".join(f"`{column}`" for column in column_names[:30])
                + (" ..." if len(column_names) > 30 else ""),
            ]
        )

    return ReportSection(
        "dataset_overview",
        "Dataset Overview",
        _section("Dataset Overview", parts),
    )


def build_data_quality_section(artifacts: Artifacts) -> ReportSection:
    """Build data quality summary from profile, cleaning, and EDA artifacts."""

    profile = _artifact(artifacts, "profile")
    cleaning_summary = _artifact(artifacts, "cleaning_summary")
    eda_summary = _artifact(artifacts, "eda_summary")
    if not any([profile, cleaning_summary, eda_summary]):
        return _skipped(
            "data_quality",
            "Data Quality Summary",
            "profile, cleaning summary, and EDA summary artifacts are unavailable.",
        )

    bullets: list[str] = []
    if profile:
        bullets.append(
            f"Profiled missing values: {_format_value(profile.get('total_missing_values'))}."
        )
        bullets.append(
            f"Profiled duplicate rows: {_format_value(profile.get('duplicate_rows'))}."
        )
        issues = _as_list(profile.get("data_quality_issues"))
        if issues:
            bullets.append(f"Data quality warnings generated: {len(issues)}.")
            for issue in issues[:8]:
                bullets.append(_issue_text(issue))
        else:
            bullets.append("No profile-level data quality warnings were generated.")

        high_missing = [
            column
            for column in _as_list(profile.get("column_profiles"))
            if (missing_percent := _safe_float(column.get("missing_percentage"))) is not None
            and missing_percent >= 20
        ][:8]
        if high_missing:
            bullets.append(
                "Columns with at least 20% missing values: "
                + ", ".join(f"`{column.get('column_name')}`" for column in high_missing)
                + "."
            )

        high_cardinality = [
            column.get("column_name")
            for column in _as_list(profile.get("column_profiles"))
            if column.get("is_high_cardinality")
        ][:8]
        if high_cardinality:
            bullets.append(
                "High-cardinality columns may need careful encoding before modeling: "
                + ", ".join(f"`{column}`" for column in high_cardinality)
                + "."
            )

        id_columns = [
            column.get("column_name")
            for column in _as_list(profile.get("column_profiles"))
            if column.get("is_id")
        ][:8]
        if id_columns:
            bullets.append(
                "ID-like columns were detected and should be treated as identifiers, not predictive evidence: "
                + ", ".join(f"`{column}`" for column in id_columns)
                + "."
            )

    if cleaning_summary:
        bullets.append(
            "Cleaning changed missing values from "
            f"{_format_value(cleaning_summary.get('missing_values_before'))} to "
            f"{_format_value(cleaning_summary.get('missing_values_after'))}."
        )
        if cleaning_summary.get("duplicate_rows_removed"):
            bullets.append(
                f"Duplicate rows removed during cleaning: {cleaning_summary['duplicate_rows_removed']}."
            )

    if eda_summary:
        remaining_missing = {
            column: count
            for column, count in (eda_summary.get("missing_values_remaining") or {}).items()
            if int(count or 0) > 0
        }
        if remaining_missing:
            bullets.append(
                "Missing values remained in the EDA dataset for "
                + ", ".join(f"`{column}` ({count})" for column, count in list(remaining_missing.items())[:8])
                + "."
            )
        else:
            bullets.append("No missing values remained in the EDA dataset.")

    markdown = _section("Data Quality Summary", _markdown_list(bullets))
    return ReportSection("data_quality", "Data Quality Summary", markdown)


def build_cleaning_methodology_section(artifacts: Artifacts) -> ReportSection:
    """Build cleaning methodology from cleaning plan and summary artifacts."""

    plan = _artifact(artifacts, "cleaning_plan")
    summary = _artifact(artifacts, "cleaning_summary")
    if not plan and not summary:
        return _skipped(
            "cleaning_methodology",
            "Cleaning Methodology",
            "cleaning_plan.json and cleaning_summary.json are unavailable.",
        )

    parts: list[str] = []
    if plan:
        duplicate_action = plan.get("duplicate_row_handling") or {}
        parts.extend(
            [
                "Cleaning plan:",
                _markdown_table(
                    ["Decision", "Value"],
                    [
                        ("Duplicate handling", duplicate_action.get("strategy")),
                        ("Duplicate action applied", duplicate_action.get("apply")),
                        ("Duplicate rationale", duplicate_action.get("reason")),
                    ],
                ),
            ]
        )
        missing_actions = _as_list(plan.get("missing_value_strategies"))
        if missing_actions:
            parts.extend(
                [
                    "",
                    "Missing value strategies:",
                    _markdown_table(
                        ["Column", "Strategy", "Applied", "Reason"],
                        [
                            (
                                action.get("column"),
                                action.get("strategy"),
                                action.get("apply"),
                                action.get("reason"),
                            )
                            for action in missing_actions
                        ],
                    ),
                ]
            )
        drops = _as_list(plan.get("columns_recommended_for_dropping"))
        if drops:
            parts.extend(["", "Columns recommended for dropping: " + ", ".join(f"`{item}`" for item in drops) + "."])
        review_warnings = _as_list(plan.get("warnings_requiring_review"))
        if review_warnings:
            parts.extend(["", "Warnings requiring review:", *_markdown_list(review_warnings)])

    if summary:
        parts.extend(
            [
                "",
                "Cleaning execution summary:",
                _markdown_table(
                    ["Metric", "Value"],
                    [
                        ("Original shape", _shape_text(summary.get("original_shape"))),
                        ("Cleaned shape", _shape_text(summary.get("cleaned_shape"))),
                        ("Duplicate rows removed", summary.get("duplicate_rows_removed")),
                        ("Missing values before", summary.get("missing_values_before")),
                        ("Missing values after", summary.get("missing_values_after")),
                    ],
                ),
            ]
        )
        imputations = summary.get("imputation_strategies_used") or {}
        if imputations:
            parts.extend(
                [
                    "",
                    "Imputation strategies applied:",
                    _markdown_table(["Column", "Strategy"], sorted(imputations.items())),
                ]
            )
        else:
            parts.extend(
                [
                    "",
                    "No learned imputation was applied during structural cleaning. "
                    "Model pipelines learn missing-value preprocessing from training data.",
                ]
            )
        if summary.get("warnings"):
            parts.extend(["", "Cleaning warnings:", *_markdown_list(summary.get("warnings", []))])

    return ReportSection(
        "cleaning_methodology",
        "Cleaning Methodology",
        _section("Cleaning Methodology", parts),
    )


def build_eda_findings_section(artifacts: Artifacts) -> ReportSection:
    """Build EDA findings from saved EDA summary and findings artifacts."""

    summary = _artifact(artifacts, "eda_summary")
    findings = _artifact(artifacts, "eda_findings")
    if not summary and not findings:
        return _skipped(
            "eda_findings",
            "Exploratory Data Analysis",
            "eda_summary.json and eda_findings.json are unavailable.",
        )

    parts: list[str] = []
    if summary:
        parts.append(
            _markdown_table(
                ["Metric", "Value"],
                [
                    ("Dataset used", summary.get("dataset_used")),
                    ("Rows", summary.get("rows")),
                    ("Columns", summary.get("columns")),
                    ("Target column", summary.get("target_column") or "None"),
                    ("Generated plots", len(_as_list(summary.get("generated_plots")))),
                    ("Duplicate rows remaining", summary.get("duplicate_rows_remaining")),
                ],
            )
        )
        if summary.get("warnings"):
            parts.extend(["", "EDA warnings:", *_markdown_list(summary.get("warnings", []))])

    if findings:
        groups = [
            ("Data quality notes", findings.get("data_quality_notes")),
            ("Univariate findings", findings.get("univariate_findings")),
            ("Bivariate findings", findings.get("bivariate_findings")),
            ("Correlation findings", findings.get("correlation_findings")),
            ("Target findings", findings.get("target_findings")),
        ]
        for label, values in groups:
            values_list = _as_list(values)
            if values_list:
                parts.extend(["", f"{label}:", *_markdown_list(values_list[:10])])

    parts.extend(
        [
            "",
            "Interpretation note: EDA patterns are associations in the observed data, not causal claims.",
        ]
    )
    return ReportSection(
        "eda_findings",
        "Exploratory Data Analysis",
        _section("Exploratory Data Analysis", parts),
    )


def build_target_analysis_section(artifacts: Artifacts) -> ReportSection:
    """Build target-specific analysis if a target was selected."""

    eda_summary = _artifact(artifacts, "eda_summary")
    eda_findings = _artifact(artifacts, "eda_findings")
    modeling_summary = _artifact(artifacts, "modeling_summary")
    target_column = (
        _value(eda_summary, "target_column")
        or _value(modeling_summary, "target_column")
        or _value(_artifact(artifacts, "workflow_state"), "target_column")
    )
    if not target_column:
        return _skipped(
            "target_analysis",
            "Target Analysis",
            "no target column was selected for EDA or modeling.",
        )

    parts = [f"Target column: `{target_column}`"]
    target_stats = (
        (eda_summary.get("key_statistics") or {}).get("target")
        if eda_summary
        else None
    )
    if isinstance(target_stats, Mapping):
        rows = [(key, value) for key, value in target_stats.items() if key != "findings"]
        if rows:
            parts.extend(["", _markdown_table(["Target Detail", "Value"], rows[:12])])

    target_findings = _as_list(eda_findings.get("target_findings") if eda_findings else [])
    if target_findings:
        parts.extend(["", "Target-specific findings:", *_markdown_list(target_findings)])
    elif modeling_summary:
        parts.extend(
            [
                "",
                "Modeling used this target column, but no target-specific EDA findings were saved.",
            ]
        )

    return ReportSection(
        "target_analysis",
        "Target Analysis",
        _section("Target Analysis", parts),
    )


def build_modeling_methodology_section(artifacts: Artifacts) -> ReportSection:
    """Build modeling methodology from modeling summary."""

    summary = _artifact(artifacts, "modeling_summary")
    if not summary:
        return _skipped(
            "modeling_methodology",
            "Modeling Methodology",
            "modeling_summary.json is unavailable. Modeling may have been skipped.",
        )

    attempted = _as_list(summary.get("models_attempted"))
    succeeded = set(_as_list(summary.get("models_succeeded")))
    failed = set(_as_list(summary.get("models_failed")))
    model_rows = [
        (
            model_name,
            "failed" if model_name in failed else "completed" if model_name in succeeded else "attempted",
            "baseline" if model_name == summary.get("baseline_model_name") else "candidate",
        )
        for model_name in attempted
    ]
    parts = [
        _markdown_table(
            ["Decision", "Value"],
            [
                ("Target column", summary.get("target_column")),
                ("Task type", summary.get("task_type")),
                ("Rows used", summary.get("rows_used")),
                ("Training rows", summary.get("train_rows")),
                ("Test rows", summary.get("test_rows")),
                ("CV folds", summary.get("cv_folds")),
                ("CV strategy", summary.get("cv_strategy")),
                ("Task inference reason", summary.get("task_inference_reason")),
                ("Primary metric", summary.get("primary_metric")),
                ("Best model selection", _selection_logic(summary.get("task_type"), summary.get("primary_metric"))),
            ],
        ),
        "",
        "Models attempted:",
        _markdown_table(["Model", "Status", "Role"], model_rows),
    ]

    excluded_reasons = summary.get("excluded_feature_reasons") or {}
    if excluded_reasons:
        parts.extend(
            [
                "",
                "Features excluded from modeling:",
                _markdown_table(["Column", "Reason"], sorted(excluded_reasons.items())),
            ]
        )
    if summary.get("warnings"):
        parts.extend(["", "Modeling warnings:", *_markdown_list(summary.get("warnings", []))])

    return ReportSection(
        "modeling_methodology",
        "Modeling Methodology",
        _section("Modeling Methodology", parts),
    )


def build_evaluation_results_section(artifacts: Artifacts) -> ReportSection:
    """Build evaluation results from evaluation summary and model results."""

    summary = _artifact(artifacts, "evaluation_summary")
    model_results = _artifact(artifacts, "model_results")
    if not summary:
        return _skipped(
            "evaluation_results",
            "Evaluation Results",
            "evaluation_summary.json is unavailable. Evaluation results are not reported.",
        )

    primary_metric = summary.get("primary_metric")
    best_metrics = summary.get("best_model_metrics") or {}
    baseline_metrics = summary.get("baseline_metrics") or {}
    all_model_metrics = summary.get("all_model_metrics") or {}
    comparison = summary.get("baseline_comparison") or {}
    parts = [
        _metric_definitions(str(summary.get("task_type"))),
        "",
        "Baseline CV metrics:",
        _markdown_table(["Metric", "Value"], sorted(baseline_metrics.items())),
        "",
        "Selected model holdout metrics:",
        _markdown_table(["Metric", "Value"], sorted(best_metrics.items())),
    ]
    if all_model_metrics:
        metric_names = sorted({metric for metrics in all_model_metrics.values() for metric in metrics})
        parts.extend(
            [
                "",
                "CV model comparison:",
                _markdown_table(
                    ["Model", *[metric.upper() for metric in metric_names]],
                    [
                        (
                            model_name,
                            *[metrics.get(metric) for metric in metric_names],
                        )
                        for model_name, metrics in sorted(all_model_metrics.items())
                    ],
                ),
            ]
        )

    if comparison.get("interpretation"):
        parts.extend(["", f"Baseline comparison: {comparison['interpretation']}"])

    failed_models = _as_list(model_results.get("failed_models") if model_results else [])
    if failed_models:
        parts.extend(
            [
                "",
                "Failed models:",
                _markdown_table(
                    ["Model", "Role", "Error"],
                    [
                        (
                            item.get("model_name"),
                            item.get("role"),
                            item.get("error"),
                        )
                        for item in failed_models
                    ],
                ),
            ]
        )

    if summary.get("warnings"):
        parts.extend(["", "Evaluation warnings:", *_markdown_list(summary.get("warnings", []))])

    if primary_metric:
        parts.extend(["", f"Primary metric used for CV model selection: `{primary_metric}`."])

    evaluated_models = _as_list(summary.get("test_evaluated_model_names"))
    if evaluated_models:
        parts.extend(
            [
                "",
                "Holdout test evaluation was run for:",
                *_markdown_list(evaluated_models),
            ]
        )

    return ReportSection(
        "evaluation_results",
        "Evaluation Results",
        _section("Evaluation Results", parts),
    )


def build_best_model_section(artifacts: Artifacts) -> ReportSection:
    """Build best-model summary when modeling artifacts are available."""

    modeling_summary = _artifact(artifacts, "modeling_summary")
    evaluation_summary = _artifact(artifacts, "evaluation_summary")
    if not modeling_summary and not evaluation_summary:
        return _skipped(
            "best_model",
            "Best Model Summary",
            "modeling and evaluation artifacts are unavailable.",
        )

    best_model = _value(modeling_summary, "best_model_name") or _value(
        evaluation_summary,
        "best_model_name",
    )
    baseline_model = _value(modeling_summary, "baseline_model_name")
    primary_metric = _value(modeling_summary, "primary_metric") or _value(
        evaluation_summary,
        "primary_metric",
    )
    best_value = _metric_value(
        evaluation_summary.get("best_model_metrics", {}) if evaluation_summary else {},
        str(primary_metric) if primary_metric else None,
    )
    comparison = evaluation_summary.get("baseline_comparison") if evaluation_summary else {}
    parts = [
        _markdown_table(
            ["Field", "Value"],
            [
                ("Best model", best_model),
                ("Baseline model", baseline_model),
                ("Primary metric", primary_metric),
                ("Best model primary metric value", best_value),
                ("Baseline absolute improvement", (comparison or {}).get("absolute_improvement")),
                ("Baseline percent improvement", _percent_text((comparison or {}).get("percent_improvement"))),
            ],
        )
    ]
    if (comparison or {}).get("interpretation"):
        parts.extend(["", str(comparison["interpretation"])])
    parts.extend(
        [
            "",
            "Feature importance or model signal should be interpreted as predictive association, not causal effect.",
        ]
    )
    return ReportSection(
        "best_model",
        "Best Model Summary",
        _section("Best Model Summary", parts),
    )


def build_artifacts_section(
    artifacts: Artifacts,
    source_artifacts_used: Sequence[str],
    source_artifacts_missing: Sequence[str],
    report_files: Sequence[Mapping[str, Any]] | None = None,
) -> ReportSection:
    """Build the generated artifacts and source artifact inventory."""

    generated_plots = []
    eda_summary = _artifact(artifacts, "eda_summary")
    evaluation_summary = _artifact(artifacts, "evaluation_summary")
    if eda_summary:
        generated_plots.extend(_as_list(eda_summary.get("generated_plots")))
    if evaluation_summary:
        generated_plots.extend(_as_list(evaluation_summary.get("generated_plots")))

    parts = [
        "Source artifacts used:",
        *_markdown_list(list(source_artifacts_used) or ["None."]),
        "",
        "Source artifacts missing:",
        *_markdown_list(list(source_artifacts_missing) or ["None."]),
    ]
    if generated_plots:
        parts.extend(
            [
                "",
                "Plots referenced by saved summaries:",
                *_markdown_list(
                    [
                        f"{plot.get('label', 'Plot')}: `{plot.get('path')}`"
                        for plot in generated_plots[:40]
                    ]
                ),
            ]
        )
    if report_files:
        parts.extend(
            [
                "",
                "Report artifacts generated:",
                *_markdown_list(
                    [
                        f"{file_info.get('name')}: `{file_info.get('path')}`"
                        for file_info in report_files
                    ]
                ),
            ]
        )

    return ReportSection(
        "artifacts",
        "Artifacts Generated",
        _section("Artifacts Generated", parts),
    )


def build_limitations_section(
    artifacts: Artifacts,
    source_artifacts_missing: Sequence[str] | None = None,
) -> ReportSection:
    """Build limitations from saved warnings, failures, and fixed guardrails."""

    modeling_summary = _artifact(artifacts, "modeling_summary")
    model_results = _artifact(artifacts, "model_results")
    limitations = [
        "This analysis is exploratory and should be reviewed before operational decisions.",
        "Correlations, feature importance, and model signals do not prove causation.",
        "Model results depend on the provided dataset, preprocessing rules, and train/test split.",
        "Heavy hyperparameter tuning was not performed.",
        "Text, time-series, deep learning, and leakage detection are not fully supported yet.",
        "LLM narrative generation was not used by the deterministic report generator.",
    ]

    missing = list(source_artifacts_missing or [])
    if missing:
        limitations.append(
            "Some source artifacts were missing, so unavailable sections are explicitly marked as skipped."
        )

    failed_models = _as_list(model_results.get("failed_models") if model_results else [])
    if failed_models:
        limitations.append(f"{len(failed_models)} model attempt(s) failed and should be reviewed.")

    if modeling_summary and modeling_summary.get("features_excluded"):
        limitations.append(
            "Some features were excluded from modeling, which may omit useful signal or prevent leakage."
        )
    elif not modeling_summary:
        limitations.append("Modeling artifacts were unavailable, so predictive performance is not claimed.")

    limitations.extend(_collect_artifact_warnings(artifacts))

    return ReportSection(
        "limitations",
        "Limitations",
        _section("Limitations", _markdown_list(_dedupe(limitations))),
    )


def build_next_steps_section(artifacts: Artifacts) -> ReportSection:
    """Build recommended next steps from saved EDA recommendations and report status."""

    eda_findings = _artifact(artifacts, "eda_findings")
    modeling_summary = _artifact(artifacts, "modeling_summary")
    recommendations = _as_list(eda_findings.get("recommended_next_steps") if eda_findings else [])
    if not recommendations:
        recommendations = [
            "Generate or review missing profile, cleaning, EDA, and modeling artifacts as needed.",
            "Review data quality warnings and validate cleaning decisions with a domain expert.",
        ]

    if modeling_summary:
        recommendations.append("Compare the selected model against business requirements before deployment.")
        recommendations.append("Add broader validation and tuning only after confirming the target and leakage risks.")
    else:
        recommendations.append("Select a valid target column and run modeling if predictive analysis is needed.")

    recommendations.append("Add experiment tracking, packaging polish, and deployment preparation as needed.")

    return ReportSection(
        "next_steps",
        "Recommended Next Steps",
        _section("Recommended Next Steps", _markdown_list(_dedupe(recommendations))),
    )


def build_appendix_section(
    artifacts: Artifacts,
    source_artifacts_used: Sequence[str],
    source_artifacts_missing: Sequence[str],
) -> ReportSection:
    """Build appendix with artifact paths and workflow trace summary."""

    trace = _artifact(artifacts, "agent_trace")
    workflow_state = _artifact(artifacts, "workflow_state")
    parts = [
        "Artifact path summary:",
        *_markdown_list(
            [
                *[f"Used: `{path}`" for path in source_artifacts_used],
                *[f"Missing: `{path}`" for path in source_artifacts_missing],
            ]
            or ["No source artifact paths were available."]
        ),
    ]

    if workflow_state:
        steps = workflow_state.get("steps") or {}
        parts.extend(
            [
                "",
                "Workflow step summary:",
                _markdown_table(
                    ["Step", "Status", "Attempts", "Error"],
                    [
                        (
                            step_name,
                            step_state.get("status"),
                            step_state.get("attempts"),
                            step_state.get("error"),
                        )
                        for step_name, step_state in steps.items()
                    ],
                ),
            ]
        )
    else:
        parts.extend(["", "Workflow state was unavailable."])

    trace_events = trace if isinstance(trace, list) else []
    if trace_events:
        parts.extend(
            [
                "",
                f"Agent trace events saved: {len(trace_events)}.",
                _markdown_table(
                    ["Timestamp", "Agent", "Step", "Event", "Message"],
                    [
                        (
                            event.get("timestamp"),
                            event.get("agent"),
                            event.get("step"),
                            event.get("event_type"),
                            event.get("message"),
                        )
                        for event in trace_events[-12:]
                    ],
                ),
            ]
        )
    else:
        parts.extend(["", "Agent trace was unavailable or empty."])

    return ReportSection(
        "appendix",
        "Appendix: Workflow Trace Summary",
        _section("Appendix: Workflow Trace Summary", parts),
    )


def _artifact(artifacts: Artifacts, key: str) -> Any:
    value = artifacts.get(key)
    if isinstance(value, Mapping):
        return value
    if key == "agent_trace" and isinstance(value, list):
        return value
    return value if value else None


def _value(mapping: Any, key: str) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _skipped(name: str, title: str, reason: str) -> ReportSection:
    markdown = _section(
        title,
        [f"This section is unavailable because {reason}"],
    )
    return ReportSection(
        name=name,
        title=title,
        markdown=markdown,
        generated=False,
        skipped_reason=reason,
    )


def _section(title: str, parts: Sequence[str | list[str]]) -> str:
    flattened: list[str] = [f"## {title}", ""]
    for part in parts:
        if isinstance(part, list):
            flattened.extend(part)
        else:
            flattened.append(str(part))
    return "\n".join(flattened).rstrip()


def _markdown_list(values: Sequence[Any]) -> list[str]:
    if not values:
        return ["- None."]
    return [f"- {_format_value(value)}" for value in values]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any] | tuple[Any, ...]]) -> str:
    if not rows:
        return "_No rows available._"

    header = "| " + " | ".join(_escape_table_cell(item) for item in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = []
    column_count = len(headers)
    for row in rows:
        values = list(row)
        if len(values) < column_count:
            values.extend([""] * (column_count - len(values)))
        body.append(
            "| "
            + " | ".join(_escape_table_cell(_format_value(item)) for item in values[:column_count])
            + " |"
        )
    return "\n".join([header, separator, *body])


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_value(value: Any) -> str:
    if value is None or value == "":
        return "Unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "None"
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{key}: {_format_value(item)}" for key, item in value.items())
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _first_nonempty(*groups: Sequence[Any]) -> Any:
    for group in groups:
        for item in group:
            if item:
                return item
    return None


def _metric_value(metrics: Mapping[str, Any], primary_metric: str | None) -> Any:
    if not primary_metric:
        return None
    return metrics.get(primary_metric)


def _shape_text(shape: Any) -> str:
    values = _as_list(shape)
    if len(values) >= 2:
        return f"{values[0]} x {values[1]}"
    return _format_value(shape)


def _issue_text(issue: Any) -> str:
    if not isinstance(issue, Mapping):
        return _format_value(issue)
    column = issue.get("column")
    prefix = f"`{column}`: " if column else ""
    return prefix + str(issue.get("message") or issue.get("issue_type") or issue)


def _collect_artifact_warnings(artifacts: Artifacts) -> list[str]:
    warnings: list[str] = []
    for key in [
        "cleaning_summary",
        "eda_summary",
        "modeling_summary",
        "evaluation_summary",
    ]:
        artifact = _artifact(artifacts, key)
        if isinstance(artifact, Mapping):
            warnings.extend(str(item) for item in _as_list(artifact.get("warnings")))
    profile = _artifact(artifacts, "profile")
    if isinstance(profile, Mapping):
        for issue in _as_list(profile.get("data_quality_issues"))[:8]:
            warnings.append(_issue_text(issue))
    return _dedupe(warnings)


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percent_text(value: Any) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "Unavailable"
    return f"{numeric:.2f}%"


def _selection_logic(task_type: Any, primary_metric: Any) -> str:
    if task_type == "regression":
        return f"Lowest mean CV {primary_metric} among successful candidate models."
    if task_type == "classification":
        return f"Highest mean CV {primary_metric} among successful candidate models."
    return "Best successful candidate model by the saved CV primary metric."


def _metric_definitions(task_type: str) -> str:
    if task_type == "regression":
        return (
            "Regression metrics: MAE is average absolute prediction error; "
            "RMSE penalizes larger mistakes; R2 estimates the share of target variation explained."
        )
    if task_type == "classification":
        return (
            "Classification metrics: macro F1 weights classes equally; weighted F1 follows "
            "class support; balanced accuracy averages recall across classes; per-class "
            "precision, recall, F1, and the confusion matrix show class-specific behavior. "
            "Binary classifiers also report ROC-AUC and average precision when probabilities "
            "are available."
        )
    return "Metric definitions were unavailable because the task type was not saved."
