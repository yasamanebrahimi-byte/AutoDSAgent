"""Report and reproducibility artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.schemas import (
    AgentPlan,
    CleaningPlan,
    DeterministicRecommendation,
    ReportDraft,
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def render_code(
    dataset_path: str,
    target_column: str,
    question: str,
    method: str,
    task_type: str,
    seed: int,
) -> str:
    return f'''"""Reproduce the approved AutoDS Agent analysis."""

from pathlib import Path
import pandas as pd
from app.deterministic import apply_cleaning, eda_summary
from app.modeling import fit_selected_model

DATASET = Path(r"{dataset_path}")
TARGET = {target_column!r}
QUESTION = {question!r}
METHOD = {method!r}
TASK_TYPE = {task_type!r}

raw = pd.read_csv(DATASET)
cleaned, cleaning_log = apply_cleaning(
    raw,
    target_column=TARGET,
    actions=[
        "trim_strings",
        "drop_exact_duplicates",
        "drop_all_null_columns",
        "drop_constant_features",
        "drop_rows_missing_target",
    ],
)
print(eda_summary(cleaned, TARGET))
result = fit_selected_model(
    cleaned,
    target_column=TARGET,
    task_type=TASK_TYPE,
    method=METHOD,
    output_dir=Path("reproduced_run") / "model",
    random_state={seed},
)
print(result)
'''


def render_report(
    question: str,
    profile: dict[str, Any],
    agent_plan: AgentPlan,
    deterministic: DeterministicRecommendation,
    validation: dict[str, Any],
    cleaning_plan: CleaningPlan,
    cleaning_log: dict[str, Any],
    eda: dict[str, Any],
    findings: list[str],
    modeling: dict[str, Any],
    draft: ReportDraft,
    artifact_names: list[str],
) -> str:
    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {item}" for item in items)

    agreement = (
        "Agreement"
        if validation["status"] == "agreement"
        else "Disagreement investigated"
    )
    chosen = validation["selected_method"]
    cv_lines = "\n".join(
        f"- <code>{key}</code>: <code>{value:.4f}</code>"
        for key, value in modeling["cv_metrics"].items()
    )
    holdout_lines = "\n".join(
        f"- <code>{key}</code>: <code>{value:.4f}</code>"
        for key, value in modeling["holdout_metrics"].items()
    )
    finding_lines = bullets(findings or draft.key_findings)
    return f"""# AutoDS Agent Analysis Report

## Question

{question}

## Validation gate: {agreement}

The workflow intentionally made a modeling decision before fitting any model.

| Source | Target | Task | Method |
| --- | --- | --- | --- |
| Independent agent | <code>{agent_plan.target_column}</code> | <code>{agent_plan.task_type}</code> | <code>{agent_plan.recommended_method}</code> |
| Deterministic recommender | <code>{deterministic.target_column}</code> | <code>{deterministic.task_type}</code> | <code>{deterministic.recommended_method}</code> |
| Final approved plan | <code>{validation['selected_target_column']}</code> | <code>{validation['selected_task_type']}</code> | <code>{chosen}</code> |

Deterministic reasoning: {deterministic.reasoning}

Validation decision: {validation.get('justification', 'The recommendations matched on target, task, and method.')}

## Data profile

- Rows before cleaning: <code>{profile['rows']}</code>
- Columns before cleaning: <code>{profile['columns']}</code>
- Duplicate rows detected: <code>{profile['duplicate_rows']}</code>
- Rows after cleaning: <code>{cleaning_log['cleaned_shape'][0]}</code>
- Columns after cleaning: <code>{cleaning_log['cleaned_shape'][1]}</code>

## Cleaning agent

The cleaning agent requested: <code>{', '.join(cleaning_plan.actions) if cleaning_plan.actions else 'no actions'}</code>.

Applied structural actions: <code>{', '.join(cleaning_log['applied_actions']) if cleaning_log['applied_actions'] else 'none'}</code>.

{cleaning_plan.reasoning}

## Exploratory data analysis

{finding_lines}

The numeric relationships and target distribution are computed deterministically; the EDA agent only interprets those values.

## Modeling

The approved model was <strong>{modeling['selected_model']}</strong> using training-only preprocessing and <code>{modeling['cv_folds']}</code>-fold <code>{modeling['cv_strategy']}</code> validation.

### Cross-validation

{cv_lines}

### Untouched holdout

{holdout_lines}

### Baseline holdout

{', '.join(f'<code>{key}={value:.4f}</code>' for key, value in modeling['baseline_metrics'].items())}

{draft.modeling_interpretation}

## Limitations and next steps

{bullets(draft.limitations)}

Recommended next steps:

{bullets(draft.next_steps)}

## Artifacts

{bullets([f'<code>{name}</code>' for name in artifact_names])}

The full reproducible Python script is in <code>reproduce_analysis.py</code>.
"""

