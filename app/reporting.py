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
    test_size: float = 0.2,
) -> str:
    return f'''"""Reproduce the approved AutoDS Agent analysis.

The semantic agent decisions are captured in decision.json. This script
replays and verifies that approved decision before running the deterministic
cleaning, EDA, and modeling stages.
"""

import json
from pathlib import Path
import pandas as pd
from app.schemas import PreprocessingContract
from app.deterministic import apply_cleaning, deterministic_recommendation, eda_summary, profile_dataframe
from app.modeling import fit_selected_model
from app.validation import freeze_supervised_split, prepare_validated_frame, validated_row_positions, validate_training_plan

DATASET = Path(r"{dataset_path}")
RUN_DIR = Path(__file__).resolve().parent
TARGET = {target_column!r}
QUESTION = {question!r}
METHOD = {method!r}
TASK_TYPE = {task_type!r}
TEST_SIZE = {test_size!r}

raw = pd.read_csv(DATASET)
decision = json.loads((RUN_DIR / "decision.json").read_text(encoding="utf-8"))
approved = decision["validation"]
APPROVED_PREPROCESSING = approved["approved_preprocessing"]
PreprocessingContract.model_validate(APPROVED_PREPROCESSING)
if approved["selected_target_column"] != TARGET:
    raise RuntimeError("Recorded target does not match the reproduction contract.")
if approved["selected_task_type"] != TASK_TYPE:
    raise RuntimeError("Recorded task does not match the reproduction contract.")
if approved["selected_method"] != METHOD:
    raise RuntimeError("Recorded method does not match the reproduction contract.")
if decision.get("gate_completed_before_training") is not True:
    raise RuntimeError("The original run did not complete its validation gate.")
if decision.get("holdout_policy", {{}}).get("frozen_before_modeling_recommendations") is not True:
    raise RuntimeError("The original run does not contain the frozen-holdout policy.")
recorded_split = decision.get("split_contract")
if not recorded_split:
    raise RuntimeError("The original run does not contain a canonical split contract.")
split = freeze_supervised_split(
    raw,
    TARGET,
    TASK_TYPE,
    test_size=TEST_SIZE,
    random_state={seed},
)
if split.as_dict() != recorded_split:
    raise RuntimeError(
        "The source dataset or split settings do not reproduce the exact recorded partition."
    )
# deterministic_recommendation is recorded for audit; reproduction never makes
# a new preprocessing decision and uses the recorded approved contract below.
profile = profile_dataframe(raw)
raw_validation = validate_training_plan(
    raw,
    TARGET,
    TASK_TYPE,
    METHOD,
    test_size=TEST_SIZE,
    random_state={seed},
    preprocessing=APPROVED_PREPROCESSING,
    split=split,
    row_positions=list(range(len(raw))),
)
raw_validation.raise_if_failed()
cleaning = json.loads((RUN_DIR / "cleaning.json").read_text(encoding="utf-8"))
ROW_POSITION_COLUMN = "__autods_row_position__"
if ROW_POSITION_COLUMN in raw.columns:
    raise RuntimeError("The reserved row-position column is present in the source dataset.")
raw_with_positions = raw.copy()
raw_with_positions[ROW_POSITION_COLUMN] = range(len(raw_with_positions))
cleaned, cleaning_log = apply_cleaning(
    raw_with_positions,
    target_column=TARGET,
    actions=cleaning["plan"]["actions"],
    row_position_column=ROW_POSITION_COLUMN,
)
cleaned_row_positions = cleaned.pop(ROW_POSITION_COLUMN).to_numpy(dtype=int)
cleaned_validation = validate_training_plan(
    cleaned,
    TARGET,
    TASK_TYPE,
    METHOD,
    test_size=TEST_SIZE,
    random_state={seed},
    preprocessing=APPROVED_PREPROCESSING,
    split=split,
    row_positions=cleaned_row_positions,
)
cleaned_validation.raise_if_failed()
cleaned_row_positions = validated_row_positions(
    cleaned,
    cleaned_validation,
    cleaned_row_positions,
)
cleaned = prepare_validated_frame(cleaned, cleaned_validation)
print(eda_summary(cleaned, TARGET))
result = fit_selected_model(
    cleaned,
    target_column=TARGET,
    task_type=TASK_TYPE,
    method=METHOD,
    preprocessing=APPROVED_PREPROCESSING,
    output_dir=RUN_DIR / "reproduced_model",
    test_size=TEST_SIZE,
    random_state={seed},
    split=split,
    row_positions=cleaned_row_positions,
)
if result["approved_preprocessing"] != APPROVED_PREPROCESSING:
    raise RuntimeError("The reproduced executable preprocessing does not match the recorded contract.")
recorded_modeling = json.loads((RUN_DIR / "modeling.json").read_text(encoding="utf-8"))
if result["excluded_features"] != recorded_modeling["excluded_features"]:
    raise RuntimeError("The reproduced feature exclusions do not match the recorded plan.")
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
    deterministic_validation = validation.get("deterministic_validation") or {}
    validation_checks = deterministic_validation.get("checks", [])
    passed_checks = sum(1 for check in validation_checks if check.get("passed"))
    excluded_features = deterministic_validation.get("excluded_features", [])
    preprocessing_comparison = validation.get("preprocessing_comparison", {})
    approved_preprocessing = validation.get("approved_preprocessing", {})
    agent_preprocessing = validation.get("agent_preprocessing", {})
    deterministic_preprocessing = validation.get("deterministic_preprocessing", {})
    material_differences = preprocessing_comparison.get("material_differences", [])
    reconciliation = validation.get("reconciliation")
    diagnostics = deterministic.diagnostics.model_dump(mode="json") if deterministic.diagnostics else {}
    score_lines = "\n".join(
        f"- <code>{method}</code>: <code>{score if score is not None else 'ineligible'}</code>"
        for method, score in deterministic.method_scores.items()
    )
    selected_assessment = deterministic.method_assessments.get(deterministic.recommended_method)
    contribution_lines = "\n".join(
        f"- <code>{item.factor}</code> ({item.points:+d}): {item.observation}"
        for item in (selected_assessment.contributions if selected_assessment else [])
        if item.points != 0
    ) or "- no non-zero contribution recorded"
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

### Deterministic compatibility evidence

Policy version: <code>{deterministic.policy_version}</code>; confidence: <code>{deterministic.confidence}</code>; score margin: <code>{deterministic.score_margin if deterministic.score_margin is not None else 'unavailable'}</code>.

Method-family compatibility scores (bounded policy points, not probabilities):

{score_lines}

Training-only diagnostics: <code>{json.dumps(diagnostics, sort_keys=True)}</code>.

Selected-family score contributions:

{contribution_lines}

Validation decision: {validation.get('justification', 'The recommendations matched on target, task, and method.')}

Holdout boundary: target/task establishment completed before the supervised split; the frozen holdout was reserved for final evaluation. Modeling-agent evidence, deterministic recommendation evidence, preprocessing requirements, and any reconciliation used the training partition only.

Deterministic contract: <code>{deterministic_validation.get('status', 'not recorded')}</code> ({passed_checks}/{len(validation_checks)} checks passed); target rows removed: <code>{deterministic_validation.get('target_rows_removed', 0)}</code>; direct leakage detected: <code>{deterministic_validation.get('direct_leakage_detected', False)}</code>.

Excluded features and reasons: <code>{', '.join(f"{item.get('column')} ({item.get('reason_code')})" for item in excluded_features) if excluded_features else 'none'}</code>.

### Preprocessing contract

| Source | Contract |
| --- | --- |
| Independent agent | <code>{json.dumps(agent_preprocessing, sort_keys=True)}</code> |
| Deterministic recommender | <code>{json.dumps(deterministic_preprocessing, sort_keys=True)}</code> |
| Final approved executable plan | <code>{json.dumps(approved_preprocessing, sort_keys=True)}</code> |

Preprocessing comparison: <code>{preprocessing_comparison.get('status', 'not recorded')}</code>; material differences: <code>{json.dumps(material_differences, sort_keys=True) if material_differences else 'none'}</code>.

Reconciliation output: <code>{json.dumps(reconciliation, sort_keys=True) if reconciliation else 'not invoked'}</code>.

The deterministic requirements and checks are persisted with the gate. All learned transformations are fitted inside the training pipeline; the holdout is used only for final evaluation. The executed pipeline components are <code>{json.dumps(modeling.get('executed_preprocessing', {}).get('pipeline_components', {}), sort_keys=True)}</code>.

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

<code>reproduce_analysis.py</code> replays the approved deterministic stages and verifies the recorded gate decision. The agent decisions themselves are preserved in <code>decision.json</code> rather than regenerated during replay.
"""
