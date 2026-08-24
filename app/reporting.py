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
from app.deterministic import eda_summary, transform_cleaning
from app.modeling import fit_selected_model
from app.validation import (
    freeze_supervised_split,
    prepare_validated_frame,
    training_partition_frame,
    validated_row_positions,
    validate_training_plan,
)

DATASET = Path(r"{dataset_path}")
RUN_DIR = Path(__file__).resolve().parent
TARGET = {target_column!r}
QUESTION = {question!r}
METHOD = {method!r}
TASK_TYPE = {task_type!r}
TEST_SIZE = {test_size!r}

raw = pd.read_csv(DATASET, dtype=object)
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
if decision.get("formulation_gate_status") != "completed":
    raise RuntimeError("The original run did not complete the formulation gate.")
if decision.get("split_frozen_after_formulation_gate") is not True:
    raise RuntimeError("The original run does not prove that the split followed formulation approval.")
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
    evidence_dataframe=raw.iloc[list(split.train_row_positions)].copy(),
)
raw_validation.raise_if_failed()
cleaning = json.loads((RUN_DIR / "cleaning.json").read_text(encoding="utf-8"))
if cleaning.get("decision_scope") != "training_partition_only":
    raise RuntimeError("The cleaning artifact does not record training-only decision scope.")
if cleaning.get("holdout_used_for_cleaning_decisions") is not False:
    raise RuntimeError("The cleaning artifact does not prove holdout isolation.")
recorded_specification = cleaning.get("fitted_specification")
if not recorded_specification:
    raise RuntimeError("The cleaning artifact does not contain a fitted specification.")
from app.schemas import CleaningSpecification
specification = CleaningSpecification.model_validate(recorded_specification)
ROW_POSITION_COLUMN = "__autods_row_position__"
if ROW_POSITION_COLUMN in raw.columns:
    raise RuntimeError("The reserved row-position column is present in the source dataset.")
raw_with_positions = raw.copy()
raw_with_positions[ROW_POSITION_COLUMN] = range(len(raw_with_positions))
valid_positions = set(split.valid_row_positions)
partition_positions = {{
    "training": list(split.train_row_positions),
    "holdout": list(split.holdout_row_positions),
    "unassigned": [
        int(position) for position in range(len(raw_with_positions)) if int(position) not in valid_positions
    ],
}}
transformed_partitions = []
transformed_by_partition = {{}}
partition_logs = {{}}
for partition_name, positions in partition_positions.items():
    transformed, partition_log = transform_cleaning(
        raw_with_positions.iloc[positions].copy(),
        specification,
        partition=partition_name,
    )
    transformed_partitions.append(transformed)
    transformed_by_partition[partition_name] = transformed
    partition_logs[partition_name] = partition_log
recorded_transformations = cleaning.get("applied_transformations")
if recorded_transformations != partition_logs:
    raise RuntimeError("The recorded cleaning specification cannot reproduce the partition transforms.")
cleaned = (
    pd.concat(transformed_partitions, axis=0, ignore_index=True)
    .sort_values(ROW_POSITION_COLUMN, kind="stable")
    .reset_index(drop=True)
)
cleaned_row_positions = cleaned.pop(ROW_POSITION_COLUMN).to_numpy(dtype=int)
training_evidence_frame = transformed_by_partition["training"].drop(columns=[ROW_POSITION_COLUMN])
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
    evidence_dataframe=training_evidence_frame,
)
cleaned_validation.raise_if_failed()
cleaned_row_positions = validated_row_positions(
    cleaned,
    cleaned_validation,
    cleaned_row_positions,
)
cleaned = prepare_validated_frame(cleaned, cleaned_validation)
eda_frame = training_partition_frame(cleaned, split, cleaned_row_positions)
print(eda_summary(eda_frame, TARGET))
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
    evidence_dataframe=training_evidence_frame,
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

    formulation = validation.get("formulation", {})
    formulation_agent = formulation.get("agent_initial") or {}
    formulation_deterministic = formulation.get("deterministic") or {}
    formulation_comparison = formulation.get("comparison") or {}
    formulation_final = formulation.get("final") or {}
    formulation_reconciliation = formulation.get("reconciliation")
    formulation_status = formulation.get("status", "not recorded")
    agreement = "Agreement" if validation["status"] == "agreement" else "Disagreement investigated"
    chosen = validation["selected_method"]
    deterministic_validation = validation.get("deterministic_validation") or {}
    hard_validation = validation.get("hard_validation") or {}
    soft_challenge = validation.get("soft_challenge") or {}
    empirical_probe = validation.get("empirical_probe") or {}
    final_decision = validation.get("final") or {}
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

## Problem formulation gate: {formulation_status}

The workflow first validated what supervised problem to solve, before creating any train/holdout split.

### User question

{question}

### Agent formulation

- Target: <code>{formulation_agent.get('target_column', 'not recorded')}</code>
- Task: <code>{formulation_agent.get('task_type', 'not recorded')}</code>
- Reasoning: {formulation_agent.get('reasoning', 'not recorded')}
- Confidence: <code>{formulation_agent.get('confidence', 'not recorded')}</code>

### Deterministic formulation

- Target: <code>{formulation_deterministic.get('target_column', 'not recorded')}</code>
- Task: <code>{formulation_deterministic.get('task_type', 'not recorded')}</code>
- Reasoning: {formulation_deterministic.get('reasoning', 'not recorded')}
- Evidence: <code>{json.dumps(formulation_deterministic.get('evidence', []), sort_keys=True)}</code>

### Formulation result

- Target agreement: <code>{formulation_comparison.get('target_agreement', 'not recorded')}</code>
- Task agreement: <code>{formulation_comparison.get('task_agreement', 'not recorded')}</code>
- Result: <strong>{'Disagreement investigated' if formulation_status != 'agreement' else 'Agreement'}</strong>
- Reconciliation: <code>{json.dumps(formulation_reconciliation, sort_keys=True) if formulation_reconciliation else 'not invoked'}</code>
- Approved target: <code>{formulation_final.get('target_column', 'not recorded')}</code>
- Approved task: <code>{formulation_final.get('task_type', 'not recorded')}</code>
- Target source: <code>{formulation_final.get('target_source', 'not recorded')}</code>; mutable: <code>{formulation_final.get('target_is_mutable', 'not recorded')}</code>
- Justification: {formulation_final.get('justification', 'not recorded')}

## Modeling decision gate: {agreement}

The workflow intentionally made a modeling decision before fitting any model.

Hard deterministic validation is authoritative for safety and executability. The deterministic model-family recommendation is an advisory soft challenger; disagreement alone is not an invalid plan.

| Source | Approved formulation | Method |
| --- | --- | --- |
| Modeling agent | <code>{validation.get('selected_target_column')}</code> / <code>{validation.get('selected_task_type')}</code> | <code>{agent_plan.recommended_method}</code> |
| Deterministic recommender | <code>{deterministic.target_column}</code> / <code>{deterministic.task_type}</code> | <code>{deterministic.recommended_method}</code> |
| Final approved plan | <code>{validation['selected_target_column']}</code> / <code>{validation['selected_task_type']}</code> | <code>{chosen}</code> |

Deterministic reasoning: {deterministic.reasoning}

### Deterministic compatibility evidence

Policy version: <code>{deterministic.policy_version}</code>; confidence: <code>{deterministic.confidence}</code>; score margin: <code>{deterministic.score_margin if deterministic.score_margin is not None else 'unavailable'}</code>.

Method-family compatibility scores (bounded policy points, not probabilities):

{score_lines}

Training-only diagnostics: <code>{json.dumps(diagnostics, sort_keys=True)}</code>.

{"Classification diagnostics use label-order-invariant eta-squared/Cramér's V class-association measures and a separate numeric-only boundary probe; the probe uses fold-local imputation/scaling, balanced-accuracy logistic diagnostics, and macro local-neighbor consistency without treating nominal class labels as ordered numeric values." if deterministic.task_type == "classification" else "Regression diagnostics use numeric-target correlation, rank, and binned-target evidence."}

The structural-complexity score is a bounded training-only compatibility heuristic; it summarizes observable feature structure and does not prove the presence of statistical feature interactions.

Selected-family score contributions:

{contribution_lines}

Hard validation: <code>{hard_validation.get('status', 'not recorded')}</code>; intervention required: <code>{hard_validation.get('intervention_required', False)}</code>; initial proposal hard-invalid: <code>{hard_validation.get('initial_hard_invalid', False)}</code>.

Soft challenge: <code>{soft_challenge.get('status', 'not recorded')}</code>; decision: <code>{soft_challenge.get('decision', 'not recorded')}</code>; decision reason: <code>{soft_challenge.get('decision_reason', 'not recorded')}</code>; method disagreement: <code>{soft_challenge.get('method_disagreement', False)}</code>; preprocessing disagreement: <code>{soft_challenge.get('preprocessing_disagreement', False)}</code>; deterministic confidence: <code>{soft_challenge.get('deterministic_confidence', deterministic.confidence)}</code>; score margin: <code>{soft_challenge.get('score_margin', deterministic.score_margin)}</code>; empirical reliability: <code>{soft_challenge.get('empirical_reliability', 'insufficient_evidence')}</code>; calibration support: <code>{soft_challenge.get('calibration_support', 0)}</code>; reconciliation invoked: <code>{soft_challenge.get('reconciliation_invoked', False)}</code>.

### Limited training-only empirical comparison

Probe invoked: <code>{validation.get('empirical_probe_invoked', False)}</code>; status: <code>{empirical_probe.get('status', 'not invoked')}</code>; policy: <code>{empirical_probe.get('policy_version', 'not invoked')}</code>; metric: <code>{empirical_probe.get('metric', 'not invoked')}</code>; folds: <code>{empirical_probe.get('cv_folds', 'not invoked')}</code>; winner: <code>{empirical_probe.get('winner', 'not invoked')}</code>; evidence strength: <code>{empirical_probe.get('evidence_strength', 'not invoked')}</code>.

The probe, when present, compares only Proposal A and Proposal B using frozen training-partition rows and fold-local preprocessing. Its result is directional evidence—not final holdout validation or an automatic model selection—and remains subject to fold variability and reconciliation judgment. Details: <code>{json.dumps(empirical_probe, sort_keys=True) if empirical_probe else 'not invoked'}</code>.

Final selection source: <code>{final_decision.get('selected_source', 'not recorded')}</code>.

Reconciliation framing: <code>{validation.get('reconciliation_mode', 'not invoked')}</code>; proposal A originated from <code>{validation.get('proposal_a_source', 'not recorded')}</code>; proposal B originated from <code>{validation.get('proposal_b_source', 'not recorded')}</code>; selected proposal: <code>{validation.get('selected_proposal', 'not recorded')}</code>. Two independently generated modeling proposals were compared using shared training-only evidence; source identity was retained for audit and revealed only after selection.

Reconciliation decision: {validation.get('justification', 'The proposals matched on target, task, and method.')}

Holdout boundary: the formulation gate completed and was validated before the supervised split. Modeling-agent evidence, deterministic recommendation evidence, modeling reconciliation, preprocessing requirements, structural-cleaning decisions, pre-evaluation EDA and plots, and cross-validation used training-partition evidence only. The EDA artifact contains <code>{eda.get('rows', 0)}</code> cleaned training rows and no holdout rows. The fail-closed validation gate may inspect the full raw or cleaned frame only to enforce target, schema, feasibility, and frozen-membership invariants; those guardrail checks are not planning evidence. The frozen holdout was scored once for final model evaluation.

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

The structural-cleaning specification was fitted from <code>{cleaning_log.get('decision_scope', 'training_partition_only')}</code> evidence and then transformed independently on each partition. Holdout values were not used to derive column removals, coercion eligibility, thresholds, or training duplicate membership. Exact-duplicate removal uses the policy <code>{cleaning_log.get('duplicate_policy', 'within_partition_only_keep_first')}</code>.

{cleaning_plan.reasoning}

## Exploratory data analysis

{finding_lines}

The pre-evaluation numeric relationships, target distribution, missingness, and plots were computed deterministically from the frozen training partition only; the EDA agent received only those training-only summaries and interpreted those values without inspecting holdout data.

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
