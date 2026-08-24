"""Reproduce the approved AutoDS Agent analysis.

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

DATASET = Path(r"C:\Users\19492\Desktop\AutoDSAgent\.pytest_tmp_interaction_final\test_end_to_end_holdout_pertur0\perturbed.csv")
RUN_DIR = Path(__file__).resolve().parent
TARGET = 'target'
QUESTION = 'Classify target from the measured features.'
METHOD = 'tree_ensemble'
TASK_TYPE = 'classification'
TEST_SIZE = 0.2

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
if decision.get("holdout_policy", {}).get("frozen_before_modeling_recommendations") is not True:
    raise RuntimeError("The original run does not contain the frozen-holdout policy.")
recorded_split = decision.get("split_contract")
if not recorded_split:
    raise RuntimeError("The original run does not contain a canonical split contract.")
split = freeze_supervised_split(
    raw,
    TARGET,
    TASK_TYPE,
    test_size=TEST_SIZE,
    random_state=42,
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
    random_state=42,
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
partition_positions = {
    "training": list(split.train_row_positions),
    "holdout": list(split.holdout_row_positions),
    "unassigned": [
        int(position) for position in range(len(raw_with_positions)) if int(position) not in valid_positions
    ],
}
transformed_partitions = []
transformed_by_partition = {}
partition_logs = {}
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
    random_state=42,
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
    random_state=42,
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
