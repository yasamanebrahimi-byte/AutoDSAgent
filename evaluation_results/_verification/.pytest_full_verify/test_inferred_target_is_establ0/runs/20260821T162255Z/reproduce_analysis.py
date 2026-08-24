"""Reproduce the approved AutoDS Agent analysis.

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

DATASET = Path(r"C:\Users\19492\Desktop\AutoDSAgent\.pytest_full_verify\test_inferred_target_is_establ0\inferred.csv")
RUN_DIR = Path(__file__).resolve().parent
TARGET = 'target'
QUESTION = 'Please classify target using the available measurements.'
METHOD = 'tree_ensemble'
TASK_TYPE = 'classification'
TEST_SIZE = 0.2

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
profile = profile_dataframe(raw)
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
    random_state=42,
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
    random_state=42,
    split=split,
    row_positions=cleaned_row_positions,
)
if result["approved_preprocessing"] != APPROVED_PREPROCESSING:
    raise RuntimeError("The reproduced executable preprocessing does not match the recorded contract.")
recorded_modeling = json.loads((RUN_DIR / "modeling.json").read_text(encoding="utf-8"))
if result["excluded_features"] != recorded_modeling["excluded_features"]:
    raise RuntimeError("The reproduced feature exclusions do not match the recorded plan.")
print(result)
