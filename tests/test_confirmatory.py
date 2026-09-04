from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import pandas as pd

from app.llm import PROMPT_SCHEMA_VERSION
from evaluation.benchmarks import BenchmarkCase
from evaluation.confirmatory import (
    CONFIRMATORY_MANIFEST_SCHEMA_VERSION,
    manifest_sha256,
    runtime_manifest_values,
    validate_confirmatory_manifest,
)
from evaluation.runner import EXPERIMENT_CONFIG_VERSION
from evaluation.runner import run_evaluation


MANIFEST_PATH = Path(__file__).parents[1] / "evaluation" / "configs" / "paper_confirmatory_v1.json"


def _manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = "frozen"
    return manifest


def _runtime(manifest: dict, **overrides):
    modeling = manifest["modeling"]
    holdout = manifest["holdout"]
    prompts = manifest["prompts"]
    repetitions = manifest["splits_and_repetitions"]
    statistics = manifest["statistics"]
    external = manifest["external_benchmark"]
    values = {
        "experiment_name": manifest["experiment_name"],
        "planner_model": modeling["planner_model"],
        "reconciler_model": modeling["reconciler_model"],
        "split_seeds": repetitions["split_seeds"],
        "llm_repetitions": repetitions["llm_repetitions"],
        "holdout_fraction": holdout["fraction"],
        "selected_ablations": manifest["ablations"]["primary"],
        "deterministic_policy_version": manifest["deterministic_policy"]["version"],
        "empirical_probe_policy_version": manifest["empirical_probe_policy"]["policy_version"],
        "planner_prompt_schema_version": prompts["planner_schema_version"],
        "reconciler_prompt_schema_version": prompts["reconciliation_prompt_version"],
        "candidate_model_families": modeling["candidate_model_families"],
        "classification_neutral_tolerance": holdout["classification_neutral_tolerance"],
        "regression_neutral_tolerance": holdout["regression_neutral_tolerance"],
        "benchmark_manifest_version": external["manifest_version"],
        "strict_live_required": manifest["strict_live_required"],
        "bootstrap_settings": {
            "method": statistics["bootstrap_method"],
            "replicates": statistics["bootstrap_replicates"],
            "confidence_level": statistics["confidence_level"],
            "seed": statistics["bootstrap_seed"],
        },
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
    }
    values.update(overrides)
    return runtime_manifest_values(**values)


def test_frozen_manifest_accepts_matching_runtime_and_records_hash():
    manifest = _manifest()
    metadata = validate_confirmatory_manifest(manifest, _runtime(manifest))
    assert metadata["status"] == "frozen"
    assert metadata["experiment_config_sha256"] == manifest_sha256(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("planner_model", "different-planner", "planner_model"),
        ("llm_repetitions", 2, "llm_repetitions"),
        ("split_seeds", [999], "split_seeds"),
        ("selected_ablations", ["full"], "selected_ablations"),
        ("deterministic_policy_version", "different-policy", "deterministic_policy_version"),
        ("planner_prompt_schema_version", "different-prompt", "planner_prompt_schema_version"),
    ],
)
def test_frozen_manifest_rejects_runtime_mismatch(field, value, message):
    manifest = _manifest()
    with pytest.raises(ValueError, match=message):
        validate_confirmatory_manifest(manifest, _runtime(manifest, **{field: value}))


def test_manifest_hash_is_canonical_and_changes_when_a_value_changes():
    manifest = _manifest()
    reordered = json.loads(json.dumps(manifest, sort_keys=False))
    assert manifest_sha256(manifest) == manifest_sha256(reordered)

    changed = copy.deepcopy(manifest)
    changed["splits_and_repetitions"]["llm_repetitions"] = 99
    assert manifest_sha256(manifest) != manifest_sha256(changed)


def test_draft_manifest_is_not_accepted_as_confirmatory():
    manifest = _manifest()
    manifest["status"] = "draft"
    with pytest.raises(ValueError, match="not frozen"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


def test_manifest_schema_version_is_explicit():
    assert _manifest()["manifest_schema_version"] == CONFIRMATORY_MANIFEST_SCHEMA_VERSION


def test_ordinary_development_run_is_unblocked_and_emits_current_prompt_metadata(tmp_path: Path):
    frame = pd.DataFrame({
        "x": list(range(24)),
        "target": ["yes" if index % 2 else "no" for index in range(24)],
    })
    case = BenchmarkCase(
        name="confirmatory_metadata_fixture",
        dataframe=frame,
        target_column="target",
        question="Classify target from x.",
        expected_task_type="classification",
        dataset_source="in-memory test fixture",
    )
    result = run_evaluation(tmp_path / "development", cases=[case], offline=True, gate_mode="llm_only")
    config = json.loads((tmp_path / "development" / "config.json").read_text(encoding="utf-8"))
    assert result["summary"]["confirmatory_mode"] is False
    assert config["planner_prompt_schema_version"] == PROMPT_SCHEMA_VERSION
    assert config["reconciler_prompt_schema_version"]
    assert config["prompt_schema_version"] == PROMPT_SCHEMA_VERSION
    assert config["prompt_schema_version"] != "modeling-gate-v1"
