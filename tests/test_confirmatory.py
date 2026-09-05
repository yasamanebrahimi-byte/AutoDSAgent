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
    config_sha256,
    deterministic_policy_config,
    empirical_probe_config,
    repository_commit,
    experiment_code_sha256,
)
from evaluation.external_benchmarks import external_benchmark_manifest_sha256, external_benchmark_specs
from evaluation.runner import EXPERIMENT_CONFIG_VERSION
from evaluation.runner import run_evaluation
from evaluation.provenance import environment_provenance


MANIFEST_PATH = Path(__file__).parents[1] / "evaluation" / "configs" / "paper_confirmatory_v1.json"


def _manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["status"] = "frozen"
    manifest["expected_experiment_code_sha256"] = experiment_code_sha256()
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
        "deterministic_policy_sha256": config_sha256(deterministic_policy_config()),
        "empirical_probe_policy_version": manifest["empirical_probe_policy"]["policy_version"],
        "empirical_probe_policy_sha256": config_sha256(empirical_probe_config()),
        "planner_prompt_schema_version": prompts["planner_schema_version"],
        "reconciler_prompt_schema_version": prompts["reconciliation_prompt_version"],
        "candidate_model_families": modeling["candidate_model_families"],
        "preprocessing_option_space": modeling["preprocessing_option_space"],
        "classification_neutral_tolerance": holdout["classification_neutral_tolerance"],
        "regression_neutral_tolerance": holdout["regression_neutral_tolerance"],
        "benchmark_manifest_version": external["manifest_version"],
        "benchmark_manifest_sha256": external_benchmark_manifest_sha256(),
        "benchmark_task_ids": [spec.task_id for spec in external_benchmark_specs()],
        "benchmark_tranches": {
            "core": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "core"],
            "stress": [spec.task_id for spec in external_benchmark_specs() if spec.tier == "stress"],
        },
        "benchmark_tier": None,
        "strict_live_required": manifest["strict_live_required"],
        "bootstrap_settings": {
            "method": statistics["bootstrap_method"],
            "replicates": statistics["bootstrap_replicates"],
            "confidence_level": statistics["confidence_level"],
            "seed": statistics["bootstrap_seed"],
        },
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "expected_experiment_code_sha256": experiment_code_sha256(),
    }
    values.update(overrides)
    return runtime_manifest_values(**values)


def test_frozen_manifest_accepts_matching_runtime_and_records_hash():
    manifest = _manifest()
    metadata = validate_confirmatory_manifest(manifest, _runtime(manifest))
    assert metadata["status"] == "frozen"
    assert metadata["experiment_config_sha256"] == manifest_sha256(manifest)
    assert metadata["expected_experiment_code_sha256"] == experiment_code_sha256()


def test_runtime_generation_settings_omit_provider_defaults_for_frozen_manifest():
    manifest = _manifest()
    runtime = _runtime(
        manifest,
        generation_settings={
            "temperature": None,
            "top_p": None,
            "seed": None,
            "reasoning_effort": "medium",
        },
    )

    assert runtime["generation_settings"] == {"reasoning_effort": "medium"}
    assert validate_confirmatory_manifest(manifest, runtime)["status"] == "frozen"


def test_frozen_manifest_rejects_wrong_experiment_code_hash():
    manifest = _manifest()
    manifest["expected_experiment_code_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected.*SHA-256|SHA-256 mismatch"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


def test_frozen_manifest_rejects_missing_experiment_code_hash():
    manifest = _manifest()
    manifest["expected_experiment_code_sha256"] = None
    with pytest.raises(ValueError, match="expected_experiment_code_sha256"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


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


def test_frozen_manifest_rejects_declared_but_wrong_selected_condition_model():
    manifest = _manifest()
    with pytest.raises(ValueError, match="selected condition"):
        validate_confirmatory_manifest(
            manifest,
            _runtime(
                manifest,
                model_conditions=manifest["model_conditions"],
                selected_model_condition_id="gpt5_mini_2025_08_07",
                planner_model="gpt-5.4-mini-2026-03-17",
                reconciler_model="gpt-5.4-mini-2026-03-17",
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_experiment_code_sha256", "0" * 64, "SHA-256"),
        ("deterministic_policy_sha256", "changed", "deterministic_policy_sha256"),
        ("empirical_probe_policy_sha256", "changed", "empirical_probe_policy_sha256"),
        ("benchmark_task_ids", [359983], "benchmark membership"),
        ("benchmark_tier", "core", "benchmark membership"),
        ("preprocessing_option_space", ["changed"], "preprocessing_option_space"),
    ],
)
def test_confirmatory_freeze_rejects_exact_identity_changes(field, value, message):
    manifest = _manifest()
    with pytest.raises(ValueError, match=message):
        validate_confirmatory_manifest(manifest, _runtime(manifest, **{field: value}))


def test_changed_policy_parameters_change_the_frozen_hash():
    manifest = _manifest()
    deterministic = deterministic_policy_config()
    deterministic["high_correlation_threshold"] = 0.81
    with pytest.raises(ValueError, match="deterministic_policy_sha256"):
        validate_confirmatory_manifest(
            manifest,
            _runtime(manifest, deterministic_policy_sha256=config_sha256(deterministic)),
        )
    empirical = empirical_probe_config()
    empirical["strong_relative_threshold"] = 0.21
    with pytest.raises(ValueError, match="empirical_probe_policy_sha256"):
        validate_confirmatory_manifest(
            manifest,
            _runtime(manifest, empirical_probe_policy_sha256=config_sha256(empirical)),
        )


def test_confirmatory_manifest_rejects_removed_empirical_probe_parameter():
    manifest = _manifest()
    manifest["empirical_probe_policy"]["weak_relative_threshold"] = 0.05
    with pytest.raises(ValueError, match="unknown/deprecated.*weak_relative_threshold"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


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
    manifest["expected_experiment_code_sha256"] = None
    with pytest.raises(ValueError, match="not frozen"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


def test_confirmatory_run_copies_exact_frozen_manifest_and_records_metadata(tmp_path: Path):
    manifest_path = tmp_path / "frozen.json"
    manifest_bytes = MANIFEST_PATH.read_bytes().replace(
        b'"status": "draft"', b'"status": "frozen"'
    ).replace(
        b'"expected_experiment_code_sha256": null',
        (b'"expected_experiment_code_sha256": ' + json.dumps(experiment_code_sha256()).encode("ascii")),
    )
    manifest_path.write_bytes(manifest_bytes)
    # The fixture is intentionally written from the checked-in bytes so the
    # result assertion covers byte-for-byte preservation, not just semantics.
    result_dir = tmp_path / "result"
    frame = pd.DataFrame({"x": list(range(24)), "target": ["yes" if i % 2 else "no" for i in range(24)]})
    case = BenchmarkCase(
        name="manifest_copy_fixture",
        dataframe=frame,
        target_column="target",
        question="Classify target from x.",
        expected_task_type="classification",
        dataset_source="in-memory test fixture",
        openml_task_id=359983,
        benchmark_suite_version="1.0.0",
    )
    metadata = {
        "status": "frozen",
        "experiment_config_version": EXPERIMENT_CONFIG_VERSION,
        "experiment_config_sha256": manifest_sha256(manifest_path),
        "benchmark_manifest_matches": True,
        "expected_experiment_code_sha256": experiment_code_sha256(),
        "source_git_commit": repository_commit(),
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("evaluation.runner.validate_confirmatory_manifest", lambda *_args: metadata)
    try:
        run_evaluation(
            result_dir,
            cases=[case],
            offline=True,
            suite="external",
            confirmatory_config_path=manifest_path,
        )
    finally:
        monkeypatch.undo()
    copied = result_dir / "frozen_confirmatory_manifest.json"
    assert copied.read_bytes() == manifest_bytes
    config = json.loads((result_dir / "config.json").read_text(encoding="utf-8"))
    assert config["experiment_config_sha256"] == manifest_sha256(manifest_path)
    assert config["frozen_manifest_path"] == str(copied)


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


def test_experiment_hash_manifest_freeze_is_self_reference_free(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "evaluation" / "configs").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname='fixture'\n")
    (tmp_path / "app" / "pipeline.py").write_bytes(b"PIPELINE = 1\n")
    manifest_path = tmp_path / "evaluation" / "configs" / "paper_confirmatory_v1.json"
    manifest_path.write_text('{"status":"draft","expected_experiment_code_sha256":null}', encoding="utf-8")
    before = experiment_code_sha256(tmp_path)
    manifest_path.write_text(
        json.dumps({"status": "frozen", "expected_experiment_code_sha256": before}, indent=2),
        encoding="utf-8",
    )
    assert experiment_code_sha256(tmp_path) == before


def test_experiment_hash_detects_included_source_and_ignores_generated_files(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "evaluation" / "evaluation_results").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname='fixture'\n")
    source = tmp_path / "app" / "policy.py"
    source.write_bytes(b"POLICY = 1\n")
    baseline = experiment_code_sha256(tmp_path)
    (tmp_path / "evaluation" / "evaluation_results" / "result.json").write_bytes(b"{}")
    assert experiment_code_sha256(tmp_path) == baseline
    source.write_bytes(b"POLICY = 2\n")
    assert experiment_code_sha256(tmp_path) != baseline


def test_validation_rejects_changed_experiment_code_with_clear_hash_error(monkeypatch):
    manifest = _manifest()
    monkeypatch.setattr("evaluation.confirmatory.experiment_code_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="expected.*observed.*result-affecting"):
        validate_confirmatory_manifest(manifest, _runtime(manifest))


def test_experiment_hash_is_deterministic_independent_of_traversal_order(tmp_path: Path):
    (tmp_path / "app").mkdir()
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\nname='fixture'\n")
    (tmp_path / "app" / "z.py").write_bytes(b"z\n")
    (tmp_path / "app" / "a.py").write_bytes(b"a\n")
    assert experiment_code_sha256(tmp_path) == experiment_code_sha256(tmp_path)


def test_environment_provenance_is_structured_deterministic_and_secret_free(tmp_path: Path):
    manifest = {"status": "frozen", "experiment": "fixture"}
    first = environment_provenance(repository_root=tmp_path, manifest=manifest)
    second = environment_provenance(repository_root=tmp_path, manifest=manifest)
    assert first == second
    assert first["python_version"]
    assert first["platform"]
    assert isinstance(first["packages"], dict)
    for package in ("numpy", "pandas", "scikit-learn", "openai", "openml"):
        assert package in first["packages"]
    assert first["experiment_code_sha256"]
    assert first["confirmatory_manifest_sha256"]
    assert "OPENAI_API_KEY" not in json.dumps(first)
    assert "api_key" not in json.dumps(first).lower()
