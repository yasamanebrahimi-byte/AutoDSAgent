from __future__ import annotations

import sys
import shutil
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

import evaluation.external_benchmarks as external
import evaluation.run as evaluation_cli
from evaluation.benchmarks import BenchmarkRole, default_benchmark_cases


CLASSIFICATION_IDS = {
    359983,
    359979,
    168868,
    146818,
    359982,
    359967,
    359955,
    359960,
    359968,
    359992,
    168757,
    359964,
    359954,
    359970,
    359966,
    359962,
    190137,
    359971,
    168350,
    359956,
    168784,
    359974,
}
REGRESSION_IDS = {
    359944,
    359938,
    359942,
    233211,
    359936,
    359952,
    359951,
    359949,
    233215,
    360945,
    167210,
    359941,
    359930,
    359931,
    359932,
    359933,
    359934,
    359935,
}


def _classification_spec(**changes):
    values = {
        "task_id": 123,
        "name": "fixture",
        "expected_task_type": "classification",
        "expected_rows": 4,
        "expected_features": 2,
        "expected_classes": 2,
    }
    values.update(changes)
    return external.OpenMLBenchmarkSpec(**values)


def _fake_openml(X, y, *, target_name="label", dataset_id=987):
    calls = {}

    class Dataset:
        name = "fake-openml-dataset"
        default_target_attribute = target_name

        def get_data(self, *, target, dataset_format):
            calls["get_data"] = (target, dataset_format)
            return X, y, [False] * X.shape[1], list(X.columns)

    class Task:
        def __init__(self):
            self.dataset_id = dataset_id
            self.target_name = target_name

        def get_dataset(self):
            calls["get_dataset"] = True
            return Dataset()

    class Tasks:
        def get_task(self, task_id, *, download_splits):
            calls["get_task"] = (task_id, download_splits)
            return Task()

    module = SimpleNamespace(
        tasks=Tasks(),
        config=SimpleNamespace(set_root_cache_directory=lambda path: calls.setdefault("cache", path)),
    )
    return module, calls


def test_frozen_manifest_has_exact_task_ids_and_distribution():
    specs = external.external_benchmark_specs()
    assert len(specs) == 40
    assert len({spec.task_id for spec in specs}) == 40
    assert {spec.task_id for spec in specs if spec.expected_task_type == "classification"} == CLASSIFICATION_IDS
    assert {spec.task_id for spec in specs if spec.expected_task_type == "regression"} == REGRESSION_IDS
    assert sum(spec.expected_task_type == "classification" for spec in specs) == 22
    assert sum(spec.expected_task_type == "regression" for spec in specs) == 18
    assert external.EXTERNAL_BENCHMARK_SUITE_VERSION == "1.0.0"
    assert all(spec.source_suite == 271 for spec in specs if spec.expected_task_type == "classification")
    assert all(spec.source_suite == 269 for spec in specs if spec.expected_task_type == "regression")


def test_external_cases_are_lazy_and_neutral():
    original_importer = external._import_openml
    external._import_openml = lambda: (_ for _ in ()).throw(AssertionError("OpenML was imported"))
    try:
        cases = external.external_benchmark_cases()
    finally:
        external._import_openml = original_importer
    assert len(cases) == 40
    assert all(case.role is BenchmarkRole.EXTERNAL_EVALUATION for case in cases)
    assert all(case.target_column == "__target__" for case in cases)
    assert all("dataset-specific" not in case.question for case in cases)
    assert all(case.openml_task_id is not None for case in cases)


def test_external_case_provenance_is_serialized_without_changing_local_metadata():
    external_case = replace(
        external.external_benchmark_cases()[0],
        dataframe=pd.DataFrame({"feature": [1, 2], "__target__": [0, 1]}),
        dataframe_loader=None,
    )
    metadata = external_case.as_dict()
    assert metadata["openml_task_id"] == 359983
    assert metadata["source_suite_id"] == 271
    assert metadata["benchmark_suite_version"] == "1.0.0"
    assert metadata["tier"] == "core"
    assert all("openml_task_id" not in case.as_dict() for case in default_benchmark_cases())


def test_loader_uses_current_openml_dataset_api_and_appends_canonical_target(monkeypatch):
    X = pd.DataFrame({"number": [1, 2, 3, 4], "category": ["a", "b", "a", "b"]})
    y = pd.Series(["yes", "no", "yes", "no"], name="label")
    module, calls = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)

    data = external.load_openml_task_data(_classification_spec())

    assert calls["get_task"] == (123, False)
    assert calls["get_data"] == ("label", "dataframe")
    assert list(data.frame.columns) == ["number", "category", "__target__"]
    assert data.frame["__target__"].tolist() == ["yes", "no", "yes", "no"]
    assert data.frame["category"].dtype == X["category"].dtype
    assert data.original_target_name == "label"
    assert data.dataset_id == 987


def test_loader_shape_mismatch_fails_loudly(monkeypatch):
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.Series([0, 1, 0])
    module, _ = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)

    with pytest.raises(ValueError, match=r"task ID 123.*expected shape rows=4, features=2.*actual shape rows=3, features=2.*fixture"):
        external.load_openml_task(_classification_spec())


def test_loader_class_count_mismatch_fails_loudly(monkeypatch):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    y = pd.Series([0, 1, 2, 0])
    module, _ = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)

    with pytest.raises(ValueError, match=r"task ID 123.*expected classes=2, actual classes=3.*fixture"):
        external.load_openml_task(_classification_spec())


def test_loader_target_collision_fails_loudly(monkeypatch):
    X = pd.DataFrame({"__target__": [1, 2, 3, 4], "other": [4, 5, 6, 7]})
    y = pd.Series([0, 1, 0, 1])
    module, _ = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)

    with pytest.raises(ValueError, match=r"task ID 123.*collides with the canonical target"):
        external.load_openml_task(_classification_spec())


def test_loader_rejects_non_numeric_regression_target(monkeypatch):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    y = pd.Series(["1", "2", "3", "4"])
    module, _ = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)
    spec = external.OpenMLBenchmarkSpec(124, "reg_fixture", "regression", 4, 2)

    with pytest.raises(ValueError, match=r"task ID 124.*is not numeric"):
        external.load_openml_task(spec)


def test_cache_directory_is_configured_only_when_requested(monkeypatch):
    X = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 5, 6, 7]})
    y = pd.Series([0, 1, 0, 1])
    module, calls = _fake_openml(X, y)
    monkeypatch.setattr(external, "_import_openml", lambda: module)
    cache = Path.cwd() / ".openml_cache" / "external-loader-test"
    monkeypatch.setenv(external.OPENML_CACHE_ENV_VAR, str(cache))

    try:
        external.load_openml_task(_classification_spec())
        assert cache.is_dir()
        assert calls["cache"] == cache.resolve()
    finally:
        shutil.rmtree(cache, ignore_errors=True)


def test_default_local_benchmark_registry_has_no_external_cases():
    cases = default_benchmark_cases()
    assert len(cases) == 18
    assert all(case.openml_task_id is None for case in cases)
    assert all(case.role in {BenchmarkRole.POLICY_DEVELOPMENT, BenchmarkRole.FINAL_EVALUATION} for case in cases)


def test_cli_defaults_to_local_and_forwards_external_suite_and_filters(monkeypatch, capsys):
    calls = []

    def fake_run_evaluation(output, **kwargs):
        calls.append((output, kwargs))
        return {"suite": kwargs["suite"], "cases": kwargs["case_names"], "tier": kwargs["tier"]}

    monkeypatch.setattr(evaluation_cli, "run_evaluation", fake_run_evaluation)
    monkeypatch.setattr(sys, "argv", ["evaluation.run"])
    evaluation_cli.main()
    assert calls[-1][1]["suite"] == "local"
    assert calls[-1][1]["case_names"] is None
    capsys.readouterr()

    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluation.run", "--suite", "external", "--case", "APSFailure", "--tier", "stress"],
    )
    evaluation_cli.main()
    assert calls[-1][1]["suite"] == "external"
    assert calls[-1][1]["case_names"] == ["APSFailure"]
    assert calls[-1][1]["tier"] == "stress"


def test_importing_external_module_does_not_import_openml(monkeypatch):
    imported = []
    real_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "openml":
            imported.append(name)
            raise AssertionError("OpenML import should be lazy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    sys.modules.pop("evaluation.external_benchmarks", None)
    try:
        __import__("evaluation.external_benchmarks")
    finally:
        import evaluation.external_benchmarks as reloaded
        assert reloaded.EXTERNAL_BENCHMARK_SUITE_VERSION == "1.0.0"
    assert imported == []
