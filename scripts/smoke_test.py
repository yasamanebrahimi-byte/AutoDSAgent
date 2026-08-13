"""Lightweight smoke checks for local AutoDS Agent setup."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_DIRECTORIES = [
    "app",
    "app/backend",
    "app/frontend",
    "app/tools",
    "app/workflows",
    "examples/sample_data",
    "runs",
    "tests",
]

REQUIRED_DATASETS = {
    "examples/sample_data/regression_housing.csv": "sale_price",
    "examples/sample_data/classification_churn.csv": "churn",
}

REQUIRED_MODULES = [
    "app.backend.main",
    "app.backend.config",
    "app.backend.services.run_manager",
    "app.backend.services.workflow_service",
    "app.backend.services.report_service",
    "app.frontend.streamlit_app",
]


def main() -> None:
    args = parse_args()
    checks: list[tuple[str, Callable[[], bool]]] = [
        ("directories", check_directories),
        ("example datasets", check_example_datasets),
        ("module imports", check_imports),
        ("settings", check_settings),
    ]
    if args.backend_url:
        checks.append(("backend health", lambda: check_backend_health(args.backend_url)))

    results = [(name, check()) for name, check in checks]
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"Smoke test summary: {passed} passed, {failed} failed.")
    if failed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight project smoke checks.")
    parser.add_argument(
        "--backend-url",
        default=None,
        help="Optional backend URL. When provided, /health is checked.",
    )
    return parser.parse_args()


def check_directories() -> bool:
    ok = True
    for relative_path in REQUIRED_DIRECTORIES:
        path = PROJECT_ROOT / relative_path
        if path.is_dir():
            print(f"[ok] directory {relative_path}")
        else:
            print(f"[fail] directory {relative_path}")
            ok = False
    return ok


def check_example_datasets() -> bool:
    ok = True
    for relative_path, target_column in REQUIRED_DATASETS.items():
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            print(f"[fail] missing dataset {relative_path}")
            ok = False
            continue
        dataframe = pd.read_csv(path)
        if dataframe.empty:
            print(f"[fail] empty dataset {relative_path}")
            ok = False
        elif target_column not in dataframe.columns:
            print(f"[fail] {relative_path} missing target {target_column}")
            ok = False
        else:
            print(
                f"[ok] dataset {relative_path} "
                f"rows={len(dataframe)} target={target_column}"
            )
    return ok


def check_imports() -> bool:
    ok = True
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            print(f"[fail] import {module_name}: {exc}")
            ok = False
        else:
            print(f"[ok] import {module_name}")
    return ok


def check_settings() -> bool:
    from app.backend.config import load_settings

    loaded = load_settings()
    if not loaded.runs_dir.is_absolute():
        print("[fail] runs_dir is not absolute")
        return False
    print(f"[ok] runs_dir {loaded.runs_dir}")
    print(f"[ok] mlflow_enabled {loaded.mlflow_enabled}")
    return True


def check_backend_health(backend_url: str) -> bool:
    url = backend_url.rstrip("/") + "/health"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[fail] backend health {url}: {exc}")
        return False

    payload = response.json()
    if payload.get("status") != "ok":
        print(f"[fail] backend health {url}: {payload}")
        return False
    print(f"[ok] backend health {url}")
    return True


if __name__ == "__main__":
    main()
