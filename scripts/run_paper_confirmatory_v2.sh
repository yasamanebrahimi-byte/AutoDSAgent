#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY must be set for the live confirmatory run." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "The repository must have a clean working tree before a confirmatory run." >&2
  exit 1
fi

MANIFEST="$ROOT_DIR/evaluation/configs/paper_confirmatory_v2.json"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing v2 manifest: $MANIFEST" >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.12 is required; set PYTHON_BIN to a Python 3.12 executable." >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-paper-confirmatory-v2}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -e ".[dev,benchmark]"

"$PYTHON" - "$ROOT_DIR" "$MANIFEST" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from evaluation.confirmatory import experiment_code_sha256

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") != "frozen":
    raise SystemExit("paper_confirmatory_v2.json must be frozen before execution")
expected = manifest.get("expected_experiment_code_sha256")
observed = experiment_code_sha256(root)
if not expected or expected != observed:
    raise SystemExit(
        "v2 experiment-code SHA mismatch: "
        f"expected={expected!r}, observed={observed!r}"
    )
source_commit = manifest.get("source_git_commit")
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
if source_commit:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, head],
        check=True,
    )
print(f"Validated frozen v2 manifest and experiment code SHA {observed}.")
print(f"HEAD {head}; source implementation {source_commit or 'not declared'}.")
PY

"$PYTHON" -m pytest \
  tests/test_llm.py \
  tests/test_validation.py \
  tests/test_confirmatory.py \
  tests/test_ablation.py \
  -q

PREFETCH_OUTPUT="${PREFETCH_OUTPUT:-$ROOT_DIR/runs/paper_confirmatory_v2_benchmark_manifest.json}"
"$PYTHON" scripts/prefetch_external_benchmarks.py --output "$PREFETCH_OUTPUT" --fail-fast

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/evaluation_results/paper_confirmatory_v2_$(date -u +%Y%m%dT%H%M%SZ)}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
elif [[ -e "$OUTPUT_DIR" ]]; then
  echo "OUTPUT_DIR already exists; choose a fresh directory or set RESUME=1." >&2
  exit 1
fi

"$PYTHON" -m evaluation.ablation \
  --suite external \
  --require-live \
  --confirmatory-config "$MANIFEST" \
  --output "$OUTPUT_DIR" \
  "${RESUME_FLAG[@]}"

"$PYTHON" - "$OUTPUT_DIR/ablation_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if summary.get("fallback_rows") != 0:
    raise SystemExit(f"fallback rows are not zero: {summary.get('fallback_rows')}")
if not summary.get("confirmatory_valid"):
    raise SystemExit("confirmatory_valid is false")
matrix = summary.get("confirmatory_matrix") or {}
if not matrix.get("complete"):
    raise SystemExit(f"confirmatory matrix is incomplete: {matrix}")
if matrix.get("expected_unit_count") != 2520 or matrix.get("observed_unit_count") != 2520:
    raise SystemExit(f"expected complete 2520-unit matrix, got {matrix}")
print(f"Validated complete live v2 matrix: {matrix['observed_unit_count']}/2520 units.")
print(f"Output directory: {Path(sys.argv[1]).parent}")
PY
