"""Prospective fresh-split/fresh-panel experiment support.

This module is intentionally a separate entry point from the historical
confirmatory runner.  It accepts a human-reviewed, frozen task-panel manifest
and forwards explicitly declared split seeds to the existing ablation runner.
No panel is sampled or frozen by this module.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Sequence

from evaluation.ablation import run_ablation_study
from evaluation.external_benchmarks import (
    PROSPECTIVE_ANALYSIS_ROLE,
    load_prospective_panel_manifest,
    prospective_benchmark_cases,
    prospective_panel_content_sha256,
    validate_prospective_panel_manifest,
)


def run_prospective_experiment(
    output_dir: str | Path,
    panel_manifest: str | Path,
    *,
    split_seeds: Sequence[int],
    repetitions: int = 1,
    model: str = "gpt-4.1-mini",
    provider: str = "openai",
    planner_model: str | None = None,
    reconciler_model: str | None = None,
    offline: bool = False,
    require_live: bool = False,
    ablations: Sequence[str] | None = None,
    case_names: Sequence[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run a future experiment on an already-frozen new task panel.

    ``split_seeds`` must be supplied by the caller rather than selected from
    observed outcomes.  The existing runner preserves split identity in trial
    IDs, proposal-cache keys, empirical-reference cache keys, and artifacts.
    """

    manifest = load_prospective_panel_manifest(panel_manifest)
    validate_prospective_panel_manifest(manifest, require_frozen=True)
    selected_seeds = tuple(int(seed) for seed in split_seeds)
    if not selected_seeds:
        raise ValueError("At least one prospective split seed must be explicitly declared.")
    if len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Prospective split seeds must be unique.")
    output = Path(output_dir).resolve()
    source_manifest = Path(panel_manifest).resolve()
    if output == source_manifest:
        raise ValueError("Prospective experiment output must be separate from the panel manifest.")
    copied_manifest = output / "prospective_task_panel_manifest.json"
    if resume:
        if not copied_manifest.is_file():
            raise ValueError("Prospective resume requires the copied panel manifest artifact.")
        existing_manifest = load_prospective_panel_manifest(copied_manifest)
        validate_prospective_panel_manifest(existing_manifest, require_frozen=True)
        if hashlib_manifest(existing_manifest) != hashlib_manifest(manifest):
            raise ValueError("Prospective resume panel manifest differs from the existing output artifact.")

    result = run_ablation_study(
        output,
        cases=prospective_benchmark_cases(manifest),
        split_seeds=selected_seeds,
        repetitions=repetitions,
        model=model,
        provider=provider,
        planner_model=planner_model,
        reconciler_model=reconciler_model,
        offline=offline,
        require_live=require_live,
        ablations=ablations,
        case_names=case_names,
        resume=resume,
        # The panel itself is supplied explicitly as cases; no historical
        # confirmatory manifest is accepted on this prospective path.
        suite="local",
    )
    if not resume:
        shutil.copyfile(source_manifest, copied_manifest)
    metadata = {
        "analysis_role": PROSPECTIVE_ANALYSIS_ROLE,
        "prospective_panel_id": manifest.get("panel_id"),
        "prospective_panel_manifest_sha256": hashlib_manifest(manifest),
        "prospective_panel_content_sha256": prospective_panel_content_sha256(manifest),
        "prospective_panel_manifest_path": str(copied_manifest),
        "prospective_split_seeds": list(selected_seeds),
        "split_selection_rule": "explicitly declared before execution; not selected from outcomes",
        "historical_confirmatory_manifest_used": False,
    }
    _annotate_prospective_trials(output, metadata)
    config_path = output / "config.json"
    summary_path = output / "ablation_summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(metadata)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(metadata)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    result["config"] = config
    result["summary"].update(metadata)
    return result


def _annotate_prospective_trials(output: Path, metadata: dict[str, Any]) -> None:
    """Stamp every newly generated trial artifact with its prospective role."""

    for trials_path in output.rglob("trials.jsonl"):
        rows: list[dict[str, Any]] = []
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    value.update({
                        "analysis_role": PROSPECTIVE_ANALYSIS_ROLE,
                        "prospective_panel_id": metadata["prospective_panel_id"],
                        "prospective_panel_content_sha256": metadata["prospective_panel_content_sha256"],
                        "prospective_split_seeds": metadata["prospective_split_seeds"],
                    })
                rows.append(value)
        trials_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def hashlib_manifest(manifest: dict[str, Any]) -> str:
    """Hash a manifest without making this wrapper depend on confirmatory code."""

    import hashlib

    return hashlib.sha256(
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parse_seeds(values: Sequence[str] | None) -> list[int]:
    if not values:
        return []
    result: list[int] = []
    for value in values:
        result.extend(int(item.strip()) for item in value.split(",") if item.strip())
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a prospective AutoDSAgent experiment from a frozen task-panel manifest."
    )
    parser.add_argument("--panel-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", action="append", dest="split_seeds")
    parser.add_argument("--split-seeds", action="append", dest="split_seeds_alias")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--provider", choices=("openai", "google"), default="openai")
    parser.add_argument("--planner-model")
    parser.add_argument("--reconciler-model")
    parser.add_argument("--ablation", action="append", dest="ablations")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    seeds = _parse_seeds((args.split_seeds or []) + (args.split_seeds_alias or []))
    if not seeds:
        parser.error("at least one --split-seed is required")
    result = run_prospective_experiment(
        args.output,
        args.panel_manifest,
        split_seeds=seeds,
        repetitions=args.repetitions,
        model=args.model,
        provider=args.provider,
        planner_model=args.planner_model,
        reconciler_model=args.reconciler_model,
        offline=args.offline,
        require_live=args.require_live,
        ablations=args.ablations,
        case_names=args.cases,
        resume=args.resume,
    )
    print(json.dumps({"output_dir": result["output_dir"], "analysis_role": PROSPECTIVE_ANALYSIS_ROLE}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["run_prospective_experiment", "main"]
