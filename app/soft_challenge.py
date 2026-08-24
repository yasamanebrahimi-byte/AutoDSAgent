"""Selective intervention policy for deterministic model-family challenges.

The deterministic recommender's compatibility points are heuristic evidence.  This
module deliberately keeps that heuristic confidence separate from empirical
reliability learned offline from policy-development records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from app.schemas import ConfidenceLevel, DeterministicDiagnostics, Method


SOFT_CHALLENGE_POLICY_VERSION = "v1"
SOFT_CHALLENGE_CALIBRATION_SCHEMA_VERSION = "soft-challenge-calibration-v1"
DEFAULT_CALIBRATION_ARTIFACT = Path(__file__).with_name("soft_challenge_calibration.json")

Decision = Literal["agree", "challenge", "abstain"]


@dataclass(frozen=True)
class SoftChallengePolicy:
    """Explicit, conservative thresholds for selective intervention."""

    version: str = SOFT_CHALLENGE_POLICY_VERSION
    min_calibration_support: int = 8
    medium_confidence_min_reliability: float = 0.80
    high_confidence_min_reliability: float = 0.65
    catastrophic_regret_threshold: float = 0.10
    catastrophic_prevention_min_rate: float = 0.75
    min_catastrophic_support: int = 3
    low_confidence_margin: float = 7.0
    high_confidence_margin: float = 15.0
    high_dimensional_ratio_threshold: float = 3.0
    medium_dimensional_ratio_threshold: float = 8.0
    high_dimensional_effective_features: int = 100
    medium_dimensional_effective_features: int = 20
    calibration_artifact_path: str | None = None

    def __post_init__(self) -> None:
        if self.min_calibration_support < 1:
            raise ValueError("min_calibration_support must be positive.")
        if self.min_catastrophic_support < 1:
            raise ValueError("min_catastrophic_support must be positive.")
        if self.low_confidence_margin < 0 or self.high_confidence_margin <= self.low_confidence_margin:
            raise ValueError("Confidence margin thresholds must be non-negative and ordered.")
        for name in (
            "medium_confidence_min_reliability",
            "high_confidence_min_reliability",
            "catastrophic_prevention_min_rate",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")


@dataclass(frozen=True)
class SoftChallengeDecision:
    """Immutable result of the soft-challenge decision layer."""

    decision: Decision
    decision_reason: str
    agent_method: Method
    deterministic_method: Method
    deterministic_confidence: ConfidenceLevel
    score_margin: float | None
    training_row_count: int | None
    calibration_regime: str
    calibration_artifact_version: str | None
    calibration_support: int
    empirical_reliability: float | None
    challenge_win_rate: float | None
    challenge_loss_rate: float | None
    mean_regret_delta: float | None
    catastrophic_regret_prevention_rate: float | None
    catastrophic_regret_support: int
    policy_version: str

    @property
    def status(self) -> str:
        return {
            "agree": "agreement",
            "challenge": "challenged",
            "abstain": "abstained",
        }[self.decision]

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "status_detail": self.status,
            "decision_reason": self.decision_reason,
            "agent_method": self.agent_method,
            "deterministic_method": self.deterministic_method,
            "deterministic_confidence": self.deterministic_confidence,
            "score_margin": self.score_margin,
            "training_row_count": self.training_row_count,
            "calibration_regime": self.calibration_regime,
            "calibration_artifact_version": self.calibration_artifact_version,
            "calibration_support": self.calibration_support,
            "empirical_reliability": self.empirical_reliability,
            "challenge_win_rate": self.challenge_win_rate,
            "challenge_loss_rate": self.challenge_loss_rate,
            "mean_regret_delta": self.mean_regret_delta,
            "catastrophic_regret_prevention_rate": self.catastrophic_regret_prevention_rate,
            "catastrophic_regret_support": self.catastrophic_regret_support,
            "policy_version": self.policy_version,
        }


def load_calibration_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load a frozen calibration artifact without fitting or recomputing it."""

    artifact_path = Path(path) if path is not None else DEFAULT_CALIBRATION_ARTIFACT
    if not artifact_path.exists():
        return {
            "calibration_schema_version": SOFT_CHALLENGE_CALIBRATION_SCHEMA_VERSION,
            "calibration_artifact_version": "missing",
            "regimes": {},
        }
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to load soft-challenge calibration artifact {artifact_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Soft-challenge calibration artifact must contain a JSON object.")
    return payload


def _diagnostic_value(diagnostics: DeterministicDiagnostics | Mapping[str, Any] | None, key: str) -> Any:
    if diagnostics is None:
        return None
    if isinstance(diagnostics, Mapping):
        return diagnostics.get(key)
    return getattr(diagnostics, key, None)


def _dimensionality_regime(
    diagnostics: DeterministicDiagnostics | Mapping[str, Any] | None,
    policy: SoftChallengePolicy,
) -> str:
    ratio = _diagnostic_value(diagnostics, "sample_to_feature_ratio")
    effective = _diagnostic_value(diagnostics, "effective_features_estimate")
    if (
        ratio is not None
        and float(ratio) < policy.high_dimensional_ratio_threshold
    ) or (
        effective is not None
        and int(effective) >= policy.high_dimensional_effective_features
    ):
        return "high"
    if (
        ratio is not None
        and float(ratio) < policy.medium_dimensional_ratio_threshold
    ) or (
        effective is not None
        and int(effective) >= policy.medium_dimensional_effective_features
    ):
        return "medium"
    return "low"


def _complexity_regime(diagnostics: DeterministicDiagnostics | Mapping[str, Any] | None) -> str:
    nonlinearity_applicable = _diagnostic_value(diagnostics, "nonlinearity_applicable")
    signal = _diagnostic_value(diagnostics, "nonlinearity_signal")
    if nonlinearity_applicable is False or signal not in {"low", "moderate", "high"}:
        signal = _diagnostic_value(diagnostics, "structural_complexity_signal")
    return str(signal) if signal in {"low", "moderate", "high"} else "low"


def _margin_regime(score_margin: float | None, policy: SoftChallengePolicy) -> str:
    if score_margin is None or score_margin < policy.low_confidence_margin:
        return "low"
    if score_margin < policy.high_confidence_margin:
        return "medium"
    return "high"


def calibration_regime_key(
    *,
    task_type: str | None,
    diagnostics: DeterministicDiagnostics | Mapping[str, Any] | None,
    score_margin: float | None,
    policy: SoftChallengePolicy | None = None,
) -> str:
    """Return the small, interpretable calibration regime used by runtime."""

    policy = policy or SoftChallengePolicy()
    task = task_type if task_type in {"classification", "regression"} else "unknown"
    return "/".join(
        (
            str(task),
            _dimensionality_regime(diagnostics, policy),
            _complexity_regime(diagnostics),
            _margin_regime(score_margin, policy),
        )
    )


def _candidate_regime_keys(regime_key: str) -> list[str]:
    task, dimensionality, complexity, margin = regime_key.split("/", 3)
    # Exact regimes are preferred.  Backoff remains interpretable and avoids
    # making every three-way interaction a required dense calibration cell.
    return [
        regime_key,
        f"{task}/{dimensionality}/all/{margin}",
        f"{task}/all/all/{margin}",
        f"{task}/all/all/all",
        "all/all/all/all",
    ]


def _regime_records(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    regimes = artifact.get("regimes", {})
    return regimes if isinstance(regimes, Mapping) else {}


def _lookup_calibration(
    artifact: Mapping[str, Any],
    regime_key: str,
    policy: SoftChallengePolicy,
) -> tuple[str, Mapping[str, Any] | None]:
    regimes = _regime_records(artifact)
    selected_key: str | None = None
    selected: Mapping[str, Any] | None = None
    for candidate_key in _candidate_regime_keys(regime_key):
        candidate = regimes.get(candidate_key)
        if isinstance(candidate, Mapping):
            selected_key = candidate_key
            selected = candidate
            if int(candidate.get("support", candidate.get("count", 0)) or 0) >= policy.min_calibration_support:
                break
    return selected_key or regime_key, selected


def decide_soft_challenge(
    *,
    agent_method: Method,
    deterministic_method: Method,
    deterministic_confidence: ConfidenceLevel,
    score_margin: float | None,
    diagnostics: DeterministicDiagnostics | Mapping[str, Any] | None,
    task_type: str | None = None,
    training_row_count: int | None = None,
    policy: SoftChallengePolicy | None = None,
    calibration_artifact: Mapping[str, Any] | None = None,
    strategy: str = "calibrated",
) -> SoftChallengeDecision:
    """Decide whether a deterministic model-family disagreement merits intervention.

    Raw compatibility margins only select a regime and are recorded as heuristic
    evidence.  They never authorize a challenge without sufficient empirical
    reliability support from the frozen calibration artifact.
    """

    if strategy not in {"calibrated", "high_confidence_only"}:
        raise ValueError(f"Unsupported soft-challenge strategy: {strategy!r}")
    policy = policy or SoftChallengePolicy()
    artifact = calibration_artifact or load_calibration_artifact(policy.calibration_artifact_path)
    regime_key = calibration_regime_key(
        task_type=task_type,
        diagnostics=diagnostics,
        score_margin=score_margin,
        policy=policy,
    )
    selected_key, selected = _lookup_calibration(artifact, regime_key, policy)
    support = int((selected or {}).get("support", (selected or {}).get("count", 0)) or 0)
    reliability_value = (selected or {}).get("empirical_reliability")
    if reliability_value is None:
        reliability_value = (selected or {}).get("challenge_success_rate")
    if reliability_value is None:
        reliability_value = (selected or {}).get("challenge_win_rate")
    reliability = float(reliability_value) if reliability_value is not None else None
    win_rate = (selected or {}).get("challenge_win_rate", reliability)
    loss_rate = (selected or {}).get("challenge_loss_rate")
    mean_regret_delta = (selected or {}).get("mean_regret_delta")
    catastrophic_rate = (selected or {}).get("catastrophic_regret_prevention_rate")
    catastrophic_support = int((selected or {}).get("catastrophic_regret_support", 0) or 0)
    artifact_version = artifact.get("calibration_artifact_version")

    common = {
        "agent_method": agent_method,
        "deterministic_method": deterministic_method,
        "deterministic_confidence": deterministic_confidence,
        "score_margin": float(score_margin) if score_margin is not None else None,
        "training_row_count": training_row_count,
        "calibration_regime": selected_key,
        "calibration_artifact_version": str(artifact_version) if artifact_version is not None else None,
        "calibration_support": support,
        "empirical_reliability": reliability,
        "challenge_win_rate": float(win_rate) if win_rate is not None else None,
        "challenge_loss_rate": float(loss_rate) if loss_rate is not None else None,
        "mean_regret_delta": float(mean_regret_delta) if mean_regret_delta is not None else None,
        "catastrophic_regret_prevention_rate": float(catastrophic_rate) if catastrophic_rate is not None else None,
        "catastrophic_regret_support": catastrophic_support,
        "policy_version": policy.version,
    }

    if agent_method == deterministic_method:
        return SoftChallengeDecision(decision="agree", decision_reason="model_family_agreement", **common)
    if strategy == "high_confidence_only":
        if deterministic_confidence == "high":
            return SoftChallengeDecision(
                decision="challenge",
                decision_reason="high_confidence_only_strategy",
                **common,
            )
        return SoftChallengeDecision(
            decision="abstain",
            decision_reason=f"{deterministic_confidence}_confidence_abstention",
            **common,
        )
    if deterministic_confidence == "low":
        return SoftChallengeDecision(
            decision="abstain",
            decision_reason="low_deterministic_confidence",
            **common,
        )
    if selected is None or support < policy.min_calibration_support:
        return SoftChallengeDecision(
            decision="abstain",
            decision_reason="insufficient_calibration_support",
            **common,
        )
    if reliability is not None:
        threshold = (
            policy.medium_confidence_min_reliability
            if deterministic_confidence == "medium"
            else policy.high_confidence_min_reliability
        )
        if reliability >= threshold:
            return SoftChallengeDecision(
                decision="challenge",
                decision_reason=(
                    "medium_confidence_and_strong_calibrated_reliability"
                    if deterministic_confidence == "medium"
                    else "high_confidence_and_calibrated_reliability"
                ),
                **common,
            )
    if (
        deterministic_confidence == "high"
        and catastrophic_rate is not None
        and catastrophic_support >= policy.min_catastrophic_support
        and catastrophic_rate >= policy.catastrophic_prevention_min_rate
    ):
        return SoftChallengeDecision(
            decision="challenge",
            decision_reason="high_confidence_and_catastrophic_regret_protection",
            **common,
        )
    return SoftChallengeDecision(
        decision="abstain",
        decision_reason="calibrated_reliability_below_intervention_threshold",
        **common,
    )


def empty_calibration_artifact() -> dict[str, Any]:
    """Return an explicit conservative artifact for callers without calibration."""

    return {
        "calibration_schema_version": SOFT_CHALLENGE_CALIBRATION_SCHEMA_VERSION,
        "calibration_artifact_version": "empty",
        "regimes": {},
    }
