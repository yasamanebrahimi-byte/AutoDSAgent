import json

from app.llm import OpenAIAgents
from app.reconciliation import (
    build_blinded_reconciliation,
    infer_selected_proposal,
    normalize_reconciliation_candidate,
)
from app.schemas import DeterministicRecommendation, ModelingPlan, ModelingResolution
from evaluation.metrics import _reconciliation_rates


def _profile():
    return {
        "rows": 20,
        "columns": 3,
        "column_details": [
            {"name": "x", "dtype": "float", "semantic_type": "numeric", "missing": 0, "unique": 20, "identifier_like": False},
            {"name": "y", "dtype": "float", "semantic_type": "numeric", "missing": 0, "unique": 20, "identifier_like": False},
            {"name": "target", "dtype": "int", "semantic_type": "numeric", "missing": 0, "unique": 2, "identifier_like": False},
        ],
    }


def _plans():
    agent = ModelingPlan(
        recommended_method="linear",
        reasoning="This proposal reasons from the observed training schema and relationship structure.",
        confidence=0.6,
    )
    deterministic = DeterministicRecommendation(
        target_column="target",
        task_type="classification",
        recommended_method="tree_ensemble",
        reasoning="The deterministic policy selected a family from structural evidence.",
        evidence=["training_rows=20"],
        method_scores={"linear": 78, "regularized_linear": 66, "tree_ensemble": 61, "boosted_tree": 48},
    )
    return agent, deterministic


def test_blinded_payload_hides_sources_scores_and_is_symmetric():
    agent, deterministic = _plans()
    blinded = build_blinded_reconciliation(
        _profile(),
        agent,
        deterministic,
        target_column="target",
        task_type="classification",
        order_seed=1,
    )
    encoded = json.dumps(blinded.payload, sort_keys=True).lower()
    assert "proposal_a" in encoded and "proposal_b" in encoded
    assert "agent proposal" not in encoded
    assert "deterministic proposal" not in encoded
    assert "deterministic validator" not in encoded
    assert "llm proposal" not in encoded
    assert "method_scores" not in encoded
    assert '"ranked_methods"' not in encoded
    assert "78" not in encoded
    assert "61" not in encoded
    assert set(blinded.payload["proposal_a"]) == set(blinded.payload["proposal_b"])
    assert "observed_statements" in blinded.payload["dataset_evidence"]


def test_candidates_use_identical_canonical_schema_and_types():
    agent, deterministic = _plans()
    evidence = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification"
    ).payload["dataset_evidence"]
    first = normalize_reconciliation_candidate(agent, evidence, task_type="classification")
    second = normalize_reconciliation_candidate(deterministic, evidence, task_type="classification")
    assert first.keys() == second.keys()
    assert {key: type(value) for key, value in first.items()} == {
        key: type(value) for key, value in second.items()
    }
    assert all(first[field] and second[field] for field in (
        "rationale", "supporting_evidence", "assumptions", "risks",
        "constraints", "expected_failure_modes",
    ))
    assert not {"reasoning", "confidence", "method_scores", "evidence"} & set(first)


def test_raw_planner_rationale_is_not_in_reconciliation_payload():
    agent, deterministic = _plans()
    distinctive = "DISTINCTIVE_RAW_PLANNER_RATIONALE_7f2a"
    agent = agent.model_copy(update={"reasoning": distinctive})
    blinded = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification"
    )
    assert distinctive not in json.dumps(blinded.payload, sort_keys=True)
    assert distinctive not in json.dumps(blinded.payload["proposal_a"], sort_keys=True)
    assert distinctive not in json.dumps(blinded.payload["proposal_b"], sort_keys=True)


def test_deterministic_metadata_does_not_create_a_template_fingerprint():
    agent, deterministic = _plans()
    blinded = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification"
    )
    assert "training_rows=20" not in json.dumps(blinded.payload)
    assert set(blinded.payload["proposal_a"]) == set(blinded.payload["proposal_b"])
    assert len(blinded.payload["proposal_a"]["rationale"]) == len(blinded.payload["proposal_b"]["rationale"])


def test_canonical_ab_swap_changes_only_candidate_assignment():
    agent, deterministic = _plans()
    first = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification",
        proposal_order=("agent", "deterministic"),
    ).payload
    swapped = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification",
        proposal_order=("deterministic", "agent"),
    ).payload
    assert first["proposal_a"] == swapped["proposal_b"]
    assert first["proposal_b"] == swapped["proposal_a"]
    assert set(first) == set(swapped)


def test_normalization_is_deterministic():
    agent, deterministic = _plans()
    evidence = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification"
    ).payload["dataset_evidence"]
    assert normalize_reconciliation_candidate(
        agent, evidence, task_type="classification"
    ) == normalize_reconciliation_candidate(
        agent, evidence, task_type="classification"
    )


def test_payload_has_no_provenance_terms_or_private_recommendation_metadata():
    agent, deterministic = _plans()
    payload = build_blinded_reconciliation(
        _profile(), agent, deterministic, target_column="target", task_type="classification"
    ).payload
    encoded = json.dumps(payload, sort_keys=True).lower()
    assert not any(term in encoded for term in (
        '"agent"', '"llm"', '"planner"', '"deterministic"',
        '"heuristic"', '"validator"', '"method_scores"', '"ranked_methods"',
    ))


def test_proposal_order_is_seeded_and_not_fixed():
    agent, deterministic = _plans()
    first = build_blinded_reconciliation(_profile(), agent, deterministic, target_column="target", task_type="classification", order_seed=1)
    repeat = build_blinded_reconciliation(_profile(), agent, deterministic, target_column="target", task_type="classification", order_seed=1)
    other = build_blinded_reconciliation(_profile(), agent, deterministic, target_column="target", task_type="classification", order_seed=2)
    assert (first.proposal_a_source, first.proposal_b_source) == (repeat.proposal_a_source, repeat.proposal_b_source)
    assert (first.proposal_a_source, first.proposal_b_source) != (other.proposal_a_source, other.proposal_b_source)


def test_selection_maps_back_to_source_after_order_swap():
    agent, deterministic = _plans()
    blinded = build_blinded_reconciliation(_profile(), agent, deterministic, target_column="target", task_type="classification", order_seed=2)
    assert blinded.proposal_a_source == "deterministic"
    result = ModelingResolution(
        selected_method="linear",
        selected_proposal="B",
        selected_preprocessing=agent.preprocessing,
        checks=["two_sided_review"],
        justification="Proposal B better matches the observed training-data evidence and its preprocessing assumptions.",
        confidence=0.7,
    )
    assert infer_selected_proposal(result, blinded) == ("B", "agent")


def test_live_reconciliation_prompt_is_blinded_and_omits_raw_scores():
    agent, deterministic = _plans()
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured["input"] = kwargs["input"]
            return type("Response", (), {
                "output_parsed": ModelingResolution(
                    selected_method="linear",
                    selected_proposal="A",
                    checks=["two_sided_review"],
                    justification="Proposal A has the stronger methodological fit to the observed evidence.",
                    confidence=0.7,
                )
            })()

    class Client:
        responses = Responses()

    agents = OpenAIAgents(api_key="test")
    agents._client = Client()
    agents.reconcile_modeling(
        "classify target",
        _profile(),
        agent,
        {
            **deterministic.model_dump(mode="json"),
            "_reconciliation_order_seed": 1,
        },
    )
    prompt = captured["input"].lower()
    assert "proposal a" in prompt and "proposal b" in prompt
    assert "agent proposal" not in prompt
    assert "deterministic proposal" not in prompt
    assert "deterministic validator" not in prompt
    assert "llm proposal" not in prompt
    assert "method_scores" not in prompt
    assert "78" not in prompt and "61" not in prompt
    assert "proposal_a" in prompt and "proposal_b" in prompt


def test_empirical_probe_is_carried_as_source_neutral_evidence():
    agent, deterministic = _plans()
    probe = {
        "status": "completed",
        "metric": "macro_f1",
        "cv_folds": 3,
        "proposal_a": {"model_family": "linear", "mean_score": 0.80, "std_score": 0.01, "fold_wins": 3},
        "proposal_b": {"model_family": "tree_ensemble", "mean_score": 0.72, "std_score": 0.02, "fold_wins": 0},
        "winner": "A",
        "evidence_strength": "strong",
    }
    blinded = build_blinded_reconciliation(
        _profile(),
        agent,
        deterministic,
        target_column="target",
        task_type="classification",
        order_seed=1,
        empirical_probe=probe,
    )
    encoded = json.dumps(blinded.payload, sort_keys=True).lower()
    assert "empirical_probe" in blinded.payload
    assert "proposal_a" in encoded and "proposal_b" in encoded
    assert "agent" not in encoded
    assert "deterministic" not in encoded


def test_reconciliation_metrics_report_position_and_order_swap_bias():
    records = [
        {"reconciliation_status": "succeeded", "selected_proposal": "A", "selected_proposal_source": "agent", "reconciliation_method_source": "agent", "order_swap_pair_id": "case-1"},
        {"reconciliation_status": "succeeded", "selected_proposal": "B", "selected_proposal_source": "agent", "reconciliation_method_source": "agent", "order_swap_pair_id": "case-1"},
        {"reconciliation_status": "succeeded", "selected_proposal": "A", "selected_proposal_source": "agent", "reconciliation_method_source": "agent", "order_swap_pair_id": "case-2"},
        {"reconciliation_status": "succeeded", "selected_proposal": "A", "selected_proposal_source": "deterministic", "reconciliation_method_source": "deterministic", "order_swap_pair_id": "case-2"},
    ]
    metrics = _reconciliation_rates(records)
    assert metrics["reconciliation_a_selected_rate"] == 0.75
    assert metrics["reconciliation_b_selected_rate"] == 0.25
    assert metrics["order_swap_consistency_rate"] == 0.5
    assert metrics["order_swap_flip_rate"] == 0.5
