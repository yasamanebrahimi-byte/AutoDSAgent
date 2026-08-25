from evaluation.reporting import render_summary_markdown


def test_dataset_table_uses_per_dataset_challenges_and_abstentions():
    report = render_summary_markdown(
        config={},
        trials=[],
        summary={
            "by_dataset": {
                "mixed_case": {
                    "openai_trial_count": 1,
                    "openai_only": {
                        "challenges": 1,
                        "abstentions": 2,
                    },
                }
            }
        },
    )

    assert "| mixed_case | 1 | 1 | 2 |" in report


def test_report_names_deterministic_challenger_advantage_and_derives_split_limitation():
    report = render_summary_markdown(
        config={"split_seeds": [42, 123, 2027]},
        trials=[],
        summary={"mean_performance_delta_conditional_on_challenge": 0.125},
    )

    assert "Mean deterministic-challenger regret advantage: **0.1250**" in report
    assert "agent normalized regret - deterministic challenger normalized regret" in report
    assert "Three train/holdout splits and a small benchmark suite still do not establish broad domain generalization." in report
    assert "Mean challenge regret improvement" not in report
