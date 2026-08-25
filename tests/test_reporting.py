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
