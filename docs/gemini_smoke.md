# Gemini provider smoke test

This smoke test makes four live Gemini API calls against a synthetic in-memory
regression profile: one planner and one blinded reconciliation call for each
of `gemini-3.8-flash` and `gemini-3.5-flash-lite`. It does not load the
external benchmark, inspect holdout outcomes, or write API credentials.

Install the project dependencies, then run:

```bash
export GEMINI_API_KEY="your-key"
python scripts/run_gemini_smoke.py
```

The JSON output should show `provider` as `google`, `fallback_used` as
`false`, `planner_schema` as `ModelingPlan`, `reconciler_schema` as
`ModelingResolution`, the requested model identifier, and provider provenance
for both calls. Token fields are populated when Gemini returns usage metadata;
`token_accounting_available` may be `false` if the provider omits it.

The primary replication uses `reasoning_effort: "medium"` for both models.
The provider transport sends this as Gemini `thinking_config.thinking_level:
"medium"` and omits null temperature, top-p, and seed values.
