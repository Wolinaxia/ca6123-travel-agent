# M1 fix round 3 — Gemini classifier never worked against the real API

Work ONLY inside this repository. Do not spawn sub-agents. No new deps. Do not touch `travel_agent/core/**`, `travel_agent/tools/**`, `travel_agent/schemas.py`. Never print or log the API key.

## Root cause (verified against the real API)
`travel_agent/guardrails/llm_classifier.py`: `_SCHEMA` (passed as `response_schema`) describes a single object `{"injection": bool, "reason": str}`, but `classify_injections` parses `{"results": [{"index", "injection", "reason"}]}`. With a schema, Gemini returns the single-object shape, so every call yields "classifier returned no verdict" → treated as unavailable, and that failure is **cached**. Unit tests passed only because the fake client returned the shape the parser expects.

Also: `gemini-2.5-flash-lite` returns 404 "no longer available to new users" for our course key; `gemini-3.5-flash-lite` works.

## K1 — schema matches parser
`_SCHEMA` = object with required `results`: array of objects with required `index` (integer), `injection` (boolean), `reason` (string). Keep the batched design.

## K2 — never cache failures
Only cache verdicts with `available=True`. "No verdict", errors and timeouts are not stored.

## K3 — fakes must follow the schema
In tests, the fake client must build its response from the `response_schema` actually passed in `config` (e.g. assert the schema has `results` and return a payload valid against it), so a schema/parser mismatch fails the test. Add a test that reproduces the original bug: a fake that returns the single-object shape must lead to `available=False` and must not be cached.

## K4 — eval must not report a broken classifier as "enabled"
`eval/run_eval.py --llm`: count classifier outcomes per case (available / unavailable). Print e.g. `classifier: 117/117 calls answered`. If any call was unavailable, print `classifier DEGRADED: N/M calls failed (reasons: …)` and label the layer-1+2 numbers as not valid. Exit code non-zero when `--llm` was requested and more than 10% of calls failed.

## K5 — model default
Default classifier model `gemini-3.5-flash-lite` (env `GUARDRAIL_MODEL` still overrides). Also change the default in `build_verifier_agent` and the dev-model mentions in `README.md` and `CONTRIBUTING.md` from `gemini-2.5-flash-lite` to `gemini-3.5-flash-lite`.

## K6 — opt-in live smoke test
`tests/test_live_classifier.py`: skipped unless env `RUN_LIVE=1` and a key is present. Calls `classify_injection` on one obvious injection and one benign travel question; asserts `available=True`, injection True / False respectively. Must not run in the default offline suite.

## Final report
Per item done/not done, files, tests; default offline `pytest -q` result.
