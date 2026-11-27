# Verifier: two causes of false holds on real claims

## Context
- Repo: ca6123-travel-agent, Python, pytest. Only work inside this repository; do not read `.env`; no
  network; do not run `python -m eval.run_eval` without `--help` (it overwrites a tracked results file);
  do not run anything under `eval/ablation.py` against `eval/results/`.
- File: `travel_agent/verification/checks.py` (`verify_claim`, `_rule_levels`, `_data_numbers`,
  `_evidence_numbers`, `_extract_numbers`). Tests: `tests/test_verifier.py`, cases in
  `eval/verifier_cases.json` (35 cases, loaded by a test).
- The Verifier ablation (260 real claims from Gemini on recorded scenarios, labelled blind by an
  independent labeller) found that the Verifier held back 67 of 253 claims labelled correct (26%). Two
  causes account for all 67. This change is made **after** seeing that result; say so in the code comments
  where the behaviour changes (one line each, pointing to `docs/VERIFIER_FALSE_HOLDS_FIX.md`).

### Cause 1 (40 holds): single-airport observations compared with the trip-level risk
`_rule_levels` adds the rule engine's **route summary** level whenever the claim (or a cited subject)
mentions any airport on that route (`same_airport`). The summary is about the whole trip. So on a trip the
engine rates `elevated` or `high`:
- "As of ..., there are no active NWS alerts near LGA." (risk `low`, cites `nws:LGA:no_alerts`)
- "BOS currently has no active FAA delay programs." (risk `low`, cites the clear FAA status)
- "BOS weather is currently VFR with 10+ SM visibility and calm winds." (risk `low`)
- "LGA has a Ground Delay Program averaging 148 minutes ..." (risk `moderate`, trip `high`)

are all returned as `contradicted: downplays rule-engine risk`. A claim's `risk` describes what that claim
observes; only a statement about the trip can downplay the trip's risk.

### Cause 2 (27 holds): numbers that are in the evidence but not read
A forecast evidence looks like:
```json
{"observation": "Light Rain (wind N 14 mph)",
 "data": {"short_forecast": "Light Rain",
          "detailed_forecast": "Rain. Cloudy, with a low around 57. North wind around 14 mph, with gusts as high as 26 mph. Chance of precipitation is 80%. New rainfall amounts between a quarter and half of an inch possible.",
          "wind": "N 14 mph", "precipitation_probability": 80}}
```
Claims "gusts up to 26 mph" and "80% chance of precipitation" are returned as
`contradicted: states 26 mph but cited evidence does not` and
`unsupported: states a percent value but cited evidence has none`. `_data_numbers` only keeps a value when
its **key** implies a unit, so `precipitation_probability: 80` and every number inside the free-text
strings are ignored.

## Task

### 1. Risk comparison (`_rule_levels` and its use in `verify_claim`)
Keep: levels from cited engine findings; route summaries matched by a cited route (`same_route`) or by a
cited subject equal to the summary's subject (`same_subject`).

Change the `same_airport` path: a route summary is compared only when the claim is a **trip-level
statement**, defined as: the claim text names **at least two different airports of that route** (e.g.
"LGA to BOS", "JFK–SFO"), using the existing `_claim_places`. One airport in the text, or an airport that
only comes from a cited subject, no longer pulls in the route summary.

Add a floor from the claim's own cited evidence, checked when `risk` is set, before the rule-level
comparison:
- a cited, trusted, unexpired `faa_nas` evidence whose `data.program` is `GDP` or `arrival_departure`
  requires at least `moderate`; `ground_stop` or `closure` requires `high`;
- skip an evidence whose `data.affects_commercial` is `False` (general-aviation-only restrictions);
- risk below the floor -> `contradicted`, reason `downplays cited FAA program`, check name
  `risk_consistent`.

Everything else in `verify_claim` (order of checks, reasons, other checks) stays as it is.

### 2. Numbers from evidence data (`_data_numbers`)
- A key containing `probability` (e.g. `precipitation_probability`, `pop`) with a numeric value is a
  `percent` (values `<= 1.0` are NOT rescaled here; NWS gives 0-100).
- For a **string** value under a key with no unit family: extract numbers with `_extract_numbers` and keep
  those that carry a unit (non-empty unit). Today they are dropped. Do not do this for the keys `raw`,
  `url`, `id`, `icao`, `scenario`, `rule`, or for keys already skipped.
- Unit-less numbers from free text are still ignored (a year, a count, "57" in "low around 57" unless
  `_extract_numbers` gives it a unit).
- If `_extract_numbers` does not recognise `mph` gusts like "gusts as high as 26 mph" or percentages like
  "is 80%", extend it minimally so both are extracted with units `mph` and `percent`.

### 3. Tests
Add to `tests/test_verifier.py` (and cases to `eval/verifier_cases.json` where that file's format fits):
- The four example claims of cause 1, on a store with an `elevated`/`high` route summary: all `supported`.
- Trip-level claims still caught: "LGA to BOS looks fine tonight." with risk `low`, citing the same clear
  FAA status, on a store whose LGA-BOS summary is `elevated` -> `contradicted: downplays rule-engine risk`.
  A claim citing the engine summary with risk `low` -> still `contradicted`.
- FAA floor: a claim citing an FAA `GDP` evidence with risk `low` -> `contradicted: downplays cited FAA
  program`; with `moderate` -> supported; `ground_stop` with `moderate` -> contradicted, with `high` ->
  supported; `closure` with `affects_commercial: false` and risk `low` -> not contradicted by the floor.
- Numbers: with the forecast evidence above, "gusts up to 26 mph" and "80% chance of precipitation" are
  `supported`; "gusts up to 46 mph" and "30% chance of precipitation" are still not supported.
- Existing tests or cases that encode the old `same_airport` behaviour (a single-airport, route-less,
  low-risk claim contradicted by a route summary) must be **listed in your final message** with the old
  and new expectation. Change them only if they fall under the redefinition above; do not weaken any
  other test.

## Constraints
- Only `travel_agent/verification/checks.py`, `tests/test_verifier.py`, `eval/verifier_cases.json`.
  Do not touch the orchestrator, agents, guardrails, hitl, or `eval/ablation.py`.
- No new dependencies. Do not spawn sub-agents. Do not commit.

## Acceptance
- New tests fail on the current code and pass after the change.
- `.venv/bin/python -m pytest -q` fully green.
- Final message lists every pre-existing test or case whose expectation changed.
