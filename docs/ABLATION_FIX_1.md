# Fix round 1 for docs/ABLATION_SPEC.md

Only work inside this repository; do not read `.env`; no network; do not run `dry-run`/`collect`;
do not run `python -m eval.run_eval`. The first round is in the working tree (uncommitted). Fix the
review findings below in `eval/ablation.py` and `tests/test_ablation.py` only.

## Findings to fix
1. **HIGH: Arm B verifies outside the scenario clock.** `compute_arm_b` calls `verify(...)` after
   `prepare_record`'s `active_scenario` block has exited, so `verify_claims` uses the wall clock, every
   evidence with a past `valid_to` counts as expired, controls and all mutations come back unsupported and
   are scored as "caught". Fix: every verification of a record (controls and mutations) must run inside
   `active_scenario(record["scenario"])`. After computing, **fail hard** (raise / non-zero exit with a clear
   message) if `controls.supported != controls.total`, since base claims are supported by construction.
2. **Headline shift misses `unknown`.** When the rule engine fails and the only risky claim is held, ON
   `overall_risk` is `unknown` and OFF is e.g. `high`; no shift is recorded. Rank `unknown` below `low`.
3. **Labels joined by `claim_id` only.** (a) `score` must compare each CSV row's `claim` text with the raw
   file's claim text for that id and fail with a clear message listing mismatching ids (stale labels after
   a re-collect). (b) Read the CSV with `utf-8-sig` so a BOM does not hide the `claim_id` column.
4. **`fabricated_id` must not exist in the store.** If the mutated id exists, keep appending `9` to the last
   segment until it does not.
5. **Turns without claims.** A record whose turn ended in clarification / out-of-scope / a block / an
   `error` has no claims. Report such records separately (`turns_without_claims`, with ids) and exclude
   them from the escalation-rate denominator. Do not default a missing intent to `assess_trip`; treat the
   record as a turn without claims.

## Do NOT change
- The mutation definitions, including the `normal_vs_program` sentence. A fault the Verifier misses is
  reported as missed.

## Tests to add (offline)
- Arm B under a real clock: a record with `scenario` set to a recorded scenario name
  (`nyc_rain_20261122_1708z`) and an evidence item whose `valid_to` is after that scenario's `now` but
  before today's date. Controls must be supported, and `no_citation` must be caught while the unmodified
  control is not. Use a fake rule engine (monkeypatch) so no network is needed. This test must fail on the
  round-1 code.
- The hard failure when a control is not supported.
- Headline shift with ON `unknown` vs OFF `high`.
- Stale-label detection (text mismatch) and BOM-prefixed CSV loading.
- A multi-agent record (claims from flight_airport, weather and alternatives) through `prepare_record`,
  asserting that the per-claim statuses line up with the claim ids and that "shown" derived from verdict
  status equals `on["shown_claims"]`.
- The sheet contains no Verifier output in its *values*: build a record whose `last_verification` and
  `final_answer` contain a distinctive marker string and assert the marker is absent from the CSV.
- `normal_vs_program`: assert `applicable` only (do not assert caught either way).

## Acceptance
`.venv/bin/python -m pytest -q` fully green; existing non-ablation tests unmodified. Do not commit.
Do not spawn sub-agents.
