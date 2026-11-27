# Verifier ablation script (`eval/ablation.py`)

## Context
- Repo: ca6123-travel-agent, Python 3, pytest, Google ADK. Only work inside this repository; do not read
  files outside it and do not read `.env`.
- Read first: `docs/VERIFIER_ABLATION_PLAN.md` (the design; its "Updates, 27 Nov" section wins where it
  differs), `eval/ablation_requests.json` (the 18 fixed requests; do not change it),
  `travel_agent/agents/orchestrator.py` (`_finish`, `assemble_answer`, `assess_with_rules`,
  `combine_verifications`, `Conversation`, `RESULT_KEYS`), `travel_agent/verification/checks.py`
  (`verify_claims`), `travel_agent/tools/replay.py` (`use_demo_scenario`, `demo_now`),
  `eval/run_intent_eval.py` (style of the existing eval scripts).
- Session state does NOT keep the rule engine's ToolResult (`rules["result"]`). Recompute it with
  `assess_with_rules(goal, live.assess_trip_risk)` while the scenario is active; it is deterministic.
- `verify_claims` uses `replay.demo_now()` as its clock. Scoring must therefore activate the record's
  scenario with `replay.use_demo_scenario(name)` (and reset it to `None` afterwards, in `finally`).

## Task

### 1. Small refactor in `travel_agent/agents/orchestrator.py`
Extract the verification block of `_finish` into a pure module-level function and call it from `_finish`:

```python
def verify_results(results, store, rules_result) -> dict:
    """flight_airport + weather claims vs store + rule-engine evidence; alternatives claims vs store only."""
```
It returns exactly what `combine_verifications(...)` returns today. Behaviour of the pipeline must not
change; all existing tests must pass unmodified.

### 2. `eval/ablation.py` with four sub-commands (`python -m eval.ablation <cmd>`)
- `dry-run`: run the first request of each scenario through `Conversation(scenario=..., verifier_enabled=True)`
  and print, per request: intent, number of claims per agent, size of `evidence_store`, number of
  `unavailable` entries per agent. Needs `GOOGLE_API_KEY` (load it the way the other eval scripts do).
- `collect`: run all 18 requests once each (a fresh `Conversation` per request) and write one JSON line per
  request to `eval/results/ablation_raw.jsonl`: `id`, `scenario`, `expected_intent`, `request`, `model`,
  `git_commit` (short hash from `git rev-parse`), `collected_at`, and the state keys `intent`, `trip_goal`,
  `specialist_results`, `evidence_store`, `last_verification`, `final_answer`. A request that raises is
  written with an `error` field and the run continues. `--limit N` and `--only id1,id2` for partial runs.
- `sheet`: from the raw file, write `eval/results/ablation_labels.csv` with one row per claim and the
  columns `claim_id` (`<request id>:<agent>:<index>`), `scenario`, `request`, `agent`, `claim`, `risk`,
  `confidence`, `evidence` (for each cited id: `id: observation`, joined with ` | `; `MISSING` when the id
  is not in the store), `label` (empty), `note` (empty). It must contain NO verdict, status or reason from
  the Verifier. If the CSV already exists and any `label` is filled, refuse unless `--force`.
- `score`: read the raw file and the labels CSV and write `eval/results/ablation.json`, `ablation.md` and
  `ablation.svg`. Fails with a clear message listing unlabelled claim ids, unless `--allow-unlabelled`
  (then unlabelled claims are excluded and the report says how many).

Allowed labels: `correct`, `wrong_number`, `wrong_place`, `risk_downplayed`, `unsupported`, `stale`,
`other_wrong`. Everything except `correct` counts as wrong.

### 3. Scoring logic (pure functions, unit-tested offline)
For each record, with its scenario active:
- `rules = assess_with_rules(trip_goal, live.assess_trip_risk)`;
  `verification = verify_results(specialist_results, evidence_store, rules.get("result"))`.
- ON answer = `assemble_answer(intent, goal, results, verification, rule_view, None)`;
  OFF answer = `assemble_answer(intent, goal, results, None, rule_view, None)`
  (`rule_view` = `rules` without `"result"`). No approval card in either arm.
- Consistency check: for `assess_trip` records compare the recomputed verdict statuses with the saved
  `last_verification`; report the number of mismatching records (expected 0).

**Arm A (real claims)**, ON vs OFF, using the human labels:
- bad claims shown: wrong claims among shown claims / shown claims;
- catch rate: wrong claims held back / wrong claims;
- false hold rate: correct claims held back / correct claims;
- escalation rate: turns with `needs_human` / turns;
- headline shift: records where ON `overall_risk` is lower than OFF, with whether each held claim there
  was labelled correct.
Break the false-hold and escalation numbers down by scenario (the calm scenario is the negative control).

**Arm B (injected faults)**: base set = claims labelled `correct` AND `supported` by the recomputed
verification (report how many correct claims were excluded for not being supported; that number is Arm
A's false holds). For each base claim and each fault below, make one mutated copy when the mutation
applies, verify it alone with the same store the pipeline would use for that agent (with rule-engine
evidence for flight_airport/weather, without for alternatives), and record caught (status != supported)
or missed. Mutations are defined without looking at verdicts:

| fault | mutation | applies when |
|---|---|---|
| `wrong_number` | first number in the text that is followed by `min`/`minutes`/`%` becomes `n*3+20` | such a number exists |
| `wrong_airport` | first IATA code in the text that appears in the subject/id of a cited evidence is replaced by another airport from {JFK, LGA, EWR, BOS} that appears in neither the text nor the cited evidence | such a code and a replacement exist |
| `fabricated_id` | every evidence id gets its last `:`-segment replaced by `zz9` (must not exist in the store) | claim has evidence ids |
| `no_citation` | `evidence_ids = []` | always |
| `risk_downplayed` | `risk = "low"` | agent is flight_airport/weather, claim risk is moderate/high, rule-engine level is elevated/high/severe |
| `normal_vs_program` | text becomes `"<AIRPORT> is operating normally with no FAA delays."`, ids and risk unchanged | a cited `faa_nas` evidence has a truthy `data.program`; AIRPORT from that evidence |
| `wrong_time_window` (out of scope, expected to get through) | first of `this evening`/`this afternoon`/`tonight`/`today` in the text becomes `tomorrow` | such a phrase exists |

Report per fault: applicable, caught, catch rate; an overall catch rate over the six in-scope faults; the
out-of-scope row separately; and the controls (unmodified base claims: all must be supported, by
construction). List every missed in-scope mutation (claim id, fault, mutated text) in the JSON and the md.

### 4. Outputs
- `ablation.json`: all numbers above plus `git_commit`, model, counts of records/claims, label counts.
- `ablation.md`: headline table ON vs OFF (bad claims shown, escalation rate), Arm A table, Arm B table,
  missed mutations, limitations (single run per request, one labeller, M4 card checks excluded, three
  recorded scenarios on one route). Plain wording, no claims beyond the numbers.
- `ablation.svg`: one self-contained SVG (no external fonts or libraries, written with plain string
  formatting): horizontal bars of catch rate per fault type for Verifier ON, each labelled with
  `caught/applicable`; the out-of-scope fault visually separated. Title and axis labels included.

### 5. Tests: `tests/test_ablation.py` (offline, no network, no model, no key)
Build a tiny synthetic record in the test (hand-written evidence store with an `faa_nas` GDP evidence and
one other evidence, 3-4 claims) and cover:
- each mutation: applies / does not apply as specified, and produces the specified change;
- `fabricated_id` ids are not in the store;
- the sheet has exactly the specified columns and none of: verdict, status, reason;
- `sheet` refuses to overwrite filled labels without `--force`;
- metric functions on a hand-labelled fixture with known answers (bad-claims-shown ON < OFF when a wrong
  claim is held; false hold counted when a correct claim is held);
- `score` fails on unlabelled claims and succeeds with `--allow-unlabelled`;
- the SVG is well-formed XML (`xml.etree.ElementTree.fromstring`);
- `verify_results` gives the same verdicts as calling `verify_claims` the way `_finish` did.

## Constraints
- No new dependencies (no matplotlib, pandas, numpy). Standard library `csv`, `json`, `argparse` only.
- Do NOT run `dry-run` or `collect`, do not call Gemini, do not read `.env`, do not run
  `python -m eval.run_eval` (it overwrites a tracked results file).
- Do not change `travel_agent/verification/`, guardrails, `travel_agent/hitl/`, the requests file, or any
  existing eval result file. In `orchestrator.py` only the refactor in section 1.
- Do not spawn sub-agents. Do not commit.

## Acceptance
- `.venv/bin/python -m pytest -q` fully green, existing tests unmodified.
- `.venv/bin/python -m eval.ablation --help` and each sub-command's `--help` work without a key.
