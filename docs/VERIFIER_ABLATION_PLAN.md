# Verifier ablation plan (M1, light version)

Status: draft, 24 Nov. Runs Fri 27 – Sat 28 Nov. Output: one table and one chart for video
segment 7 (Responsible AI).

## Question
Does the Verifier stop wrong claims from reaching the user, and how many correct claims does it
hold back?

## Design
- **No self-grading.** Whether a claim is wrong is never decided by `verify_claims`.
  - Arm A: I label the real claims myself, without looking at the verdicts.
  - Arm B: we inject the faults, so we already know which claims are wrong.
- **Paired.** Verification is the last step before the answer, and everything before it is the
  same with or without the Verifier. So each request runs once. Both answers are built from the
  same saved claims and evidence store:
  - ON: `assemble_answer(..., verify_claims(claims, store, as_of=demo_now))`
  - OFF: `assemble_answer(..., None)`

  `as_of` must be the scenario's frozen time; otherwise every recorded piece of evidence counts
  as expired.

## Arm A: real claims (small)
- 18 fixed requests over the recorded scenarios:
  - intents: `assess_trip` ×10, `plan_trip` ×5, `rebook` ×3;
  - `calm_baseline_20261121` is the negative control;
  - each request runs once.
- I label every claim myself as one of: `correct`, `wrong_number`, `wrong_place`,
  `risk_downplayed`, `unsupported`, `stale`, or `other_wrong`.
- What this arm measures: how many correct claims get held back (false holds) and how often a
  turn escalates to a human.

## Arm B: injected faults (main evidence)
- Take the Arm A claims labelled `correct` and make one broken copy per fault type below. Each
  copy keeps its real evidence store.
- Keep the unmodified originals as controls; they must stay `supported`.

| Fault | Mutation | Check expected to catch it |
|---|---|---|
| wrong_number | change one number beyond tolerance | `numbers_match` |
| wrong_airport | swap in another airport on the route | `places_match` |
| fabricated_id | well-formed but non-existent evidence id | `citations_exist` |
| no_citation | drop `evidence_ids` | `citation_present` |
| risk_downplayed | lower `risk` below the rule engine | `risk_consistent` |
| normal_vs_program | "operations normal" while the cited FAA record shows a program | `airport_status` |

There is also one out-of-scope row: a wrong time window ("tomorrow" instead of "this evening").
We expect it to get through, and it shows where the Verifier's coverage ends.

## Metrics (ON vs OFF)
- **Bad claims shown:** the share of claims shown to the user that are wrong.
- **Catch rate:** wrong claims held back ÷ all wrong claims.
- **False hold rate:** correct claims held back ÷ all correct claims.
- **Escalation rate:** the share of turns with `needs_human`.
- **Headline shift:** cases where `overall_risk` is lower when the Verifier is ON, i.e. a held
  claim pulled the headline down. For each case, note whether the held claim was correct.

## Steps
1. **Dry run** (26 Nov): one request per scenario.
   - Check that claims parse and `evidence_store` is filled.
   - Check how many tool calls hit the recording. If more than 20% miss (they return
     `ok=false`), reword the request.
2. **Collect** (27 Nov): `python -m eval.ablation collect` saves raw state to
   `eval/results/ablation_raw.jsonl`.
3. **Label** (27 Nov evening): label from a sheet with the verdict column hidden.
4. **Score** (28 Nov): `python -m eval.ablation score` writes `ablation.{json,md,png}`. Then add
   a short section to `eval/README.md`.

## Risks
- **Specialists.** The real M3/M4 specialists may not be on main yet. We run with whatever is
  there and record which version was used.
- **Few real errors.** Arm A may produce very few wrong claims. That is fine: Arm B carries the
  catch-rate result.
- **Missing scenario file.** `nyc_dc_20260924_0619Z` has no `scenario.json`, so it is skipped.

## Expectations (written before running)
- Arm B catch rate ≥ 90%, with 0 false holds on the controls.
- If the false hold rate on real claims comes out above 20%, the video reports it as a cost. We
  do not tune the Verifier on these labels.

## Updates, 27 Nov (before any run)
- **Scenarios.** Only the three November recordings have weather data. Requests are fixed in
  `eval/ablation_requests.json`: 10 on `nyc_rain_20261122_1708z`, 5 on `nyc_rain_20261122_1550z`,
  3 on `calm_baseline_20261121` (negative control). All are New York ↔ Boston, the only airports
  in those recordings.
- **What "ON" means now.** M2 verifies flight/weather claims against the tool evidence plus the
  rule-engine evidence, and alternatives claims against the tool evidence only. The ablation uses
  exactly that function. M4's two extra checks on the approval card are not part of the Verifier
  and are left out of both arms.
- **Chart.** SVG written by the script itself (no plotting dependency), not PNG.
- **Rule-engine evidence** is not saved in the session state; scoring recomputes it from the
  recorded scenario (deterministic).

## Deviation, 28 Nov (written before any label exists)
- **Labeller.** Arm A labels are produced by Gemini (through Antigravity), not by me as planned; I did
  not have the time. The labeller sees the same blind sheet (claim, cited evidence, no Verifier output)
  and the written rules in the hand-off letter (kept with the results). It has no access to the repo or
  to the Verifier's verdicts.
- **What this changes.** The labels are an LLM's judgement, so Arm A's "wrong" is "wrong according to an
  independent LLM reading the cited evidence", not human ground truth. The report must say so. Arm B is
  less affected: its faults are injected, and the labels only select which claims are used as the base.
- **Check, if time allows.** I label a random sample of about 25 claims myself, blind to both the
  Verifier and the LLM labels, and report the agreement.
- **Collection.** The first collection (233 claims) was discarded before labelling because it exposed two
  parsing bugs that removed weather claims and flight citations (fixed in #18 and #20). The run used for
  the results is the second one (260 claims, commit `968e3cc`). Requests and design were not changed.

## Outcome, 28 Nov
- Results and how to read them: `eval/README.md`, section "Verifier ablation (M1)". Files:
  `eval/results/ablation_before_fix.md` and `ablation.md` (same claims and labels, Verifier before and
  after #22), `ablation_raw.jsonl`, `ablation_labels.csv`, `ablation_labelling_rules.md`.
- Against the expectations written above: injected-fault catch rate 92% (expected at least 90%), controls
  all supported. The false hold rate on real claims was 26%, above the 20% line, and is reported as a cost.
- **Deviation from "we do not tune the Verifier on these labels":** the two causes of the false holds
  were fixed in the Verifier (#22) after seeing the result, and the fix was developed against these same
  claims. Both result sets are reported; the second is not a held-out measurement.
- The human sample check listed under "Deviation, 28 Nov" was not done.
- The `normal_vs_program` sentence was not changed, and the Verifier was not extended to match it (0/46).
