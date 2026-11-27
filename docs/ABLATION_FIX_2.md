# Fix round 2 for the ablation script (found on the real collection)

Only work inside this repository; do not read `.env`; no network; do not run `dry-run`/`collect`; do not
run `python -m eval.run_eval`; do not run `python -m eval.ablation score` or `sheet` against
`eval/results/` (real labelled data lives there; use `tmp_path` in tests). Change only `eval/ablation.py`
and `tests/test_ablation.py`.

## Findings
1. **`normal_vs_program` never applies on real data (0 applicable, should be about 48 claims).**
   A real store entry is `{"evidence": {..., "source": "faa_nas", "subject": "LGA", "data": {"program":
   "GDP", ...}}, "tool_ok": true, "source": "deadline_planner"}`. The entry's top-level `source` is the
   *tool that stored it*, not the evidence source. `mutate_normal_vs_program` reads
   `record.get("source") or evidence.get("source")`, gets `deadline_planner`, and skips. Use the evidence's
   own `source` only. Check every other place in `eval/ablation.py` that reads `source` from a store entry
   for the same mistake and fix it the same way.
2. **Labeller is not recorded.** Add `--labeller TEXT` to `score` (default `"not recorded"`). Put it in
   `ablation.json` (`labeller`) and in `ablation.md`: one line under the title
   (`Labels: <text>`) and replace the limitation "One labeller." with
   `One labeller (<text>); Arm A's "wrong" means wrong according to that labeller.`
3. **Silent zero.** When an in-scope fault has 0 applicable base claims, `ablation.md` shows `n/a`. Keep
   that, and add a line under the Arm B table listing in-scope faults with 0 applicable claims
   ("No applicable base claims for: ..."), so a mutation that never fires is visible.
4. **Why claims were held.** Add to the report (json + md) a table "Held claims by Verifier reason":
   for claims held with Verifier ON, group by `(status, reason)` with numbers normalised so that reasons
   differing only in a number group together (replace digits runs in the reason with `N`), and show for
   each group how many of the held claims were labelled correct and how many wrong. Sort by count.

## Tests (offline, tmp_path)
- A store entry shaped exactly like the real one above (top-level `source` = a tool name, evidence
  `source` = `faa_nas`, truthy `data.program`): `normal_vs_program` applies and produces
  `"LGA is operating normally with no FAA delays."`. Must fail on the current code.
- `--labeller` appears in json and md; default text when omitted.
- The "No applicable base claims for" line appears when a fault has none.
- The held-by-reason table groups `states 26 mph but ...` and `states 25 mph but ...` together and splits
  correct/wrong counts.

## Acceptance
`.venv/bin/python -m pytest -q` fully green. No new dependencies, no sub-agents, do not commit.
