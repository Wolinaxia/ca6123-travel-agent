# Labelling sheet: show the evidence data, not only the one-line observation

## Context
- Repo: ca6123-travel-agent. Only work inside this repository; do not read `.env`; no network; do not run
  `python -m eval.ablation dry-run` or `collect`; do not run `python -m eval.run_eval`.
- `eval/ablation.py`, `sheet` sub-command (`build_sheet_rows`, `_observation`): the `evidence` column shows
  `id: observation` for each cited evidence. The observation is a short summary, e.g.
  `faa_nas:LGA:GDP:2: Ground Delay Program at LGA: avg 148 min (low visibility)`, while a claim may say
  "max delay of 362 minutes". That number is only in the evidence's `data`
  (`{"program": "GDP", "avg_delay_min": 148, "max_delay_min": 362, "reason": "low visibility", ...}`),
  so the labeller cannot check the claim from the sheet.
- A store entry looks like `{"evidence": {"id", "source", "subject", "observation", "fetched_at", "data",
  "valid_from", "valid_to", "url"}, "tool_ok", "source"}`.

## Task
1. In the `evidence` column, for each cited id write
   `<id>: <observation> || data: <compact JSON of data> || valid: <valid_from>..<valid_to>`.
   - `data`: `json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))` after removing
     the keys `replay` and `scenario` (recording bookkeeping, not evidence). Omit the `data:` part when the
     remaining dict is empty. If the JSON is longer than 1200 characters, cut it at 1200 and append `…`.
   - Omit the `valid:` part when both `valid_from` and `valid_to` are missing.
   - `MISSING` behaviour for unknown ids is unchanged. Evidence items stay joined with ` | `.
2. Add a column `cited_ids` (between `confidence` and `evidence`) with the cited ids joined by `, `
   (empty when the claim cites nothing), so an uncited claim is visibly uncited.
3. The stale-label check in `score` compares only the `claim` text; make sure the new/changed columns do
   not break `score`, `load_labels`, or the refuse-to-overwrite check.
4. The sheet must still contain no Verifier output (verdict, status, reason, held/shown, needs_human).
5. Update tests in `tests/test_ablation.py`: exact column list; a row whose evidence has `data` with a
   number absent from the observation shows that number; `replay`/`scenario` keys are not shown; long data
   is cut; missing validity is omitted; uncited claim has empty `cited_ids` and empty `evidence`.

## Constraints
Only `eval/ablation.py` and `tests/test_ablation.py`. No new dependencies. No sub-agents. Do not commit.
Acceptance: `.venv/bin/python -m pytest -q` fully green.
