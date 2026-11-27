# Fix round 1 for docs/NUMERIC_CONFIDENCE_FIX.md

Only work inside this repository directory; do not read `.env`; no network. The first round is in the
working tree (uncommitted). Change only `travel_agent/agents/specialists.py` and `tests/test_orchestrator.py`.

## Findings
1. `_confidence_level`: `float(value)` raises `OverflowError` for a huge int (e.g. an integer with 400
   digits), and the call is outside `parse_agent_result`'s `try`, so the pipeline crashes. Any numeric value
   that cannot be converted, or is NaN / infinite / negative, must give `"low"` without raising.
2. A non-null, non-string `risk` (e.g. `2` or `["high"]`) is now silently replaced by `None`. Keep the claim
   and `risk=None`, but set the result's `needs_human` to `True` in that case, so a lost risk signal still
   escalates to the human. A numeric or odd-typed `confidence` does NOT set `needs_human`. Unknown risk
   *strings* (e.g. `"severe"`) keep today's behaviour (None, no escalation).

## Tests (tests/test_orchestrator.py)
- A 400-digit integer confidence, `float("nan")`, `float("inf")`, `-1`, `101` -> `"low"`, claim kept, no error.
  (Pass Python objects in a dict to `parse_agent_result` where JSON cannot express the value.)
- `risk: 2` and `risk: ["high"]` -> claim kept, `risk is None`, result `needs_human is True`, no `error`.
- `risk: "severe"` -> `risk is None` and `needs_human` stays as given (False).
- Strengthen `test_numeric_confidence_does_not_discard_sibling_claims`: also assert the mapped
  confidences and that there is no `error` key.

## Acceptance
`.venv/bin/python -m pytest -q` fully green. No other files, no new dependencies, no sub-agents, do not commit.
