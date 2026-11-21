# Evaluation (M1)

Historical disruption cases (with negative controls) + guardrail test set.
Each run writes a scored report here: `latest.json` (machine-readable) and
`latest.md` (table + every failure).

```bash
.venv/bin/python -m eval.run_eval          # rules only
.venv/bin/python -m eval.run_eval --llm    # rules + Gemini classifier
```

`--llm` uses the classifier only when `GOOGLE_API_KEY` is present; otherwise it
prints "classifier unavailable, rules only". It reports how many classifier
calls were answered; if any call did not answer, layer-1+2 numbers are labelled
NOT VALID and the run exits non-zero when more than 10% of calls failed.
Layer-1 and layer-1+2 numbers are reported separately. The verifier set covers mixed trusted/untrusted evidence,
failed tool calls, non-summary and uncited-summary risk downplay, unit
conversion (`1 h 23 min`, `83-minute`), numbers that only appear in ids or
coordinates, common-word codes and substring codes. Failures are listed, never
hidden.

## Held-out set

`heldout_cases.jsonl` (43 cases) was written by a reviewer and **never shown to whoever tuned the detectors**, so it measures how well the guardrails generalise; the dev set above was written alongside the rules and scores 100% by construction.

Run `python -m eval.run_heldout` (rules only) or `python -m eval.run_heldout --llm` (rules + Gemini).

Results on 2026-11-21 (Gemini `gemini-3.5-flash-lite`):

| Measure | First rules version | Final rules only | Rules + Gemini |
|---|---|---|---|
| Injections blocked | 5/12 | 8/12 | **12/12** |
| Benign requests wrongly blocked | 2/10 | 0/10 | **0/10** |
| Harmful topics blocked | 3/4 | 4/4 | 4/4 |
| Allowed topics wrongly blocked | 1/3 | 0/3 | 0/3 |
| PII redacted / over-redacted | 6/6, 0/8 | 6/6, 0/8 | 6/6, 0/8 |

Now that it is public it is no longer held out; new detector changes need a fresh held-out set.
