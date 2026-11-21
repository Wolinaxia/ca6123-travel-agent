# Evaluation (M1)

Historical disruption cases (with negative controls) + guardrail test set.
Each run writes a scored report here: `latest.json` (machine-readable) and
`latest.md` (table + every failure).

```bash
.venv/bin/python -m eval.run_eval          # rules only
.venv/bin/python -m eval.run_eval --llm    # rules + Gemini classifier
```

`--llm` uses the classifier only when `GOOGLE_API_KEY` is present; otherwise it
prints "classifier unavailable, rules only". Layer-1 and layer-1+2 numbers are
reported separately. The verifier set covers mixed trusted/untrusted evidence,
failed tool calls, non-summary and uncited-summary risk downplay, unit
conversion (`1 h 23 min`, `83-minute`), numbers that only appear in ids or
coordinates, common-word codes and substring codes. Failures are listed, never
hidden.
