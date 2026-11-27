# PII filter: do not redact decimal fractions as card numbers

## Context
- Repo: ca6123-travel-agent, Python, pytest. Only work inside this repository; do not read files outside it.
- File: `travel_agent/guardrails/pii.py` (`_CARD_CANDIDATE_RE`, `_luhn_valid`, `redact_pii`)
- Tests: `tests/test_guardrails.py` (existing card tests around lines 56-66)
- Bug: decimal fractions in tool output (e.g. BTS rates serialised as
  `{"rate": 0.8474337369372327}`) are redacted to `0.[CARD]` when the 16
  fraction digits happen to pass the Luhn check (about 10% of random floats).

## Task
Make the card detector ignore digit runs that are the fractional part of a decimal number.
1. Change `_CARD_CANDIDATE_RE` so a candidate is rejected when it is
   immediately preceded by `<digit>.` (fixed-width lookbehind `(?<!\d\.)`,
   in addition to the existing lookbehind).
2. Add tests in `tests/test_guardrails.py`:
   - `{"rate": 0.8474337369372327}`, `0.763774618976614`, `0.762280082457942`
     are returned unchanged with no findings.
   - A property-style test: 5,000 seeded `random.random()` values dumped with
     `json.dumps` are never changed by `redact_pii`.
   - Real cards are still redacted: `4111 1111 1111 1111`, `5555-5555-5555-4444`,
     and `Card no.4111111111111111` (a dot after a word, not after a digit).
3. Update the module docstring's "payment cards" bullet with one clause about decimals.

## Constraints
- Do not change `redact_pii`'s signature, the placeholders or the other detectors.
- Do not touch `travel_agent/tools/live.py` or any file other than the two named above.
- No new dependencies. Do not spawn sub-agents for review. Do not commit.

## Acceptance
- The new tests fail on the current code and pass after the change.
- `.venv/bin/python -m pytest -q` is fully green.
