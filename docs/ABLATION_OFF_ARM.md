# Make `verifier_enabled=False` reach the approval card

## Context
- Repo: ca6123-travel-agent, Python, pytest. Only work inside this repository; do not read files outside it.
- `travel_agent/agents/orchestrator.py`: `TravelOrchestrator.verifier_enabled` (default True) is the switch for
  M1's ablation study. When False, `_finish` skips `verify_claims` and `verification` stays `None`.
- But `_finish` then calls `build_hitl_proposal(...)`, which calls `hitl.build_proposal(view, m4_result)` in
  `travel_agent/hitl/approval.py`. `build_proposal` always runs `verify_claims` + `_m4_checks`, so with the
  Verifier "off" the approval card still carries per-claim verdicts and "Claim N is unsupported/contradicted"
  warnings (which also block a plain "approve"). The OFF arm of the ablation is therefore not clean.
- Tests to read first: `tests/test_m4_hitl.py`, `tests/test_m2_m4_integration.py`, `tests/test_orchestrator.py`
  (the existing `verifier_enabled=False` test near line 157).

## Task
1. `hitl.build_proposal(state, result, now=None)`: add a keyword-only parameter `verify: bool = True`.
   - `verify=True`: behaviour exactly as today.
   - `verify=False`: do not call `verify_claims` or `_m4_checks`; do not write `state["last_verification"]`;
     `Proposal.verdicts == []`; `Proposal.verification_summary == "verification disabled"`; no
     "Claim N is ..." warnings. All other warnings stay (option not comfortable, data gaps, parse error,
     malformed claims), and `needs_human` is computed from those warnings and `result["needs_human"]` only.
     Everything else (option lookup, `ProposalError` cases, plan B, evidence age, logging) is unchanged.
2. `orchestrator.build_hitl_proposal(...)`: pass `verify=False` when its `verification` argument is `None`
   (i.e. the Verifier is disabled), `verify=True` otherwise. Do not change its return shape.
3. Make sure `render_card_text` and `render_card_html` work with an empty `verdicts` list; the card should
   say that claims were not verified instead of showing a verdict per claim. Keep the wording short.
4. Tests:
   - In `tests/test_m4_hitl.py`: a result with a claim that is contradicted/unsupported today
     (reuse an existing bad result from that file) gives, with `verify=False`: empty `verdicts`, the
     summary above, no "Claim" warning, and `state` has no `last_verification` key; with `verify=True` the
     same input still produces the claim warning (guards the default).
   - In `tests/test_m2_m4_integration.py`: one end-to-end run with `verifier_enabled=False` on the existing
     offline scenario: the answer has `verifier_enabled False`, `last_verification` is `None`, the pending
     proposal has `verdicts == []`, and the card text contains no "Claim 1 is". And the same run with the
     default (enabled) still has non-empty verdicts.

## Constraints
- Default behaviour must not change: every existing test must pass unmodified.
- Do not change `approve`, `execute_approved`, `reject`, `request_change`, the booking tool or the tool gate.
- Do not touch `travel_agent/verification/`, guardrails, eval result files, or notebooks.
- No new dependencies. Do not spawn sub-agents for review. Do not commit.

## Acceptance
- New tests fail on the current code and pass after the change.
- `.venv/bin/python -m pytest -q` is fully green.
- Update the M4/HITL paragraph in `docs/INTERFACES.md` with one sentence about `verify=False`.
