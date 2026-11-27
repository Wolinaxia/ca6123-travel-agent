# Fix round 1 for docs/ABLATION_OFF_ARM.md

Only work inside this repository. The first round is in the working tree (uncommitted). Fix these review findings.

## Findings
1. `travel_agent/hitl/approval.py`, `render_card_text` and `render_card_html`: the new line
   "Claims were not verified." is shown whenever `proposal.verdicts` is empty. With the default `verify=True`
   and a result with no usable claims (`"claims": []`, or all claims malformed), verdicts is also empty, so the
   card wrongly says claims were not verified (summary is "no claims to verify"). Before the change this case
   rendered the header with no rows. Fix: show the line only when verification was disabled. Use one shared
   module constant for the summary text (`VERIFICATION_DISABLED = "verification disabled"`) and branch on
   `proposal.verification_summary == VERIFICATION_DISABLED`. Do not add a field to `Proposal`.
2. `render_card_text`: the line "Agent summary (not fact-checked, read the checked claims below): ..." is
   wrong when verification is disabled (there are no checked claims). When disabled, use
   "Agent summary (not fact-checked): ..." instead. The default wording must stay exactly as it is.

## Tests
- `tests/test_m4_hitl.py`: with the default `verify=True` and `{**GOOD_RESULT, "claims": []}`, neither the
  text card nor the HTML card contains "Claims were not verified.", and the summary is "no claims to verify".
- Extend the existing verify=False test: the text card contains "Agent summary (not fact-checked):" and does
  not contain "read the checked claims below" (GOOD_RESULT has a summary; if not, set one in the test).
- In `tests/test_m2_m4_integration.py::test_verifier_off_leaves_the_approval_card_unchecked` remove the
  vacuous assertion `"Claim 1 is" not in answer["text"]`.

## Constraints
Same as the original spec: default behaviour unchanged, no other files, no new dependencies, no sub-agents,
do not commit. Acceptance: `.venv/bin/python -m pytest -q` fully green.
