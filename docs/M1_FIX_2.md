# M1 fix round 2 (small)

Work ONLY inside this repository. Do not spawn sub-agents. No new deps. Do not change `travel_agent/core/**`, `travel_agent/schemas.py`, `travel_agent/tools/**`.
Each item: fix + a test that fails before the fix. All tests offline, no API key. Do not special-case exact texts; fix the general rule.

## H1 — topics regression
Intent to get a prohibited item (knife, blade, gun, weapon, explosive, lighter fluid…) past airport security/screening/TSA covertly ("sneak", "get … past", "slip … through", "without being detected", "hide … from screening") must be blocked. Asking whether an item is *allowed* (carry-on vs checked rules) must stay allowed. Add ≥4 dev cases on each side to `eval/guardrail_cases.jsonl` (your own wording).

## H2 — output guard: noun-form completion claims
Completion stated with a noun subject and no first-person verb must count as a claim: e.g. "<booking|reservation|rebooking|ticket|purchase|change> (is )?(complete|completed|done|successful|finalised|confirmed)", "booking reference/confirmation number is …" when presented as done. Keep negation handling ("not complete yet", "pending", "once complete"). Add tests both ways.

## H3 — risk downplay with route-less citations
`checks.py` risk check: if the claim has `risk` and cites no engine summary/finding with a route, also compare against every engine summary in the store whose route contains an airport that appears in (a) the claim's place mentions or (b) the `subject` of any cited trusted evidence (e.g. FAA evidence with subject `SFO` → summaries for routes `*-SFO` or `SFO-*`). Lower than any such summary → `contradicted`. Add a verifier eval case and a unit test.

## H4 — tool gate name matching
Match gated keywords against name tokens split on `_`/`-`/camelCase (word match), not raw substrings, so `paypal_lookup` or `display_hold_music` are not gated by accident while `pay_fare`, `hold_fare`, `rebook_flight`, `reserve_seat`, `purchase_ticket`, `change_flight` still are (`rebook` must match `book`: treat `re`-prefixed verbs as the base verb). Tests.

## Final report
Per item done/not done, files, tests; rerun `python -m eval.run_eval` and report numbers.
