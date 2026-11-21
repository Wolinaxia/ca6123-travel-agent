# M1 fix round 1 (review findings + independent test results)

Work ONLY inside this repository; do not read/write outside it. Do not spawn sub-agents.
Allowed deps: what is in `.venv` (httpx, pydantic, pytest, google-adk 2.15, google-genai). No new ones.
May modify: `travel_agent/guardrails/**`, `travel_agent/verification/**`, `travel_agent/agents/verifier.py`, `travel_agent/observability/**`, `eval/**`, `tests/**`, the READMEs, `docs/INTERFACES.md` state-key section, and `travel_agent/tools/live.py` ONLY for item G2.
Do not change `travel_agent/core/**` or `travel_agent/schemas.py`.
Every fix needs a test that fails before the fix. All tests must pass offline with no API key.

## G1 (blocker) — callback signatures: ADK passes KEYWORD arguments
My earlier spec was wrong. ADK (1.x and 2.x) invokes callbacks with keywords and only falls back to positional if binding fails:
- `before_model_callback(callback_context, llm_request)`
- `after_model_callback(callback_context, llm_response)`
- `before_tool_callback(tool, args, tool_context)`
- `after_tool_callback(tool, args, tool_context, tool_response)`
Rename the parameters of every callback (guards, record_and_sanitize, tracer callbacks) to exactly these names. Rewrite `chain_callbacks` so it forwards both positional and keyword calls correctly (simplest: accept `*args, **kwargs` and pass them through unchanged), and so that **if a guard-position callback raises, the chain fails closed**: for model/tool *before* hooks return a refusal/blocked result instead of None. Async callbacks: if a callback returns an awaitable, the chain must be async-aware (or document and reject async callbacks with a clear error).
Tests must invoke callbacks the way ADK does: use `google.adk.utils._callback_pipeline._invoke_callback` if importable, AND an end-to-end test (G10).

## G2 — untrusted text must not survive in any field
- `sanitize_tool_result`: for evidence with `data["untrusted"]`, sanitize **every string** in the evidence recursively (observation, subject, data values, nested lists/dicts), not only `data["text"]`.
- `travel_agent/tools/live.py` (`assess_trip_risk`): for TomTom/MTA engine items, stop duplicating the raw operator text into `data["detail"]`/`data["title"]`/`observation`; keep it only in `data["text"]`.
- `record_and_sanitize`: if sanitizing raises, fail **closed** for untrusted content (drop untrusted evidence items, keep trusted ones, log). Non-ToolResult responses: log a `guardrail` event "unrecognised tool response shape" and pass through.

## G3 — input guard: whole conversation, every call
ADK rebuilds the request from session history on every call with shallow copies, so redaction of one request does not persist.
- On every call, scan **all** `role="user"` text parts in `llm_request.contents` (all turns, all parts), redact PII in each one in place, and run injection/topic checks on each.
- Latest user turn flagged → return the refusal (as now).
- An **earlier** user part flagged → replace that part's text in the request with `"[removed: earlier message blocked by guardrail]"` and continue (don't refuse forever).
- Text split across multiple parts of one message: check the concatenation too.

## G4 — output guard: booking claims and PII
- Replace the single regex with a small, documented detector: completion verbs (booked, rebooked, reserved, confirmed, issued, purchased, ticketed, changed your flight, "you're all set") about the user's trip, with negation handling ("not booked", "haven't booked", "not confirmed yet") and adjective use ("flights are fully booked") NOT counted as claims.
- Allowed only if `state["completed_actions"]` contains a successful booking-type tool result (written by `record_and_sanitize` when a gated tool returns `ok=True`). A non-empty `approved_actions` list is no longer enough.
- Redact PII in `function_call.args` of model output too.

## G5 — tool gate: deny-by-default for side effects
- Gated if the tool name matches any of: `book|reserve|purchase|pay|ticket|cancel|change_flight|hold` (case-insensitive), or is listed in `GATED_TOOLS`. Document.
- Approvals are **single-use**: a matching entry in `approved_actions` is removed when the call is allowed.
- Add `assert_guarded(agent)` helper: raises with a clear message if an `LlmAgent` lacks `input_guard`, `output_guard`, `tool_gate` or `record_and_sanitize` in its callbacks (works with single callables, lists, and chains). Teammates call it in their tests.
- `build_verifier_agent` registers input/output guards.

## G6 — verifier checks: use only trusted, relevant evidence
1. Evidence from `tool_ok=False` results cannot support anything (treated like a missing id).
2. Numbers and places are taken **only from trusted evidence** (untrusted items still count as "cited" for check 3, never as support).
3. **Unit-aware numbers**: extract `(value, unit-family)` from the claim (families: minutes [min/minute/minutes/h/hour; "1 h 23 min" → 83 min], percent, knots, mph, km, temperature). Support hyphenated forms (`90-minute`) and number words one–twenty plus tens. From evidence take numbers only from `observation` and from `data` values whose key implies a unit (`*_min`, `*minutes*`, `*_pct`, `*rate*` (fraction → ×100 as percent), `*_kt`/`gust*`, `*_km`, `*_mph`, `temp*`). Ignore ids, urls, timestamps, `rule`, coordinates, `source_status`. Match within the same family, tolerance `max(0.5, 5%)`. A number with no same-family candidate in evidence → `unsupported` (not `contradicted`); a same-family candidate exists but none matches → `contradicted`.
4. **Places**: word-boundary match (not substring). Only treat a 3-letter uppercase token as an airport code if it exists in the airport reference data AND is not in a stoplist of common abbreviations (USA, PDT, PST, EDT, EST, CDT, CST, MDT, MST, ETA, TSA, SAT, SUN, HOT, AND, THE…). Also accept lowercase/city-name mentions only if you can map them reliably via the airport data; otherwise ignore.
5. **Risk**: if the claim has `risk`, compare with (a) the highest `data.level` among cited engine findings and (b) every engine summary in the store for the same route (`assess:<ROUTE>:...:summary`), even if not cited. Lower than either → `contradicted` ("downplays rule-engine risk"). Higher → supported with note.
6. Persist the result: `verify_claims_tool` writes `tool_context.state["last_verification"]`; document that the HITL step must read the verdict from state, not from the verifier's prose.
7. Empty evidence store while claims cite ids → summary says "evidence store is empty — is record_and_sanitize registered as after_tool_callback?".

## G7 — tracer
- Redact `error` strings and use the **sanitized** tool response.
- Timing key: prefer `tool_context.function_call_id` when present; otherwise a per-key FIFO list. Protect with a lock.
- Unknown latency → `null`, excluded from means in `summarize_run`.

## G8 — injection detection: two layers
1. **Rules (layer 1)** generalise better: broaden each category with synonyms and paraphrases (override/forget/disregard/new instructions/from now on…, role change, prompt extraction incl. "repeat/print/output the text above", fake delimiters and closing tags followed by role words, forged approvals/confirmations, encoded payloads). Fix false positives: "act as" and "you are now" must require a role/persona noun or instruction context ("act as fast as you can", "you are now my favourite helper" are benign). Write **at least 20 new paraphrased attack cases and 10 new tricky benign cases yourself** in `eval/guardrail_cases.jsonl` (dev set). Do not special-case case texts.
2. **Gemini classifier (layer 2)** in `travel_agent/guardrails/llm_classifier.py`: `classify_injection(text) -> dict(available: bool, injection: bool|None, reason)`, using `google.genai` with model from env `GUARDRAIL_MODEL` (default `gemini-2.5-flash-lite`), temperature 0, JSON output, the text inside clear delimiters with an instruction that it is data, timeout ~5 s, in-process cache by sha256. No `GOOGLE_API_KEY` → `available=False`, no call. Errors → `available=False`, logged.
   - `detect_injection(text, use_llm: bool | None = None)`: rules first; if rules don't flag and the classifier is enabled (`use_llm` True, or env `GUARDRAIL_LLM=1` when None), ask the classifier; flagged if it says injection. Result includes `layer: "rules"|"llm"|None`.
   - Used for the latest user turn in `input_guard` and for untrusted text in `sanitize_tool_result` (one call per tool result, batching the texts).
   - Tests mock the genai client (no network): available/unavailable, cache hit, error path, rules-first short-circuit.
3. **Topics**: asking what is allowed (e.g. carrying an item in checked luggage, customs rules) must be allowed; block only intent to evade, conceal, smuggle, forge, or harm. Add dev cases for both sides.

## G9 — eval
- `eval/run_eval.py` reports guardrails with `--llm` flag (uses classifier if a key is present; otherwise prints "classifier unavailable, rules only") and reports layer-1 and layer-1+2 numbers separately.
- Add verifier dev cases for every G6 item (mixed trusted+untrusted, ok=False evidence, non-summary downplay, uncited summary downplay, hyphenated/converted numbers, numbers only in ids/coordinates, common-word codes, substring codes).
- Report honestly; list failures.

## G10 — end-to-end ADK test (no network, no key)
Write `tests/test_adk_wiring.py` using a fake model: subclass `google.adk.models.BaseLlm` (implement `generate_content_async` yielding scripted `LlmResponse`s, and record each `LlmRequest` it receives), plus `google.adk.runners.Runner` / `InMemoryRunner` and an `LlmAgent` wired with the guards via `chain_callbacks` and as lists. Cover:
1. Injection prompt → refusal, fake model never called.
2. Fake model issues a function call to a `book_flight` tool with no approval → tool not executed, blocked result. With a matching approval → executes once; a second identical call is blocked (single-use).
3. A tool returning a ToolResult with an injected untrusted headline → the next model request contains the marker, not the injection; evidence lands in `state["evidence_store"]`.
4. PII in turn 1 → the request the fake model receives on turn 2 contains no raw PII.
If some ADK API differs from what's described here, adapt to the installed 2.15 API and note it in the final report.

## Final report
Per item G1–G10: done / not done, files, tests added, and eval numbers (rules-only). No review claims.
