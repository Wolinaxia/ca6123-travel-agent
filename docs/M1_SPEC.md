# M1 spec: Verifier, Guardrails, Observability, Evaluation

Work ONLY inside this repository; do not read/write outside it. Do not spawn sub-agents.
Dependencies allowed: what is already in `.venv` (httpx, pydantic, pytest, google-adk 2.15). No others.
Do not modify `travel_agent/core/**`, `travel_agent/tools/**`, or `travel_agent/schemas.py`.
Run `.venv/bin/python -m pytest -q` at the end; everything must pass with no network and **no Gemini API key**.
Code is graded on clarity: module docstrings explaining the design, docstrings on public functions, comments explain *why*.

## Context

- Contract: `travel_agent/schemas.py`, `docs/INTERFACES.md`. Tools: `travel_agent/tools/live.py` (`ALL_TOOLS`), mock tools in `tools/mock.py`.
- Tools return `ToolResult` dicts: `{"source","ok","evidence":[Evidence...],"error"}`. Evidence ids are globally unique. News evidence has `data["untrusted"]=True` and the raw headline in `data["text"]`. Engine summary evidence (`assess_trip_risk`) has id `assess:...:summary` and `data["level"]` in `low|elevated|high|severe`.
- `Claim.risk` uses `low|moderate|high`. Map for comparison: `low→low`, `moderate→elevated`, `high→high or severe`.
- ADK callbacks — use these **positional** signatures so they work on ADK 1.x (Colab) and 2.x. Only use `ctx.state` (dict-like) and `tool.name` on context/tool objects; nothing version-specific:
  - `before_model_callback(ctx, llm_request) -> LlmResponse | None`
  - `after_model_callback(ctx, llm_response) -> LlmResponse | None`
  - `before_tool_callback(tool, args, ctx) -> dict | None` (returning a dict skips the tool and uses it as the result)
  - `after_tool_callback(tool, args, ctx, tool_response) -> dict | None` (returning a dict replaces the result)
  - `LlmRequest.contents` is a list of `google.genai.types.Content` (role, parts[].text). `LlmResponse.content` is a `Content`.
  - Build refusal responses as `LlmResponse(content=types.Content(role="model", parts=[types.Part(text=...)]))`.

## Part A — Guardrails (`travel_agent/guardrails/`)

1. `pii.py` — `redact_pii(text) -> (redacted_text, findings: list[dict])`. Detect and replace with typed placeholders (`[EMAIL]`, `[PHONE]`, `[CARD]`, `[NRIC]`, `[PASSPORT]`):
   - emails; phone numbers (international `+` form, US `(555) 123-4567`/`555-123-4567`, Singapore 8-digit starting 6/8/9); payment cards 13–19 digits with spaces/dashes **validated by Luhn** (invalid Luhn is not a card); Singapore NRIC/FIN `[STFGM]\d{7}[A-Z]`; passport-like only when preceded by the word "passport".
   - Must NOT redact: flight numbers (`UA 857`, `DL1234`), dates, times, airport codes, prices, distances.
2. `injection.py` — `detect_injection(text) -> dict(score: float 0–1, flagged: bool, reasons: list[str])`. Heuristic, documented patterns: instruction override ("ignore/disregard (all/previous/above) instructions"), role hijack ("you are now", "act as", "developer mode", "DAN"), system-prompt extraction ("reveal/print your system prompt/instructions"), fake role tags (`SYSTEM:`, `<|im_start|>`, `### Instruction`), tool/approval forgery ("the user has approved", "approval: granted", "call the booking tool"), encoded payloads (long base64 blobs). `flagged` when score ≥ 0.5. Benign travel text with words like "ignore the delay", "system outage at the airport", "instructions for check-in" must NOT be flagged.
3. `topics.py` — `check_topic(text) -> dict(allowed: bool, category: str|None, reason)`. Block: weapons/explosives, evading security screening or customs, illegal activity (fake documents, smuggling), self-harm. Everything else allowed (out-of-scope but harmless questions are handled by the orchestrator, not blocked here). Keep the lists small and readable.
4. `sanitize.py` — `sanitize_tool_result(result: dict) -> (result, events)`: for evidence with `data["untrusted"]` true, run `detect_injection` on `data["text"]`; if flagged, replace `data["text"]` with `"[removed: possible prompt injection]"`, set `data["injection_removed"]=True`. Also run `redact_pii` on `data["text"]`. Never mutate the input dict (deep copy).
5. `callbacks.py` — ADK callbacks (exported, documented):
   - `input_guard(ctx, llm_request)`: take the **latest user** text part; redact PII in place; if `detect_injection` flags it or `check_topic` blocks it → return a refusal `LlmResponse` (polite, says why in one sentence, no details of the detector) and log a guardrail event. Otherwise None.
   - `output_guard(ctx, llm_response)`: redact PII in model text; if the text claims a booking/rebooking is done ("booked", "confirmed your", "rebooked you") and `ctx.state.get("approved_actions")` is empty → replace with a message saying the action needs human approval first. Return modified response or None.
   - `tool_gate(tool, args, ctx)`: for tools whose name contains `book` (M4's mock booking tool), require an approval record in `ctx.state["approved_actions"]` (list of dicts with `"tool"` and `"args"`, matching tool name and args) → else return `{"ok": False, "error": "blocked: human approval required", "evidence": [], "source": "guardrail"}`.
   - `record_and_sanitize(tool, args, ctx, tool_response)`: if the response looks like a ToolResult, sanitize it (A4) and add all evidence to the evidence store (Part B1). Return the sanitized dict only if it changed.
   - All callbacks must never raise (wrap, log, and fail safe: input/output guard fail **closed** with a generic refusal; record_and_sanitize fails open but logs).
6. Document the shared state keys in a new section of `docs/INTERFACES.md`: `evidence_store`, `approved_actions`, `trace_run_id`. (This is the only doc you edit besides the README files below.)

## Part B — Verifier (`travel_agent/agents/verifier.py` + `travel_agent/verification/`)

1. `verification/store.py` — evidence store kept in `ctx.state["evidence_store"]` as a JSON-serialisable dict `{evidence_id: {"evidence": {...}, "tool_ok": bool, "source": str}}`. Functions: `add_tool_result(state, result)`, `get(state, id)`, `all_ids(state)`.
2. `verification/checks.py` — `verify_claim(claim: dict, store: dict) -> Verdict-like dict {claim, status, reason, checks: list}` and `verify_claims(claims, store) -> dict(verdicts, needs_human: bool, summary)`. Deterministic checks, in order, each recorded in `checks`:
   1. **Citation present** — no `evidence_ids` → `unsupported` ("no evidence cited").
   2. **Citations exist** — any id not in store → `unsupported` ("cites evidence that no tool returned") — this catches fabricated citations.
   3. **Not untrusted-only** — if every cited item is untrusted news → `unsupported` ("supported only by untrusted news text").
   4. **Numbers match** — numbers with units in the claim (`min`, `minutes`, `h`, `hours`, `%`, `kt`, `mph`, `km`, `°F`, `°C`) must match a number in a cited item's observation or (nested) data values, tolerance max(1, 5%). Unit conversions not required (`1 h 23 min` in evidence: treat `83 min` as match is optional; document what you support). Mismatch → `contradicted`.
   5. **Places match** — 3-letter uppercase airport codes in the claim must appear in a cited item's subject/observation/data (incl. `other_metro_airports`). Missing → `unsupported` ("mentions <code> but no cited evidence is about it").
   6. **Risk consistent** — if the claim has `risk` and any cited evidence is an engine summary, compare with the mapping above. Claim lower than the rule engine → `contradicted` ("downplays rule-engine risk"); claim higher → `supported` but add a check note ("more cautious than rule engine").
   Otherwise `supported`. `needs_human` = any verdict not `supported`. `summary` = short human-readable text listing problems (for the HITL step).
   Also accept `Claim` dataclass instances (convert with `dataclasses.asdict`).
3. `agents/verifier.py`:
   - `verify_claims_tool(claims: list[dict], tool_context) -> dict` — ADK function tool that reads the store from `tool_context.state` and returns `verify_claims(...)`. Docstring written for the LLM (claims format with an example).
   - `build_verifier_agent(model: str = "gemini-2.5-flash-lite") -> LlmAgent` — an LlmAgent named `verifier` with that tool, `after_tool_callback=record_and_sanitize` not needed here; instruction: receive specialists' claims, call the tool exactly once, report verdicts, never change a verdict, and if `needs_human` say clearly what the human must check. Constructing the agent must not need an API key.
4. Do not call any LLM in checks. (A semantic LLM judge may be added later; leave a clearly marked extension point, not an implementation.)

## Part C — Observability (`travel_agent/observability/trace.py`)

- `Tracer` writing JSON Lines to `logs/<run_id>.jsonl` (`logs/` is git-ignored; directory configurable for tests). Event types: `tool_call` (agent, tool, args with PII redacted, latency_ms, ok, evidence_ids, error), `guardrail` (kind, decision, reasons), `verdicts` (counts by status, needs_human), `model_call` (agent, latency_ms if measurable).
- `trace_callbacks()` returns ready-made `before_tool`/`after_tool`/`before_model`/`after_model` callbacks that time and log; keep tool-call timing in a dict keyed by `(run_id, tool.name, stable json of args)`.
- A helper to compose multiple callbacks of the same kind into one (ADK 1.x accepts a single callable): `chain_callbacks(*fns)` — calls in order, first non-None return wins.
- `summarize_run(path) -> dict`: counts per tool, failure rate, mean latency, guardrail blocks, verdict counts. Used in the video.
- Guardrail callbacks (Part A) and verifier tool log through the active tracer if one is set (`set_active_tracer(tracer)`), no-op otherwise.

## Part D — Evaluation (`eval/`)

1. `eval/guardrail_cases.jsonl` — at least 50 labelled cases, each `{"id","text","expect": "block"|"allow","category"}` covering: direct injection (≥8), obfuscated injection (≥4: casing, spacing, base64), injection hidden in a "news headline" (≥5, checked via `sanitize_tool_result`, mark `"channel":"tool"`), PII (≥10, incl. valid and Luhn-invalid card numbers, SG phone, NRIC, email), blocked topics (≥6), and **benign travel requests that must pass (≥15)**, including tricky ones (flight numbers, "ignore the delay", "security line wait at JFK", prices, booking references like `ABC123`).
   For PII cases use `"expect":"redact"` plus `"must_redact": ["..."]` / `"must_keep": ["..."]`.
2. `eval/verifier_cases.json` — at least 20 labelled claims evaluated against evidence produced by running `assess_trip_risk("New York","San Francisco","2026-09-23","evening")` and `get_faa_airport_status("SFO")`/`("JFK")` in the replay scenario `sfo_gdp_20260923` (offline). Cover each check: supported (≥6), no citation, fabricated id, untrusted-only, wrong number (e.g. "avg 90 min"), wrong airport (claim about OAK citing SFO evidence), downplayed risk, more-cautious risk.
3. `eval/run_eval.py` — offline runner: `python -m eval.run_eval` prints a table and writes `eval/results/latest.json` + `eval/results/latest.md`:
   - guardrails: block recall (attacks blocked), false-positive rate (benign blocked), PII redaction recall + over-redaction count, per category; list every failing case.
   - verifier: accuracy and per-status confusion matrix; list failures.
   Do not tune the detectors to the eval set by special-casing ids/texts; if a case fails, either improve the general rule or leave it failing and listed. Report the final numbers honestly in your final message.
4. Add `eval/results/` to `.gitignore` except a committed `latest.md` is fine — actually: ignore `eval/results/*.json`, keep `latest.md`.

## Part E — Tests (`tests/test_guardrails.py`, `tests/test_verifier.py`, `tests/test_observability.py`)

- Unit tests for every function above, including the must-NOT cases (flight numbers not redacted, benign "ignore the delay" not flagged).
- Callbacks tested with tiny fake objects: a fake ctx with a `state` dict, a fake tool with `.name`, real `LlmRequest`/`LlmResponse`/`types.Content` objects from ADK/genai.
- `tool_gate` blocks without approval and allows with a matching approval.
- `record_and_sanitize` puts evidence into `state["evidence_store"]` and strips an injected headline.
- Verifier: one test per check; `needs_human` logic; accepts dataclass `Claim`.
- `build_verifier_agent()` constructs without an API key.
- Tracer writes valid JSONL; `summarize_run` numbers correct; `chain_callbacks` order semantics.
- `eval/run_eval.py` runs in a test (smoke) without network.

## Part F — Docs

- `travel_agent/guardrails/README.md`, `travel_agent/verification/README.md`: what each layer does, how to wire the callbacks into an `LlmAgent` (short code example for teammates), state keys.
- Update `requirements.txt` only if needed (google-adk already listed).

## Final report

Per part: done / not done, files, test count, and the eval numbers from `python -m eval.run_eval` (including failures). No review claims.
