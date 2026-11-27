# Interfaces — the shared contract

Read this before writing any agent or tool. The code version is `travel_agent/schemas.py`.

## Why this matters

The Verifier can only check a claim if the claim says **which evidence it is based on**.
So: tools produce `Evidence` with ids; agents produce `Claim`s that cite those ids.
An agent statement with no evidence id is treated as unsupported.

## Tool output: `ToolResult`

Every tool function returns a `ToolResult` (as a dict when called through ADK):

| Field | Type | Meaning |
|---|---|---|
| `source` | str | e.g. `"faa_nas"`, `"nws"`, `"aviation_weather"` |
| `ok` | bool | False if the source failed. **Never invent data on failure.** |
| `evidence` | list[Evidence] | normalized observations |
| `error` | str \| None | short reason when `ok` is False |

### `Evidence`

| Field | Type | Meaning |
|---|---|---|
| `id` | str | unique, e.g. `"faa_nas:SFO:GDP:2026-11-27T18:00Z"` |
| `source` | str | same as ToolResult.source |
| `subject` | str | what it is about: airport code, route, city |
| `observation` | str | short human-readable fact, e.g. `"Ground Delay Program at SFO, avg 34 min"` |
| `data` | dict | raw numbers the Verifier can check against (e.g. `{"avg_delay_min": 34}`) |
| `valid_from` / `valid_to` | ISO datetime \| None | time window where this evidence is meaningful |
| `fetched_at` | ISO datetime | when we fetched it |
| `url` | str \| None | where a human can check it |

## Agent output: `AgentResult`

| Field | Type | Meaning |
|---|---|---|
| `agent` | str | e.g. `"flight_airport"` |
| `claims` | list[Claim] | what the agent concludes |
| `needs_human` | bool | agent itself wants a human decision |

### `Claim`

| Field | Type | Meaning |
|---|---|---|
| `text` | str | the statement, e.g. `"SFO arrivals are delayed ~34 min this evening"` |
| `evidence_ids` | list[str] | **required**; ids of the Evidence this is based on |
| `risk` | `"low"\|"moderate"\|"high"\|None` | optional risk judgement |
| `confidence` | `"low"\|"medium"\|"high"` | how sure, kept separate from risk |

## Verifier output: `Verdict`

| Field | Meaning |
|---|---|
| `claim` | the claim checked |
| `status` | `supported` / `unsupported` / `contradicted` |
| `reason` | short explanation |

Any `unsupported` or `contradicted` claim on the decision path → escalate to the human
before a recommendation is shown as final.

## Mock tools

`travel_agent/tools/mock.py` returns fixed `ToolResult`s with the same shape as the real tools.
Build against these until the real tools are merged. Swapping should need no agent changes.

## Agent state keys (M1)

ADK callbacks and tools share one dict-like `ctx.state`. M1 adds three keys; code in
other milestones must use the same names.

| Key | Type | Owner | Meaning |
|---|---|---|---|
| `evidence_store` | `{evidence_id: {"evidence": Evidence dict, "tool_ok": bool, "source": str}}` | `guardrails.callbacks.record_and_sanitize` | Every evidence item a tool actually returned, keyed by id. The Verifier resolves `Claim.evidence_ids` here, so fake ids are visible as missing keys. Records with `tool_ok=False` cannot support a claim. |
| `approved_actions` | `list[{"tool": str, "args": dict}]` | human-in-the-loop / approval step (M4) | Permission to try a side effect. Tools matching `book|reserve|purchase|pay|ticket|cancel|change_flight|hold` (or listed in `GATED_TOOLS`) are blocked unless an entry matches the tool name **and** the exact call args. Each entry is **single-use**: it is removed when the call is allowed. |
| `completed_actions` | `list[{"tool": str, "args": dict, "ok": True}]` | `record_and_sanitize` | Written when a gated tool actually returns `ok=True`. A booking claim in model text is allowed only if this list holds a successful gated result; `approved_actions` alone is not proof. |
| `last_verification` | `{"verdicts", "needs_human", "summary", "counts"}` | `verify_claims_tool` | The authoritative Verifier output. The HITL step must read the verdict from this state key, never from the Verifier agent's prose. |
| `trace_run_id` | `str` | tracer setup | Id of the active JSONL trace under `logs/<run_id>.jsonl`. Set it when creating the `Tracer` so guardrail/verifier events and tool calls land in the same file. |

Callbacks receive ADK keyword arguments (`callback_context`, `llm_request`,
`llm_response`, `tool`, `args`, `tool_context`, `tool_response`); `chain_callbacks`
forwards both positional and keyword calls and fails closed for guard callbacks.
Use `guardrails.callbacks.assert_guarded(agent)` in tests to prove all four hooks
are wired (single callables, lists and chains are understood).

Guardrail events are appended to `ctx.state["guardrail_events"]` as
`{"kind", "decision", "reasons"}`; this key is for diagnostics only and is not part of
the Verifier contract.

## Orchestrator state keys (M2)

Written by `agents.planner.plan_request_tool` at the start of every user turn.
Specialists read their sub-task from `plan` and the resolved trip from
`trip_goal`; they should not re-parse the user's text.

| Key | Type | Meaning |
|---|---|---|
| `intent` | `"assess_trip" \| "plan_trip" \| "rebook" \| "out_of_scope"` | Routed intent for this turn. |
| `trip_goal` | dict (`TripGoal`) | `origin`/`destination` (city), `origin_airport`/`destination_airport` (IATA), `departure_date` (ISO, origin local), `departure_window` (`morning`/`afternoon`/`evening`/`whole_day`), `arrive_by` (ISO datetime with destination offset, or null), `days_ahead` (int, use it to pick live vs forecast sources), `flight_number`, `departure_options` (NYC airports to compare when the user named the metro rather than one airport, else `[]`), `constraints` (list of user preferences), `missing` (slots still needed), `notes`, `classified_by` (`llm`/`rules`). |
| `plan` | `list[{"id", "agent", "goal", "depends_on", "parallel_group", "condition"}]` | Ordered sub-tasks. `agent` is one of `flight_airport`, `weather`, `alternatives`, `verifier`, `synthesizer`, `human_approval`, `clarify`. Tasks sharing a `parallel_group` have no dependency on each other. A non-null `condition` means the step may be skipped (e.g. alternatives for `assess_trip` only when the rule engine or some claim says moderate/high risk). |
| `pending_clarification` | dict or null | Raw intent slots and missing fields for a short follow-up in the same ADK session. |

Specialist `AgentResult.agent` values must use the same names as `plan[*].agent`.

### Specialist output and answer keys (M2)

Each specialist is an ADK agent named `flight_airport`, `weather` or `alternatives`
with `output_key="result_<name>"`. Its final message is JSON:

```json
{"agent": "weather", "needs_human": false,
 "claims": [{"text": "...", "evidence_ids": ["nws:..."], "confidence": "high", "risk": "low"}],
 "unavailable": ["SFO get_weather_alerts: timeout"]}
```

Tools that returned `ok=false`, and coverage gaps (no data for part of the trip
window), go in `unavailable`, **not** in a claim: a claim with no evidence is
unsupported and would escalate for no reason. Entries are short sentences
(clipped at 160 characters, at most 12 per specialist). Plug a real
agent in with `build_orchestrator(specialists={"weather": my_agent})`; it must
carry the four guardrail callbacks (see `agents.specialists.guarded_callbacks`).

| Key | Written by | Meaning |
|---|---|---|
| `result_<name>` | each specialist (`output_key`) | Raw final message of that specialist. |
| `specialist_results` | orchestrator | Parsed `AgentResult` dicts per specialist (claims tagged with `agent`). |
| `rule_assessment` | orchestrator | Deterministic rule-engine result for the trip (`tools.live.assess_trip_risk`, run in parallel with the specialists): `{"risk", "engine_level", "confidence", "action", "evidence_id"}` or `{"error"}`. `risk` maps the engine's `low/elevated/high/severe` onto the claim scale. The engine evidence is added to the Verifier's view only, not to `evidence_store`. |
| `final_answer` | orchestrator | `{"text", "overall_risk", "rule_risk", "shown_claims", "held_back", "unavailable", "needs_human", "awaiting_approval", "verifier_enabled"}`. `overall_risk` is the higher of the rule engine's level and the shown claims' risks. The analysis flow sets `awaiting_approval=False`. The HITL workflow owns approval for a concrete replacement proposal that it validates for the current trip. |

`agents.orchestrator.run_pipeline(request, scenario=None, verifier_enabled=True)`
runs one request end to end and returns these keys plus `evidence_store` and
`trace_run_id`. `risk_engine=None` skips the rule engine (offline tests). `verifier_enabled=False` is the ablation "Verifier OFF" arm.

## M4: Alternatives agent, mock booking, HITL, memory

**Agent.** `travel_agent.agents.alternatives.build_alternatives_agent(model=..., tracer=None, use_cases=True)` returns a
guarded `LlmAgent` named `alternatives` (all four M1 callbacks; `assert_guarded` passes). It has **no**
booking tool. Its final answer is JSON, also saved in `state["alternatives_result"]`; parse it with
`parse_agent_result(text)`. The output is a superset of `AgentResult`:

| Field | Meaning |
|---|---|
| `agent`, `claims`, `needs_human` | as in `AgentResult` |
| `recommended_option_id` / `plan_b_option_id` | `plan:<mode>:<route>` evidence ids |
| `switch_trigger`, `decide_by` | copied from `plan:contingency` |
| `lessons_used` / `preferences_used` | `case:...` / `pref:...` ids |
| `summary` | 2–3 sentences for the traveller |

**New tools and evidence ids.**

| Tool | Evidence ids | Notes |
|---|---|---|
| `tools.deadline_planner.plan_trip_options(origin, destination, deadline, earliest_departure="")` | `plan:<mode>:<route>`, `plan:contingency`, `plan:excluded:...`, `plan:data_gaps` (+ the FAA/BTS/Amtrak/OSRM evidence it used) | Deterministic. `data` has `door_to_door_min`, `expected_delay_min`, `slack_min`, `delay_tolerance_min`, `verdict`, `level` (low/elevated/high, so the Verifier's risk check applies), `delay_unknown` (FAA data missing: never rated comfortable), `avoids_delays_at` (e.g. `{"SFO": 34}`), `history_margin_min` / `history_id` / `history_downgraded` (learned from real FAA+BTS history: landing at an airport with a ground delay program is "comfortable" only if it survives the 90th-percentile delay (beyond the announced average) real flights had in similar 2025 programs; options landing at a same-metro neighbour cite their own statistic but are not downgraded; evidence `history:gdp:<kind>:<gdp|neighbour>`). Refuses unknown places, unreadable or past deadlines, and origin = destination (< 30 km) |
| `memory.cases.search_disruption_cases(query, top_k=3)` | `case:<id>` | Agentic RAG over 33 real past US disruptions (`provenance: public_report`, with `event_date` and `sources`) plus 2 recorded demo days. Meaning-based (Gemini embeddings, then a Gemini re-rank of the 10 closest that may only reorder real case ids; `TDA_CASE_RERANK=0` turns it off) when a key is available, keyword (BM25) otherwise; `data.retriever` says which. Always `data.untrusted=True`: guidance only, never proof |
| `memory.preferences.get_user_preferences()` | `pref:<user>:<key>` | Reads `state["user_id"]` |
| `tools.booking.book_trip_option(option_id)` | `booking:MOCK-xxxxxx` | Gated by `tool_gate`; only planned, non-`likely_miss` options |

**HITL.** `travel_agent.hitl`: `build_proposal(state, result)` (writes `last_verification`),
`render_card_text/html`, `approve(..., acknowledge_warnings=...)`, `reject`, `request_change`, `explain`,
`execute_approved`. Approval records in `approved_actions` carry extra keys `proposal_id`, `approved_at`,
`expires_at` (15 min); `execute_approved` also blocks if the option's evidence is older than 30 min.
Decisions are logged to `state["hitl_log"]` and the active tracer.
`build_proposal` adds two M4 checks after M1's Verifier: a claim that something is booked/confirmed
without a cited `booking:` id is `unsupported`, and a claim citing only an active FAA program with
risk below its floor (GDP → moderate, ground stop/closure → high) is `contradicted`.
`request_change` returns `ok=False` and `message_for_agent=None` when the text looks like an instruction to the AI
(`memory.text_safety.looks_like_instruction`: M1's filter after NFKC normalisation, plus common non-English phrasings).
Learned cases store only tool-generated text (never the model's summary). Preference files are per user; ids that
need sanitising get a hash suffix so `alice.smith` and `alice_smith` never share memory.
Red-team tests: `tests/test_m4_danger.py`. Evals: `python -m eval.run_m4_eval` (offline) and
`python -m eval.run_m4_live_eval` (real Gemini).

**Memory files** live in `TDA_MEMORY_DIR` (default `.memory/`, git-ignored).
