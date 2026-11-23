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
| `trip_goal` | dict (`TripGoal`) | `origin`/`destination` (city), `origin_airport`/`destination_airport` (IATA), `departure_date` (ISO, origin local), `departure_window` (`morning`/`afternoon`/`evening`/`whole_day`), `arrive_by` (ISO datetime with destination offset, or null), `days_ahead` (int, use it to pick live vs forecast sources), `flight_number`, `constraints` (list of user preferences), `missing` (slots still needed), `notes`, `classified_by` (`llm`/`rules`). |
| `plan` | `list[{"id", "agent", "goal", "depends_on", "parallel_group", "condition"}]` | Ordered sub-tasks. `agent` is one of `flight_airport`, `weather`, `alternatives`, `verifier`, `synthesizer`, `human_approval`, `clarify`. Tasks sharing a `parallel_group` have no dependency on each other. A non-null `condition` means the step may be skipped (e.g. alternatives for `assess_trip` only when some claim has moderate/high risk). |

Specialist `AgentResult.agent` values must use the same names as `plan[*].agent`.

### Specialist output and answer keys (M2)

Each specialist is an ADK agent named `flight_airport`, `weather` or `alternatives`
with `output_key="result_<name>"`. Its final message is JSON:

```json
{"agent": "weather", "needs_human": false,
 "claims": [{"text": "...", "evidence_ids": ["nws:..."], "confidence": "high", "risk": "low"}],
 "unavailable": ["get_weather_alerts"]}
```

Tools that returned `ok=false` go in `unavailable`, **not** in a claim: a claim
with no evidence is unsupported and would escalate for no reason. Plug a real
agent in with `build_orchestrator(specialists={"weather": my_agent})`; it must
carry the four guardrail callbacks (see `agents.specialists.guarded_callbacks`).

| Key | Written by | Meaning |
|---|---|---|
| `result_<name>` | each specialist (`output_key`) | Raw final message of that specialist. |
| `specialist_results` | orchestrator | Parsed `AgentResult` dicts per specialist (claims tagged with `agent`). |
| `final_answer` | orchestrator | `{"text", "overall_risk", "shown_claims", "held_back", "unavailable", "needs_human", "awaiting_approval", "verifier_enabled"}`. For `rebook`, `awaiting_approval=True` means the HITL step (M4) should ask the human; nothing has been booked. |

`agents.orchestrator.run_pipeline(request, scenario=None, verifier_enabled=True)`
runs one request end to end and returns these keys plus `evidence_store` and
`trace_run_id`. `verifier_enabled=False` is the ablation "Verifier OFF" arm.
