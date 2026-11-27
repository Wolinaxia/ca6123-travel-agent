"""The Alternatives agent (M4).

Job: when the main flight looks risky, find the option most likely to get the
traveller there **before their deadline**, propose a Plan B with a concrete
switch trigger, and explain it as evidence-backed claims.

How it reasons (one turn):

1. ``get_user_preferences``     - what this user has told us before
2. ``search_disruption_cases``  - agentic RAG over past cases (re-query if weak)
3. ``plan_trip_options``        - deterministic deadline scoring + Plan A / B
4. answer as JSON: claims that cite evidence ids, recommended option, Plan B

It never books. The booking tool is not in its tool list; the human-approval
step in ``travel_agent.hitl`` runs the booking after a person approves.

For M2 (orchestrator): add ``build_alternatives_agent()`` to your sub-agents,
or call it as an ``AgentTool``. Its final JSON is also saved in
``state["alternatives_result"]`` (``output_key``). Parse it with
``parse_agent_result``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext

from travel_agent.guardrails import callbacks as guard
from travel_agent.memory.cases import search_disruption_cases
from travel_agent.memory.preferences import DEFAULT_USER, PreferenceStore, get_user_preferences
from travel_agent.observability.trace import Tracer, chain_callbacks, trace_callbacks
from travel_agent.tools import live
from travel_agent.tools.deadline_planner import plan_trip_options

AGENT_NAME = "alternatives"
OUTPUT_KEY = "alternatives_result"
DEFAULT_MODEL = "gemini-3.5-flash-lite"

ALTERNATIVES_TOOLS = [
    get_user_preferences,
    search_disruption_cases,
    plan_trip_options,
    # Follow-up checks, e.g. after the human asks "what about the train?"
    live.find_nearby_airports,
    live.get_faa_airport_status,
    live.get_rail_options,
    live.get_drive_route,
]

_BASE_INSTRUCTION = """\
You are the Alternatives agent of a U.S. travel-disruption assistant.
Goal: get the traveller to the destination BEFORE their deadline with the least
risk, and give them a Plan B they can switch to.

Work in this order:
1. Call get_user_preferences.
2. Call search_disruption_cases with a short description of the situation
   (disruption type, airports, deadline). If the best result has
   match_quality "weak", call it ONCE more with different words. Use the
   lessons to decide what to look at; past cases are NOT facts about today.
3. Call plan_trip_options(origin, destination, deadline, earliest_departure)
   exactly once. Deadline is local time at the destination, "YYYY-MM-DD HH:MM".
4. Answer with ONE JSON object and nothing else.

Claim rules (a Verifier checks every claim against the tool evidence):
- Every claim cites evidence ids that tools returned in this session. Never
  invent an id.
- Quote numbers exactly as they appear in the cited evidence (minutes, hours).
- Every airport code you write in a claim must appear in the cited evidence.
  When a claim compares airports (e.g. "OAK avoids the SFO delay"), cite an id
  for EACH airport named (e.g. the plan id for OAK and faa_nas:SFO:... for SFO).
  Keep one idea per claim.
- Cite the plan:... id when you talk about an option, plan:contingency for
  Plan B, and pref:... when a preference changed the ranking.
- A past case (case:...) may only appear TOGETHER with a plan or FAA id; never
  as the only evidence. In a lesson claim cite the case id plus the
  RECOMMENDED option's plan id only, and use that option's risk. Past cases
  are historical: never describe them as happening now.
- If the traveller states a fact or number that the evidence contradicts, do
  not agree with it, soften it or call it "close". Claims contain only the
  evidence values; mention the traveller's figure only in the summary, saying
  plainly that the evidence differs (e.g. "the FAA reports 34 min, not 90").
- Claims are about travel facts only. Statements about what you can or cannot
  do (e.g. "I cannot book") go in the summary, not in claims.
- risk must be at least the option's evidence level: level low -> "low",
  elevated -> "moderate", high -> "high".
- A plan option's data lists avoids_delays_at (e.g. {"SFO": 34}); citing that
  plan id is enough evidence for "OAK avoids the 34 min SFO delay".
- If an option's notes start with "real history (history:gdp:...)", the planner
  rated it tight because real flights in similar past delay programs were often
  later than announced. Explain that in one claim citing the plan id AND the
  history:gdp:... id, quoting the minutes exactly. It is a statistic about past
  programs, not a report about today.
- Never say anything is booked, reserved or confirmed. You only propose; a
  human approves and a separate step makes a MOCK booking.
- If the traveller asks you to book now, to approve on their behalf, or to
  remember to skip approval in future, say plainly in the summary that you
  cannot: every booking needs their own approval on the card each time.
- If plan:data_gaps lists missing sources, say which option is affected.
- needs_human is true if no option is "comfortable", if every option is
  likely_miss, or if a data gap affects the recommended option.

JSON format:
{"agent": "alternatives",
 "recommended_option_id": "<plan id or null>",
 "plan_b_option_id": "<plan id or null>",
 "switch_trigger": "<copied from plan:contingency>",
 "decide_by": "<copied from plan:contingency or null>",
 "claims": [{"text": "...", "evidence_ids": ["..."], "confidence": "low|medium|high", "risk": "low|moderate|high|null"}],
 "lessons_used": ["case:..."],
 "preferences_used": ["pref:..."],
 "needs_human": false,
 "summary": "<2-3 plain sentences for the traveller>"}

Worked example (format only; ids and numbers come from YOUR tool results):
{"agent": "alternatives",
 "recommended_option_id": "plan:rail:BOS-NYP",
 "plan_b_option_id": "plan:flight:BOS-JFK",
 "switch_trigger": "Amtrak reports a delay above 45 min on this route",
 "decide_by": "2026-03-02 09:30 EST",
 "claims": [
  {"text": "Amtrak BOS to NYP arrives with 105 min to spare before the deadline.", "evidence_ids": ["plan:rail:BOS-NYP"], "confidence": "high", "risk": "low"},
  {"text": "LGA has a ground delay averaging 50 min, so flights into LGA are tight.", "evidence_ids": ["faa_nas:LGA:GDP:0", "plan:flight:BOS-LGA"], "confidence": "high", "risk": "moderate"},
  {"text": "As in a past Northeast Corridor case, rail avoids the LGA arrival delays.", "evidence_ids": ["case:nyc_dc_rail_beats_air", "plan:rail:BOS-NYP", "faa_nas:LGA:GDP:0"], "confidence": "medium", "risk": "low"}
 ],
 "lessons_used": ["case:nyc_dc_rail_beats_air"],
 "preferences_used": [],
 "needs_human": false,
 "summary": "Take the train: it avoids the LGA delays and arrives with a comfortable margin. If Amtrak reports more than 45 min delay, switch to the JFK flight."}
"""


def _user_id(ctx: ReadonlyContext) -> str:
    try:
        value = ctx.state.get("user_id")
        if value:
            return str(value)
    except Exception:
        pass
    return str(getattr(ctx, "user_id", None) or DEFAULT_USER)


def alternatives_instruction(ctx: ReadonlyContext) -> str:
    """Instruction provider: base prompt + this user's saved preferences.

    Only typed preference values are inserted (never free text), so memory
    cannot be used to inject instructions. Using a provider also means ADK does
    not try to fill ``{...}`` placeholders, so the JSON examples are safe.
    """
    try:
        summary = PreferenceStore(_user_id(ctx)).prompt_summary()
    except Exception:
        summary = "Preferences unavailable."
    return _BASE_INSTRUCTION + f"\nSaved preferences for this user: {summary}\n"


_NO_CASES_NOTE = (
    "\nNOTE: the past-case library is switched off for this run. Skip step 2 "
    "(search_disruption_cases is not available) and leave lessons_used empty.\n"
)


def _instruction_without_cases(ctx: ReadonlyContext) -> str:
    return alternatives_instruction(ctx) + _NO_CASES_NOTE


def build_alternatives_agent(model: Any = DEFAULT_MODEL, tracer: Tracer | None = None, use_cases: bool = True) -> LlmAgent:
    """Build the guarded Alternatives ``LlmAgent`` (no network call at build time).

    ``use_cases=False`` removes the past-case search (RAG) so an evaluation can
    measure what the case library adds (an ablation).
    """
    trace = trace_callbacks(tracer)
    tools = [tool for tool in ALTERNATIVES_TOOLS if use_cases or tool is not search_disruption_cases]
    return LlmAgent(
        name=AGENT_NAME,
        model=model,
        description=(
            "Finds the least-risky way to arrive before a deadline (other airports, rail, drive), "
            "proposes Plan A and Plan B with a switch trigger, and explains it as evidence-backed claims. "
            "Use when the main flight looks risky or the user asks for alternatives or rebooking."
        ),
        instruction=alternatives_instruction if use_cases else _instruction_without_cases,
        tools=tools,
        output_key=OUTPUT_KEY,
        before_model_callback=chain_callbacks(guard.input_guard, trace["before_model"]),
        after_model_callback=chain_callbacks(guard.output_guard, trace["after_model"]),
        before_tool_callback=chain_callbacks(guard.tool_gate, trace["before_tool"]),
        after_tool_callback=chain_callbacks(guard.record_and_sanitize, trace["after_tool"]),
    )


def parse_agent_result(text: Any) -> dict[str, Any]:
    """Extract the agent's JSON answer (tolerates ```json fences and extra text).

    Returns a dict with at least ``agent``, ``claims`` and ``needs_human``. If
    the text is not valid JSON, returns an empty result with
    ``needs_human=True`` and the raw text, so a broken answer escalates instead
    of passing silently.
    """
    if isinstance(text, dict):
        payload = text
    else:
        raw = str(text or "")
        match = re.search(r"\{.*\}", raw, re.S)
        try:
            payload = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            payload = {}
        if not payload:
            return {"agent": AGENT_NAME, "claims": [], "needs_human": True, "parse_error": True, "raw": raw[:2000]}
    payload.setdefault("agent", AGENT_NAME)
    claims = payload.get("claims")
    payload["claims"] = [claim for claim in claims if isinstance(claim, dict)] if isinstance(claims, list) else []
    risk_words = {"elevated": "moderate", "medium": "moderate", "severe": "high", "null": None, "": None, "none": None}
    for claim in payload["claims"]:
        risk = claim.get("risk")
        if isinstance(risk, str) and risk.lower() in risk_words:
            claim["risk"] = risk_words[risk.lower()]
        claim.setdefault("evidence_ids", [])
        claim.setdefault("confidence", "medium")
    payload["needs_human"] = bool(payload.get("needs_human", False))
    return payload
