"""M4: Alternatives agent wiring, mock booking and the human-approval gate.

No network and no API key: the agent run uses the scripted fake model from
``test_adk_wiring`` and recorded replay scenarios.
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import pytest
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

from tests.test_adk_wiring import ScriptedModel, _call_response, _text_response
from travel_agent import hitl
from travel_agent.agents.alternatives import (
    OUTPUT_KEY,
    alternatives_instruction,
    build_alternatives_agent,
    parse_agent_result,
)
from travel_agent.guardrails import callbacks as guard
from travel_agent.memory import cases
from travel_agent.memory.preferences import PreferenceStore
from travel_agent.tools import replay
from travel_agent.tools.booking import book_trip_option
from travel_agent.tools.deadline_planner import plan_trip_options

GOOD_RESULT = {
    "agent": "alternatives",
    "recommended_option_id": "plan:flight:EWR-OAK",
    "plan_b_option_id": "plan:flight:JFK-SFO",
    "claims": [
        {"text": "Flight EWR to OAK arrives with 133 min to spare.", "evidence_ids": ["plan:flight:EWR-OAK"], "confidence": "high", "risk": "low"},
        {"text": "SFO has a ground delay averaging 34 min.", "evidence_ids": ["faa_nas:SFO:GDP:0"], "confidence": "high", "risk": "moderate"},
    ],
    "needs_human": False,
    "summary": "Fly Newark to Oakland.",
}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TDA_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("TDA_CASE_EMBEDDINGS", "0")  # offline, deterministic keyword search
    cases._INDEX.invalidate()
    replay.use_demo_scenario("sfo_gdp_20260923")
    yield
    replay.use_demo_scenario(None)


def _planned_state() -> dict:
    state: dict = {"user_id": "hitl_user"}
    ctx = SimpleNamespace(state=state)
    result = plan_trip_options("New York", "San Francisco", "2026-09-23 21:30", tool_context=ctx)
    guard.record_and_sanitize(FunctionTool(plan_trip_options), {}, ctx, result)
    return state


# ---------------------------------------------------------------- agent wiring

def test_agent_is_fully_guarded_and_cannot_book():
    agent = build_alternatives_agent()
    guard.assert_guarded(agent)
    names = {getattr(tool, "__name__", getattr(tool, "name", "")) for tool in agent.tools}
    assert "book_trip_option" not in names
    assert not any(guard.is_gated_tool(name) for name in names)


def test_instruction_includes_typed_preferences_only():
    PreferenceStore("pref_user").set("avoid_redeye", True)
    ctx = SimpleNamespace(state={"user_id": "pref_user"})
    text = alternatives_instruction(ctx)
    assert "avoid_redeye=True" in text


def test_parse_agent_result_handles_fences_and_garbage():
    parsed = parse_agent_result("```json\n" + json.dumps({**GOOD_RESULT, "claims": [{"text": "x", "risk": "elevated"}]}) + "\n```")
    assert parsed["recommended_option_id"] == "plan:flight:EWR-OAK"
    assert parsed["claims"][0]["risk"] == "moderate" and parsed["claims"][0]["evidence_ids"] == []
    broken = parse_agent_result("sorry, I could not do it")
    assert broken["needs_human"] is True and broken["parse_error"] is True


def test_scripted_agent_run_records_evidence_and_output():
    model = ScriptedModel(script=[
        _call_response("search_disruption_cases", {"query": "fog ground delay SFO evening"}),
        _call_response("plan_trip_options", {"origin": "New York", "destination": "San Francisco", "deadline": "2026-09-23 21:30"}),
        _text_response(json.dumps(GOOD_RESULT)),
    ])
    agent = build_alternatives_agent(model=model)
    runner = InMemoryRunner(agent=agent, app_name="m4")

    async def go():
        session = await runner.session_service.create_session(app_name="m4", user_id="u", state={"user_id": "u"})
        message = types.Content(role="user", parts=[types.Part(text="NYC to SF by 21:30, SFO delayed")])
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            pass
        return await runner.session_service.get_session(app_name="m4", user_id="u", session_id=session.id)

    session = asyncio.run(go())
    store = session.state["evidence_store"]
    assert "plan:flight:EWR-OAK" in store and "case:sfo_fog_gdp_oak" in store
    assert parse_agent_result(session.state[OUTPUT_KEY])["recommended_option_id"] == "plan:flight:EWR-OAK"
    proposal = hitl.build_proposal(dict(session.state), parse_agent_result(session.state[OUTPUT_KEY]))
    assert proposal.verification_summary == "all 2 claims supported"


# ---------------------------------------------------------------- booking tool

def test_booking_is_blocked_by_tool_gate_without_approval():
    state = _planned_state()
    blocked = guard.tool_gate(FunctionTool(book_trip_option), {"option_id": "plan:flight:EWR-OAK"}, SimpleNamespace(state=state))
    assert blocked is not None and "approval" in blocked["error"]


def test_booking_refuses_unknown_and_likely_miss_options():
    state = _planned_state()
    ctx = SimpleNamespace(state=state)
    assert book_trip_option("plan:flight:XXX-YYY", tool_context=ctx)["ok"] is False
    state["evidence_store"]["plan:flight:EWR-OAK"]["evidence"]["data"]["verdict"] = "likely_miss"
    assert book_trip_option("plan:flight:EWR-OAK", tool_context=ctx)["ok"] is False


# ---------------------------------------------------------------- approval flow

def test_proposal_for_unknown_option_escalates():
    with pytest.raises(hitl.ProposalError):
        hitl.build_proposal(_planned_state(), {**GOOD_RESULT, "recommended_option_id": "plan:flight:FAKE-ONE"})


def test_card_shows_verifier_result_and_writes_last_verification():
    state = _planned_state()
    bad = {**GOOD_RESULT, "claims": GOOD_RESULT["claims"] + [
        {"text": "SFO delays are 90 min.", "evidence_ids": ["faa_nas:SFO:GDP:0"], "confidence": "high", "risk": "moderate"}]}
    proposal = hitl.build_proposal(state, bad)
    assert state["last_verification"]["counts"]["contradicted"] == 1
    card = hitl.render_card_text(proposal)
    assert "[WRONG]" in card and "MOCK" in card
    assert "&#10060;" in hitl.render_card_html(proposal)


def test_verifier_off_leaves_the_card_unchecked_and_does_not_write_last_verification():
    bad = {**GOOD_RESULT, "claims": GOOD_RESULT["claims"] + [
        {"text": "SFO delays are 90 min.", "evidence_ids": ["faa_nas:SFO:GDP:0"], "confidence": "high", "risk": "moderate"}]}
    state = _planned_state()
    proposal = hitl.build_proposal(state, bad, verify=False)
    assert proposal.verdicts == []
    assert proposal.verification_summary == "verification disabled"
    assert not any("Claim" in warning for warning in proposal.warnings)
    assert "last_verification" not in state
    card = hitl.render_card_text(proposal)
    assert "Claims were not verified." in card
    assert "Claims were not verified." in hitl.render_card_html(proposal)
    assert "Agent summary (not fact-checked):" in card
    assert "read the checked claims below" not in card

    enabled = hitl.build_proposal(_planned_state(), bad)
    assert any("Claim" in warning for warning in enabled.warnings)
    assert any(verdict.get("status") != "supported" for verdict in enabled.verdicts)


def test_enabled_verification_with_no_usable_claims_does_not_say_unverified():
    state = _planned_state()
    proposal = hitl.build_proposal(state, {**GOOD_RESULT, "claims": []})
    assert proposal.verdicts == []
    assert proposal.verification_summary == "no claims to verify"
    card = hitl.render_card_text(proposal)
    assert "Claims were not verified." not in card
    assert "Claims were not verified." not in hitl.render_card_html(proposal)
    assert "no claims to verify" in card


def test_warnings_must_be_acknowledged():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)  # drive data gap -> warning
    assert proposal.needs_human
    assert hitl.approve(state, proposal)["ok"] is False
    assert hitl.approve(state, proposal, acknowledge_warnings=True)["ok"] is True


def test_full_happy_path_books_once_and_allows_booking_claim():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)
    hitl.approve(state, proposal, acknowledge_warnings=True)
    first = hitl.execute_approved(state, proposal)
    assert first["ok"] and first["booking"]["data"]["mock"] is True
    assert first["booking"]["data"]["confirmation"].startswith("MOCK-")
    assert state["completed_actions"][0]["tool"] == "book_trip_option"
    assert hitl.execute_approved(state, proposal)["ok"] is False  # single use
    assert first["learned_case"] is not None  # decision saved for future retrieval


def test_expired_approval_is_blocked():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)
    start = replay.demo_now()
    hitl.approve(state, proposal, acknowledge_warnings=True, now=start)
    out = hitl.execute_approved(state, proposal, now=start + timedelta(minutes=hitl.APPROVAL_TTL_MIN + 1))
    assert out["blocked"] and "expired" in out["error"]
    assert "completed_actions" not in state


def test_stale_evidence_is_blocked_even_with_fresh_approval():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)
    later = replay.demo_now() + timedelta(minutes=hitl.MAX_EVIDENCE_AGE_MIN + 5)
    hitl.approve(state, proposal, acknowledge_warnings=True, now=later)
    out = hitl.execute_approved(state, proposal, now=later)
    assert out["blocked"] and out["needs_recheck"]


def test_reject_and_change_feed_memory_with_consent():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)
    prefs = PreferenceStore("hitl_user")
    assert hitl.reject(state, proposal, "I don't want Oakland", prefs)["suggestions"] == []
    change = hitl.request_change(state, proposal, "avoid oakland, give me SFO", prefs)
    assert change["suggestions"][0]["key"] == "avoid_airports"
    assert "plan:flight:EWR-OAK" in change["message_for_agent"]
    assert prefs.get_all() == {}
    events = [item["event"] for item in state["hitl_log"]]
    assert "rejected" in events and "change_requested" in events


def test_explain_uses_evidence_only():
    state = _planned_state()
    proposal = hitl.build_proposal(state, GOOD_RESULT)
    text = hitl.explain(state, proposal)
    assert "133 min" in text and "faa_nas:OAK:status" in text
