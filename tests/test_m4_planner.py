"""M4: deadline planner ("Will I make it?" + Plan B) on recorded scenarios."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from travel_agent.memory import cases
from travel_agent.memory.preferences import PreferenceStore
from travel_agent.tools import replay
from travel_agent.tools.deadline_planner import plan_trip_options
from travel_agent.verification.checks import verify_claims
from travel_agent.verification.store import add_tool_result


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TDA_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("TDA_CASE_EMBEDDINGS", "0")  # offline, deterministic keyword search
    cases._INDEX.invalidate()
    yield
    replay.use_demo_scenario(None)


def _plan(origin, destination, deadline, earliest="", user="planner_user"):
    ctx = SimpleNamespace(state={"user_id": user})
    result = plan_trip_options(origin, destination, deadline, earliest, tool_context=ctx)
    assert result["ok"], result["error"]
    return result, {item["id"]: item for item in result["evidence"]}


def test_sfo_ground_delay_prefers_oakland_and_builds_plan_b():
    replay.use_demo_scenario("sfo_gdp_20260923")
    _, by_id = _plan("New York", "San Francisco", "2026-09-23 21:30")
    top = by_id["plan:flight:EWR-OAK"]["data"]
    assert top["rank"] == 1 and top["verdict"] == "comfortable"
    jfk_sfo = by_id["plan:flight:JFK-SFO"]["data"]
    assert jfk_sfo["expected_delay_min"] == 34  # the recorded SFO GDP
    assert jfk_sfo["verdict"] == "tight"
    contingency = by_id["plan:contingency"]["data"]
    assert contingency["plan_a"] == "plan:flight:EWR-OAK"
    assert contingency["plan_b"] == "plan:flight:JFK-SFO"  # shares no airport with Plan A
    assert "ground stop" in contingency["switch_trigger"]
    assert by_id["plan:data_gaps"]["data"]["count"] >= 1  # drive is not in this snapshot


def test_nyc_dc_rail_wins_and_plan_b_uses_a_different_mode():
    replay.use_demo_scenario("nyc_dc_20260924_0716Z")
    _, by_id = _plan("New York", "Washington", "2026-09-24 13:00", "2026-09-24 08:00")
    contingency = by_id["plan:contingency"]["data"]
    assert contingency["plan_a"].startswith("plan:rail:")
    assert not contingency["plan_b"].startswith("plan:rail:")


def test_avoided_airport_is_excluded_and_explained():
    replay.use_demo_scenario("sfo_gdp_20260923")
    PreferenceStore("avoid_user").set("avoid_airports", ["OAK"])
    _, by_id = _plan("New York", "San Francisco", "2026-09-23 21:30", user="avoid_user")
    assert not any("OAK" in key for key in by_id if key.startswith("plan:flight:"))
    assert "plan:excluded:airport:OAK" in by_id
    assert "pref:avoid_user:avoid_airports" in by_id


def test_drive_limit_preference_excludes_long_drive():
    replay.use_demo_scenario("nyc_dc_20260924_0716Z")
    PreferenceStore("drive_user").set("max_drive_hours", 2)
    _, by_id = _plan("New York", "Washington", "2026-09-24 13:00", "2026-09-24 08:00", user="drive_user")
    assert not any(key.startswith("plan:drive:") for key in by_id)
    assert any(key.startswith("plan:excluded:drive:") for key in by_id)


def test_impossible_deadline_escalates():
    replay.use_demo_scenario("sfo_gdp_20260923")
    _, by_id = _plan("New York", "San Francisco", "2026-09-23 17:00")
    contingency = by_id["plan:contingency"]["data"]
    assert contingency["plan_a"] is None and contingency["needs_human"] is True


def test_deadline_before_departure_is_an_error_not_a_guess():
    replay.use_demo_scenario("sfo_gdp_20260923")
    result = plan_trip_options("New York", "San Francisco", "2026-09-22 10:00")
    assert result["ok"] is False and result["evidence"] == []


def test_planner_numbers_are_checkable_by_the_verifier():
    replay.use_demo_scenario("sfo_gdp_20260923")
    result, by_id = _plan("New York", "San Francisco", "2026-09-23 21:30")
    state: dict = {}
    add_tool_result(state, result)
    good = {"text": "Flight EWR to OAK arrives with 133 min to spare, about 10.1 h door to door.",
            "evidence_ids": ["plan:flight:EWR-OAK"], "confidence": "high", "risk": "low"}
    wrong = {"text": "Flight EWR to OAK arrives with 200 min to spare.",
             "evidence_ids": ["plan:flight:EWR-OAK"], "confidence": "high", "risk": "low"}
    downplay = {"text": "Flight JFK to SFO is fine.", "evidence_ids": ["plan:flight:JFK-SFO"],
                "confidence": "high", "risk": "low"}
    verdicts = verify_claims([good, wrong, downplay], state["evidence_store"])["verdicts"]
    assert [item["status"] for item in verdicts] == ["supported", "contradicted", "contradicted"]


def test_avoided_delay_is_stated_so_true_comparisons_verify_and_false_ones_do_not():
    replay.use_demo_scenario("sfo_gdp_20260923")
    state: dict = {"user_id": "avoid"}
    plan = plan_trip_options("New York", "San Francisco", "2026-09-23 21:30", tool_context=SimpleNamespace(state=state))
    oak = next(item for item in plan["evidence"] if item["id"] == "plan:flight:EWR-OAK")
    assert oak["data"]["avoids_delays_at"] == {"SFO": 34}
    add_tool_result(state, plan)
    true_claim = {"text": "OAK avoids the 34 min SFO ground delay.", "evidence_ids": ["plan:flight:EWR-OAK"], "risk": "low"}
    false_claim = {"text": "OAK avoids the 90 min SFO ground delay.", "evidence_ids": ["plan:flight:EWR-OAK"], "risk": "low"}
    verdicts = verify_claims([true_claim, false_claim], state["evidence_store"])["verdicts"]
    assert [v["status"] for v in verdicts] == ["supported", "contradicted"]
