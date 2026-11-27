"""M4 Learn stage: lessons checked on real history change the plan (code, not the model).

The table used here is a small FAKE table so the tests do not depend on the real
numbers; the real table (travel_agent/data/processed/gdp_history.json) is checked
for shape and provenance at the end.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from travel_agent.memory import cases
from travel_agent.tools import deadline_planner, gdp_history, replay
from travel_agent.tools.deadline_planner import plan_trip_options
from travel_agent.verification.checks import verify_claims
from travel_agent.verification.store import add_tool_result


def _table(gdp_q90: int, neighbour_q90: int) -> dict:
    row = {"programs": 50, "gdp_flights": 1000, "gdp_extra_over_announced_q50": 0, "gdp_extra_over_announced_q90": gdp_q90,
           "gdp_cancel_or_divert_rate": 0.03, "neighbour_flights": 1000, "neighbour_delay_q50": 0,
           "neighbour_delay_q90": neighbour_q90, "neighbour_cancel_or_divert_rate": 0.01}
    return {"learned_from": "test table", "min_flights": 150, "groups": {"low_visibility": row, "all": row},
            "checks_case_lessons": ["sfo_fog_gdp_oak"]}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("TDA_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("TDA_CASE_EMBEDDINGS", "0")
    cases._INDEX.invalidate()
    yield
    replay.use_demo_scenario(None)


def _plan(monkeypatch, table):
    monkeypatch.setattr(gdp_history, "load_table", lambda: table)
    replay.use_demo_scenario("sfo_gdp_20260923")
    ctx = SimpleNamespace(state={"user_id": "history_user"})
    result = plan_trip_options("New York", "San Francisco", "2026-09-23 21:30", tool_context=ctx)
    assert result["ok"], result["error"]
    return result, {item["id"]: item for item in result["evidence"]}


@pytest.mark.parametrize("text,group", [
    ("WEATHER / THUNDERSTORMS", "thunderstorms"), ("low ceilings", "low_visibility"), ("WEATHER / FOG", "low_visibility"),
    ("WEATHER / WIND", "wind"), ("WEATHER / SNOW-ICE", "snow_ice"), ("STAFFING / STAFFING", "other"), (None, "other"),
])
def test_faa_reasons_map_to_program_kinds(text, group):
    assert gdp_history.condition_group(text) == group


def test_history_rule_only_downgrades_comfortable_options_with_too_little_slack():
    history = {"id": "history:gdp:low_visibility:gdp", "role": "gdp", "minutes": 100}
    option = {"verdict": "comfortable", "slack": 95, "history": history}
    notes: list[str] = []
    deadline_planner._apply_history(option, notes)
    assert option["verdict"] == "tight" and "history:gdp:low_visibility:gdp" in notes[0]
    enough = {"verdict": "comfortable", "slack": 120, "history": history}
    deadline_planner._apply_history(enough, [])
    assert enough["verdict"] == "comfortable"
    already_tight = {"verdict": "tight", "slack": 10, "history": history}
    deadline_planner._apply_history(already_tight, [])
    assert already_tight["verdict"] == "tight" and not already_tight.get("history_downgraded")


def test_history_applies_to_the_arrival_airport_and_its_neighbours_only(monkeypatch):
    monkeypatch.setattr(gdp_history, "load_table", lambda: _table(80, 40))
    reasons = {"SFO": "low ceilings"}
    side = ["SFO", "OAK", "SJC"]
    assert deadline_planner._history_for_arrival("SFO", side, reasons)["role"] == "gdp"
    assert deadline_planner._history_for_arrival("OAK", side, reasons)["role"] == "neighbour"
    # a program at the ORIGIN airport does not hold flights leaving it, so no history margin
    assert deadline_planner._history_for_arrival("OAK", ["OAK"], {"JFK": "low ceilings"}) is None


def test_history_changes_the_plan_and_is_cited(monkeypatch):
    # With a huge learned margin, flights INTO the program airport (SFO) cannot be "comfortable"...
    _, by_id = _plan(monkeypatch, _table(gdp_q90=600, neighbour_q90=600))
    flights = {key: item["data"] for key, item in by_id.items() if key.startswith("plan:flight:")}
    into_sfo = [data for key, data in flights.items() if key.endswith("-SFO")]
    assert into_sfo and all(data["verdict"] != "comfortable" for data in into_sfo)
    # ...while the neighbour (OAK) cites its own statistic but is not downgraded (rule v2).
    oak = flights["plan:flight:EWR-OAK"]
    assert oak["history_id"] == "history:gdp:low_visibility:neighbour" and not oak["history_downgraded"]
    assert oak["verdict"] == "comfortable"


def test_history_downgrade_is_explained_and_cited(monkeypatch):
    replay.use_demo_scenario("sfo_gdp_20260923")
    monkeypatch.setattr(gdp_history, "load_table", lambda: _table(gdp_q90=600, neighbour_q90=600))
    ctx = SimpleNamespace(state={"user_id": "history_user"})
    # without history EWR-SFO is "comfortable" here (96 min to spare after the announced 34 min)
    result = plan_trip_options("New York", "San Francisco", "2026-09-23 21:30", tool_context=ctx)
    by_id = {item["id"]: item for item in result["evidence"]}
    downgraded = [item["data"] for key, item in by_id.items() if key.startswith("plan:flight:") and item["data"]["history_downgraded"]]
    assert downgraded, "an SFO option that was comfortable is downgraded by history"
    for data in downgraded:
        assert data["route"].endswith("SFO") and data["verdict"] == "tight"
        assert data["history_id"] in data["based_on"] and data["history_id"] in by_id
        assert any("real history" in note for note in data["notes"])


def test_small_learned_margin_leaves_the_plan_unchanged(monkeypatch):
    _, with_history = _plan(monkeypatch, _table(gdp_q90=0, neighbour_q90=0))
    _, without = _plan(monkeypatch, None)
    verdicts = lambda by_id: {k: v["data"]["verdict"] for k, v in by_id.items() if k.startswith("plan:flight:")}  # noqa: E731
    assert verdicts(with_history) == verdicts(without)
    assert not any(key.startswith("history:") for key in without)  # no table -> no history evidence


def test_history_statistic_is_citable_but_says_it_is_not_about_today(monkeypatch):
    result, by_id = _plan(monkeypatch, _table(gdp_q90=600, neighbour_q90=600))
    item = next(item for key, item in by_id.items() if key.startswith("history:gdp:"))
    assert "not a report about today" in item["observation"]
    assert item["data"]["sources"] and item["url"].startswith("https://")
    # A claim that quotes the learned number with its id verifies; a made-up number does not.
    state: dict = {}
    add_tool_result(state, result)
    minutes = item["data"]["q90_minutes"]
    good = {"text": f"In similar past programs 1 in 10 flights were {minutes} min or more late.", "evidence_ids": [item["id"]], "risk": "low"}
    bad = {"text": "In similar past programs 1 in 10 flights were 12 min or more late.", "evidence_ids": [item["id"]], "risk": "low"}
    verdicts = verify_claims([good, bad], state["evidence_store"])["verdicts"]
    assert [v["status"] for v in verdicts][0] == "supported", verdicts[0]
    assert verdicts[1]["status"] != "supported"


def test_real_history_table_has_provenance():
    table = gdp_history.load_table()
    if table is None:
        pytest.skip("history table not built (run python -m eval.backtest_gdp_history)")
    assert "2025" in table["learned_from"]
    assert {"all"} <= set(table["groups"])
    for row in table["groups"].values():
        assert row["gdp_flights"] >= 0 and 0 <= row["gdp_cancel_or_divert_rate"] <= 1
    assert "sfo_fog_gdp_oak" in table["checks_case_lessons"]
