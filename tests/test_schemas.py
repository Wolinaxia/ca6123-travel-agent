from travel_agent.tools import mock


def test_mock_tool_shape():
    r = mock.get_airport_delays("sfo")
    assert r["ok"] is True
    ev = r["evidence"][0]
    assert ev["id"] and ev["subject"] == "SFO"
    assert ev["data"]["avg_delay_min"] == 34


def test_failing_tool_has_no_evidence():
    r = mock.failing_tool("JFK")
    assert r["ok"] is False and r["evidence"] == [] and r["error"]
