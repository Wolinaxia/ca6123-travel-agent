"""Cross-cutting consistency checks on every Assessment the engine can produce.

These encode properties the UI relies on: the leg strip, the evidence table and
the summary must never disagree with each other.
"""

from datetime import datetime, timedelta

import pytest

from travel_agent.core.engine.evaluate import LEVELS, evaluate
from travel_agent.core.engine.models import SourceInputs
from travel_agent.core.models import NwsAlert
from travel_agent.core.sources.aviation_wx import parse as parse_wx
from travel_agent.core.sources.faa_nas import parse as parse_faa
from tests.core.test_engine import FAA, METAR, NOW, TAF, _result, _trip


def _alert(event: str, area: str) -> NwsAlert:
    return NwsAlert(
        event=event,
        severity="Moderate",
        onset=NOW,
        ends=NOW + timedelta(days=3),
        headline=f"{event} for {area}",
        area_desc=area,
        id=f"{event}-{area}",
    )


def _scenarios():
    weather = parse_wx(TAF, METAR)
    faa = _result("faa", parse_faa(FAA, NOW))
    empty = _result("alerts", [])
    yield "sfo_gdp", SourceInputs(
        faa=faa,
        taf_origin=_result("wx", weather),
        taf_destination=_result("wx", weather),
        nws_origin_alerts=empty,
        nws_destination_alerts=empty,
    )
    yield "origin_alert_only", SourceInputs(
        nws_origin_alerts=_result("alerts", [_alert("Wind Advisory", "New York")]),
        nws_destination_alerts=empty,
    )
    yield "destination_alert_only", SourceInputs(
        nws_origin_alerts=empty,
        nws_destination_alerts=_result("alerts", [_alert("Winter Storm Warning", "San Francisco")]),
    )
    yield "both_alerts_and_gdp", SourceInputs(
        faa=faa,
        nws_origin_alerts=_result("alerts", [_alert("Wind Advisory", "New York")]),
        nws_destination_alerts=_result("alerts", [_alert("Winter Storm Warning", "San Francisco")]),
    )
    yield "nothing", SourceInputs()


SCENARIOS = list(_scenarios())


@pytest.mark.parametrize("name,inputs", SCENARIOS, ids=[name for name, _ in SCENARIOS])
def test_assessment_is_internally_consistent(name, inputs):
    assessment = evaluate(_trip(), inputs, NOW)
    evidence_by_id = {item.id: item for item in assessment.evidence}
    rated = [item for item in assessment.evidence if item.level != "info"]

    assert [leg.name for leg in assessment.legs] == ["access", "flight", "egress"]
    for leg in assessment.legs:
        leg_levels = [item.level for item in rated if item.leg == leg.name]
        expected = max(leg_levels, key=LEVELS.get) if leg_levels else "low"
        assert leg.level == expected, f"{leg.name}: level {leg.level} but evidence says {expected}"
        assert all(evidence_by_id[eid].leg == leg.name for eid in leg.evidence_ids)

    worst = max((leg.level for leg in assessment.legs), key=LEVELS.get)
    assert assessment.level == worst
    if assessment.driving_leg is not None:
        assert any(item.leg == assessment.driving_leg for item in rated)
    for eid in assessment.top_evidence_ids:
        assert eid in evidence_by_id


def test_origin_alert_lands_on_access_and_destination_alert_on_egress():
    assessment = evaluate(_trip(), dict(SCENARIOS)["both_alerts_and_gdp"], NOW)
    legs_by_area = {item.detail: item.leg for item in assessment.evidence if item.rule_id.startswith("NWS_")}
    assert any("New York" in detail and leg == "access" for detail, leg in legs_by_area.items())
    assert any("San Francisco" in detail and leg == "egress" for detail, leg in legs_by_area.items())
