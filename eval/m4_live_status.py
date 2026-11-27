"""What the live tools would report during each KIND of disruption (for the retrieval eval).

In the real app the agent searches past cases AFTER it has called the live tools,
so it knows e.g. "FAA: ground delay program, reason low ceilings". The traveller's
message alone often says only "I'll miss my meeting".

To score that situation fairly, every test message is paired with a status line
in the FAA NAS / Amtrak / news vocabulary for the case's KIND of disruption. The
lines are coarse on purpose: several kinds share one line (all ATC failures read
"ground stop, equipment outage"), and they never contain a case name, airport,
airline, date or any text from the case itself. Written before any score with
live status was computed.
"""

from __future__ import annotations

STATUS_BY_KIND = {
    "ground_delay_program": "FAA status: ground delay program at the destination airport, reason low ceilings.",
    "smoke_low_visibility": "FAA status: ground delay program, reason low visibility.",
    "thunderstorm": "FAA status: ground delay program and ground stops, reason thunderstorms.",
    "winter_storm": "FAA status: departure delays and deicing, reason snow / ice.",
    "winter_storm_ice": "FAA status: departure delays and deicing, reason snow / ice.",
    "blizzard": "FAA status: airport closure, reason snow / ice.",
    "hurricane": "FAA status: airport closure, reason hurricane.",
    "hurricane_flood": "FAA status: airport closure, reason hurricane / flooding.",
    "flood_airport_closure": "FAA status: airport closure, reason flooding.",
    "nationwide_ground_stop": "FAA status: ground stop at many airports, reason equipment outage.",
    "atc_equipment_staffing": "FAA status: ground stop, reason equipment outage / staffing.",
    "atc_staffing": "FAA status: ground delays, reason staffing.",
    "atc_capacity_cuts": "FAA status: flight reductions ordered at many airports, reason staffing.",
    "atc_facility_outage": "FAA status: ground stop, reason equipment outage.",
    "drone_ground_stop": "FAA status: ground stop, reason security.",
    "airline_it_outage": "FAA status: no airport program. News: an airline is cancelling flights after a computer outage.",
    "airline_meltdown": "FAA status: no airport program. News: an airline is cancelling many flights.",
    "crew_shortage_illness": "FAA status: no airport program. News: an airline is cancelling many flights.",
    "aircraft_grounding": "FAA status: no airport program. News: an airline is cancelling many flights.",
    "airport_power_outage": "FAA status: ground stop, reason other.",
    "airport_closure_accident": "FAA status: airport closure, reason other.",
    "extreme_heat": "FAA status: departure delays, reason weather.",
    "rail_outage": "Amtrak status: service suspended on the route.",
    "none_short_haul": "FAA status: arrival delays at the destination airport. Amtrak status: trains running.",
}


def with_live_status(message: str, kind: str | None) -> str:
    """The search text the agent would build: traveller message + live status line."""
    status = STATUS_BY_KIND.get(kind or "")
    return f"{message} {status}" if status else message
