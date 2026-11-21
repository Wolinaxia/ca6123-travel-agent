"""Mock tools with the same output shape as the real data-source tools.

Build agents against these until the real tools are merged (target: Tue 24 Nov).
Values are made up for development only. Never show them as real data in the demo.
"""

from __future__ import annotations

from datetime import datetime, timezone

from travel_agent.schemas import Evidence, ToolResult


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_airport_delays(airport: str) -> dict:
    """Return current FAA delay programs for a U.S. airport (IATA code, e.g. 'SFO')."""
    airport = airport.upper()
    ev = Evidence(
        id=f"mock_faa:{airport}:GDP",
        source="mock_faa",
        subject=airport,
        observation=f"[MOCK] Ground Delay Program at {airport}, average delay 34 min",
        data={"program": "GDP", "avg_delay_min": 34},
        fetched_at=_now(),
    )
    return ToolResult(source="mock_faa", ok=True, evidence=[ev]).to_dict()


def get_weather_alerts(airport: str) -> dict:
    """Return active weather alerts near a U.S. airport (IATA code)."""
    airport = airport.upper()
    ev = Evidence(
        id=f"mock_nws:{airport}:wind",
        source="mock_nws",
        subject=airport,
        observation=f"[MOCK] Wind gusts up to 30 kt expected near {airport}",
        data={"gust_kt": 30},
        fetched_at=_now(),
    )
    return ToolResult(source="mock_nws", ok=True, evidence=[ev]).to_dict()


def failing_tool(airport: str) -> dict:
    """Always fails. Use it to test that agents do not invent data when a source is down."""
    return ToolResult(source="mock_down", ok=False, error="source unavailable").to_dict()
