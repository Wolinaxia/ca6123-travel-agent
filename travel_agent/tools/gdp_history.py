"""What real flights did during past FAA ground delay programs (M4, Learn stage).

The past cases say things like "a delay program at SFO often does not apply to
Oakland" or "storm delays build during the day". This module holds the same
lessons *measured on real data*, so the planner (code, not the model) can use
them when it rates an option:

    travel_agent/data/processed/gdp_history.json

That table is built by ``eval/backtest_gdp_history.py`` from two public US
government datasets: FAA ATCSCC ground delay program advisories and BTS
per-flight on-time records. It was learned on 2025 programs and back-tested on
2026 programs (``eval/results/m4_gdp_backtest.md``).

Rule used by the planner (``deadline_planner._apply_history``): when an option
lands at an airport with an active ground delay program, it is rated
"comfortable" only if it would still make the deadline at the 90th-percentile
delay (beyond the announced average) that real flights had in 2025 programs of
the same kind. Otherwise it is rated "tight" and the reason is shown with its
evidence id. Options landing at a neighbour airport cite the neighbour statistic
but are not downgraded: the data showed neighbours were safer (rule v1 did
downgrade them and made choices slightly worse).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from travel_agent.schemas import Evidence

TABLE_PATH = Path(__file__).parents[1] / "data" / "processed" / "gdp_history.json"
SOURCE = "faa_bts_history"
SOURCES = ["https://www.fly.faa.gov/adv/advADB", "https://www.transtats.bts.gov/"]
GROUP_TEXT = {
    "thunderstorms": "thunderstorms",
    "low_visibility": "low ceilings / fog",
    "wind": "wind",
    "snow_ice": "snow / ice",
    "other": "volume, staffing, construction or other causes",
    "all": "any cause",
}


def condition_group(text: str | None) -> str:
    """Map an FAA 'impacting condition' / NAS 'reason' text to a program kind."""
    text = (text or "").upper()
    if "THUNDER" in text or "TSTM" in text:
        return "thunderstorms"
    if any(word in text for word in ("CEILING", "FOG", "VISIB")):
        return "low_visibility"
    if "WIND" in text:
        return "wind"
    if "SNOW" in text or "ICE" in text:
        return "snow_ice"
    return "other"


@lru_cache(maxsize=1)
def load_table() -> dict[str, Any] | None:
    try:
        return json.loads(TABLE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row(group: str, role: str) -> tuple[str, dict[str, Any]] | None:
    table = load_table()
    if not table:
        return None
    groups = table.get("groups") or {}
    minimum = int(table.get("min_flights") or 0)
    count_key = "gdp_flights" if role == "gdp" else "neighbour_flights"
    for key in (group, "all"):
        row = groups.get(key)
        if row and int(row.get(count_key) or 0) >= minimum:
            return key, row
    return None


def history_margin(reason: str | None, role: str) -> dict[str, Any] | None:
    """Minutes of extra delay to plan for, learned from real 2025 programs.

    ``role="gdp"``: the option lands at the program airport; the margin is the
    90th percentile of (real arrival delay - announced average).
    ``role="neighbour"``: the option lands at another airport of the same metro;
    the margin is the 90th percentile of real arrival delay there.
    Returns ``None`` when no table is available.
    """
    found = _row(condition_group(reason), role)
    if found is None:
        return None
    group, row = found
    if role == "gdp":
        minutes = row["gdp_extra_over_announced_q90"]
        lost = row["gdp_cancel_or_divert_rate"]
    else:
        minutes = row["neighbour_delay_q90"]
        lost = row["neighbour_cancel_or_divert_rate"]
    return {"id": f"history:gdp:{group}:{role}", "group": group, "role": role, "minutes": max(0, int(minutes)),
            "cancel_or_divert_rate": lost, "programs": row["programs"],
            "flights": row["gdp_flights" if role == "gdp" else "neighbour_flights"]}


def history_evidence(margin: dict[str, Any]) -> Evidence:
    """One evidence item the agent can cite (a historical statistic, not today's status)."""
    table = load_table() or {}
    what = GROUP_TEXT.get(margin["group"], margin["group"])
    if margin["role"] == "gdp":
        place = "flights into the program airport"
        delay = f"arrived {margin['minutes']} min or more after the announced average delay"
    else:
        place = "flights into a neighbouring airport of the same metro (no program of its own)"
        delay = f"arrived {margin['minutes']} min or more late"
    observation = (
        f"Real history ({table.get('learned_from', '2025')}, {margin['programs']} FAA ground delay programs caused by {what}): "
        f"1 in 10 {place} {delay}; {100 * margin['cancel_or_divert_rate']:.1f}% were cancelled or diverted. "
        "Historical statistic, not a report about today."
    )
    return Evidence(
        id=margin["id"],
        source=SOURCE,
        subject=margin["group"],
        observation=observation,
        data={
            "kind": "history_statistic",
            "program_kind": margin["group"],
            "role": margin["role"],
            "q90_minutes": margin["minutes"],
            "cancel_or_divert_rate": margin["cancel_or_divert_rate"],
            "programs": margin["programs"],
            "flights": margin["flights"],
            "learned_from": table.get("learned_from"),
            "checks_case_lessons": table.get("checks_case_lessons") or [],
            "sources": SOURCES,
        },
        fetched_at=datetime.now(timezone.utc).isoformat(),
        url=SOURCES[0],
    )
