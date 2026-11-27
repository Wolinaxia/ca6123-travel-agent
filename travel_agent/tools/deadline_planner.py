"""Deadline-aware option planner: "Will I make it?" + Plan B with a trigger (M4).

This is a deterministic ADK tool. The Alternatives agent calls it once, then
explains the result as claims. All arithmetic happens here, in plain Python, so
the numbers are reproducible and the Verifier can check every number the agent
quotes against the evidence this tool returns.

For each candidate option (flights between every pair of metro airports, rail,
drive) it computes:

* ``door_to_door_min``  access + airport buffer + flight block time + egress
                        (rail: train time + station access; drive: route time)
* ``expected_delay_min`` delay already announced by the FAA right now
* ``slack_min``          deadline minus earliest possible arrival (with delay)
* ``delay_tolerance_min`` extra delay the option can absorb before missing
* ``latest_departure``  last departure that still keeps the user's buffer
* ``verdict``           comfortable / tight / at_risk / likely_miss

and then a contingency plan: Plan A, Plan B, the trigger that should make the
traveller switch, and the time by which they must decide.

Learned from history (M4 Learn stage): when a flight lands at an airport with an
active FAA ground delay program, the option is rated "comfortable" only if it
still makes the deadline at the 90th-percentile delay that REAL flights had in
2025 programs of the same kind (neighbour airports of the same metro get their
own, lower statistic as evidence but are not downgraded) (FAA advisories + BTS
on-time data, see ``gdp_history.py``; back-tested on 2026 in
``eval/results/m4_gdp_backtest.md``). The statistic is cited as evidence
``history:gdp:<kind>:<role>``.

Honest limits (also returned in the evidence): we have no live flight
schedules or seat availability, so a flight option means "a non-stop on this
route leaving between earliest departure and latest departure". Rail delay is
not live. Missing sources are reported in ``plan:data_gaps``, never guessed.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from travel_agent.core.engine.models import Policy
from travel_agent.core.reference import bts
from travel_agent.core.reference.airports import airports_near
from travel_agent.memory.preferences import PreferenceStore, _user_id_from, preference_evidence
from travel_agent.schemas import Evidence, ToolResult

from . import gdp_history, live

SOURCE = "deadline_planner"
DEFAULT_MIN_BUFFER_MIN = 60
COMFORTABLE_MIN = 90          # slack at or above this (and the buffer) is "comfortable"
MAX_AIRPORTS_PER_SIDE = 3
MAX_FLIGHT_OPTIONS = 6
DECIDE_MARGIN_MIN = 30        # decide on Plan B this long before its latest departure

_VERDICT_RANK = {"comfortable": 0, "tight": 1, "at_risk": 2, "likely_miss": 3}
_LEVEL_RANK = {"low": 0, "elevated": 1, "high": 2}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

SAME_PLACE_KM = 30  # closer than this, origin and destination are the same metro


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    import math

    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _parse_local(text: str, tz_name: str) -> datetime:
    """Parse '2026-09-23 18:00' (local to ``tz_name``) or an ISO time with offset."""
    raw = (text or "").strip().replace("Z", "+00:00")
    if not raw:
        raise ValueError("empty time")
    value = datetime.fromisoformat(raw.replace(" ", "T", 1) if "T" not in raw else raw)
    if value.tzinfo is None:
        value = value.replace(tzinfo=ZoneInfo(tz_name))
    return value


def _fmt(value: datetime | None, tz_name: str) -> str | None:
    if value is None:
        return None
    local = value.astimezone(ZoneInfo(tz_name))
    return local.strftime("%Y-%m-%d %H:%M ") + (local.tzname() or "")


def _minutes_between(later: datetime, earlier: datetime) -> int:
    return int(round((later - earlier).total_seconds() / 60.0))


def _metro_airports(lat: float, lon: float, radius_km: float, avoid: set[str]) -> tuple[list[str], list[str]]:
    """Busiest scheduled airports near a point; returns (kept, avoided)."""
    ranked = []
    for airport in airports_near(lat, lon, radius_km):
        code = str(airport.get("iata") or "").upper()
        if not code:
            continue
        flights = bts.airport_flights(code)
        if flights > 0:
            ranked.append((flights, code))
    ranked.sort(reverse=True)
    kept, avoided = [], []
    for _, code in ranked:
        if code in avoid:
            avoided.append(code)
        elif len(kept) < MAX_AIRPORTS_PER_SIDE:
            kept.append(code)
    return kept, avoided


def _airport_delay(result: dict[str, Any]) -> tuple[int, bool, list[str]]:
    """(expected delay minutes, disrupted?, evidence ids) from an FAA ToolResult."""
    delay = 0
    disrupted = False
    ids = []
    for item in result.get("evidence") or []:
        data = item.get("data") or {}
        ids.append(item["id"])
        program = data.get("program")
        if program == "GDP":
            delay = max(delay, int(data.get("avg_delay_min") or 0))
        elif program == "arrival_departure":
            low = int(data.get("min_delay_min") or 0)
            high = int(data.get("max_delay_min") or low)
            delay = max(delay, (low + high) // 2)
        elif program in {"ground_stop", "closure"}:
            disrupted = True
    return delay, disrupted, ids


def _downgrade_if_delay_unknown(option: dict[str, Any], notes: list[str]) -> None:
    """Missing FAA data is not "no delay": such an option is never rated comfortable."""
    if option.get("unknown_airports"):
        codes = ", ".join(option["unknown_airports"])
        notes.append(f"FAA status for {codes} unavailable: delay unknown, so this option is never rated comfortable")
        if option["verdict"] == "comfortable":
            option["verdict"] = "tight"


def _gdp_reason(result: dict[str, Any]) -> str | None:
    """Reason text of an active ground delay program in an FAA ToolResult, if any."""
    for item in result.get("evidence") or []:
        data = item.get("data") or {}
        if data.get("program") == "GDP":
            return str(data.get("reason") or "")
    return None


def _history_for_arrival(arrival: str, side: list[str], gdp_reasons: dict[str, str]) -> dict[str, Any] | None:
    """History margin for landing at ``arrival`` (learned from real 2025 programs).

    Ground delay programs hold flights bound FOR the program airport, so only the
    arrival airport matters: either it has a program itself ("gdp"), or another
    airport of the same metro has one ("neighbour").
    """
    if arrival in gdp_reasons:
        return gdp_history.history_margin(gdp_reasons[arrival], "gdp")
    for code in side:
        if code != arrival and code in gdp_reasons:
            return gdp_history.history_margin(gdp_reasons[code], "neighbour")
    return None


def _apply_history(option: dict[str, Any], notes: list[str]) -> None:
    """Learned lesson: "comfortable" must survive the delays real flights had in similar programs.

    If an option landing at the program airport would miss the deadline at the
    90th-percentile delay seen in real 2025 programs of the same kind (beyond the
    announced average), it is rated "tight", never "comfortable". Neighbours keep
    their verdict; their (lower) history is still cited as evidence.
    """
    history = option.get("history")
    # Only the program airport itself: the back-test showed same-metro neighbours were
    # safer in 94% of 2026 programs, so penalising them made choices worse (rule v1).
    if not history or history["role"] != "gdp" or option["verdict"] != "comfortable":
        return
    if option["slack"] < history["minutes"]:
        notes.append(
            f"real history ({history['id']}): 1 in 10 similar flights were {history['minutes']} min or more "
            f"{'later than announced' if history['role'] == 'gdp' else 'late'}, more than this option's "
            f"{option['slack']} min to spare, so it is rated tight, not comfortable")
        option["verdict"] = "tight"
        option["history_downgraded"] = True


def _verdict(slack: int, min_buffer: int, level: str, disrupted: bool) -> str:
    if slack < 0:
        return "likely_miss"
    if disrupted:
        return "at_risk"
    if slack >= max(COMFORTABLE_MIN, min_buffer):
        return "tight" if level == "high" else "comfortable"
    return "tight"


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

def plan_trip_options(origin: str, destination: str, deadline: str, earliest_departure: str = "", tool_context: Any = None) -> dict:
    """Compare every way to reach `destination` before `deadline` and build Plan A / Plan B.

    `origin` and `destination` are US cities or IATA codes. `deadline` is the
    latest acceptable arrival, local time at the destination, as
    "YYYY-MM-DD HH:MM". `earliest_departure` is the earliest the traveller can
    leave, local time at the origin, "YYYY-MM-DD HH:MM"; leave it empty to use
    the current time.

    Compares non-stop flights between all metro airports, rail and driving,
    using live FAA delays, BTS history, Amtrak and road data, and the user's
    saved preferences. Each option is one evidence item `plan:<mode>:<route>`
    with door-to-door minutes, expected delay, slack before the deadline, the
    latest safe departure and a verdict (comfortable / tight / at_risk /
    likely_miss). `plan:contingency` holds Plan A, Plan B, the switch trigger
    and the decide-by time. `plan:data_gaps` lists sources that failed. Quote
    numbers exactly as given and cite the plan ids in your claims.
    """
    try:
        policy = Policy()
        store = PreferenceStore(_user_id_from(tool_context))
        prefs = store.get_all()
        avoid = set(prefs.get("avoid_airports") or [])
        min_buffer = int(prefs.get("min_buffer_minutes") or DEFAULT_MIN_BUFFER_MIN)
        max_drive_h = prefs.get("max_drive_hours")
        rail_pref_h = prefs.get("prefer_rail_under_hours")
        avoid_redeye = bool(prefs.get("avoid_redeye"))

        origin_info = live._place_info(origin)
        destination_info = live._place_info(destination)
        origin_tz = origin_info["timezone"]
        destination_tz = destination_info["timezone"]
        now = live._now()
        deadline_at = _parse_local(deadline, destination_tz)
        start_at = _parse_local(earliest_departure, origin_tz) if (earliest_departure or "").strip() else now
        if start_at < now:
            start_at = now
        if deadline_at <= start_at:
            raise ValueError("deadline is not after the earliest departure")
        apart_km = _distance_km(origin_info["place_lat"], origin_info["place_lon"],
                                destination_info["place_lat"], destination_info["place_lon"])
        if apart_km < SAME_PLACE_KM:
            raise ValueError(f"origin and destination are the same place ({apart_km:.0f} km apart); nothing to plan")

        evidence: list[Evidence] = []
        passthrough: list[dict[str, Any]] = []   # evidence dicts from the underlying live tools
        gaps: list[str] = []
        options: list[dict[str, Any]] = []
        excluded: list[tuple[str, str, str]] = []  # (id suffix, observation, reason)
        pref_ids = {f"pref:{store.user_id}:{key}" for key in prefs}

        def keep(result: dict[str, Any], label: str) -> bool:
            if result.get("ok"):
                passthrough.extend(result.get("evidence") or [])
                return True
            error = str(result.get("error") or "unavailable").split(": GET ")[0][:120]
            gaps.append(f"{label}: {error}")
            return False

        # ---- flights between metro airports -------------------------------
        origin_airports, origin_avoided = _metro_airports(origin_info["place_lat"], origin_info["place_lon"], policy.origin_metro_radius_km, avoid)
        destination_airports, destination_avoided = _metro_airports(destination_info["place_lat"], destination_info["place_lon"], policy.destination_metro_radius_km, avoid)
        for code in origin_avoided + destination_avoided:
            excluded.append((f"airport:{code}", f"{code} skipped because the user avoids it", "avoid_airports"))

        faa: dict[str, tuple[int, bool, list[str]]] = {}
        faa_missing: set[str] = set()   # airports whose FAA status could not be read
        gdp_reasons: dict[str, str] = {}  # airports with an active ground delay program -> reason
        for code in origin_airports + destination_airports:
            result = live.get_faa_airport_status(code)
            if keep(result, f"FAA status {code}"):
                faa[code] = _airport_delay(result)
                reason = _gdp_reason(result)
                if reason is not None:
                    gdp_reasons[code] = reason
            else:
                # Unknown is NOT the same as "no delay": remember it so the
                # option can never be called comfortable on missing data.
                faa[code] = (0, False, [])
                faa_missing.add(code)

        routes = []
        for o in origin_airports:
            for d in destination_airports:
                block = bts.block_minutes(o, d)
                if block:
                    routes.append((block, o, d))
        routes.sort()
        for block, o, d in routes[:MAX_FLIGHT_OPTIONS]:
            history = live.get_route_ontime_history(o, d, deadline_at.month)
            history_ids, dep_rate, cancel_rate = [], None, None
            if history.get("ok") and history.get("evidence"):
                passthrough.extend(history["evidence"])
                item = history["evidence"][0]
                history_ids = [item["id"]]
                dep_rate = item["data"].get("dep_del15")
                cancel_rate = item["data"].get("cancel_rate")
            o_delay, o_disrupted, o_ids = faa.get(o, (0, False, []))
            d_delay, d_disrupted, d_ids = faa.get(d, (0, False, []))
            delay = max(o_delay, d_delay)
            disrupted = o_disrupted or d_disrupted
            unknown = sorted({o, d} & faa_missing)
            # Metro airports with an FAA delay/stop that this option does NOT use:
            # stated in the evidence so "OAK avoids the SFO delay" is checkable.
            avoided = {
                code: faa[code][0]
                for side, used in ((origin_airports, o), (destination_airports, d))
                for code in side
                if code != used and code in faa and (faa[code][0] > 0 or faa[code][1])
            }
            level = "elevated" if unknown else "low"
            if disrupted or delay >= policy.gdp_high_avg_minutes:
                level = "high"
            elif delay >= policy.gdp_elevated_avg_minutes or (dep_rate or 0) >= policy.bts_delay_threshold or (cancel_rate or 0) >= policy.bts_cancel_threshold:
                level = "elevated"
            door = int(round(policy.access_fallback_minutes + policy.airport_buffer_hours * 60 + block + policy.egress_hours * 60))
            options.append(
                {
                    "mode": "flight",
                    "route": f"{o}-{d}",
                    "label": f"Flight {o} to {d}",
                    "airports": [o, d],
                    "door": door,
                    "delay": delay,
                    "level": level,
                    "disrupted": disrupted,
                    "unknown_airports": unknown,
                    "history": _history_for_arrival(d, destination_airports, gdp_reasons),
                    "based_on": o_ids + d_ids + history_ids,
                    "extra": {
                        "block_min": int(round(block)),
                        "airport_buffer_min": int(policy.airport_buffer_hours * 60),
                        "access_min": int(policy.access_fallback_minutes),
                        "egress_min": int(policy.egress_hours * 60),
                        "historical_dep_delay_rate": dep_rate,
                        "avoids_delays_at": avoided,
                        "historical_cancel_rate": cancel_rate,
                    },
                    "assumption": "Non-stop on this route; no live schedule or seat data. Door-to-door uses 45 min to the airport, 2 h airport buffer, median block time and 1 h at arrival.",
                }
            )
        if not routes:
            gaps.append("flights: no non-stop route with BTS block time between the metro airports")

        # ---- rail ----------------------------------------------------------
        rail = live.get_rail_options(origin, destination)
        if keep(rail, "rail"):
            services = [item for item in rail.get("evidence") or [] if (item.get("data") or {}).get("min_duration_min")]
            services.sort(key=lambda item: float(item["data"]["min_duration_min"]))
            for item in services[:2]:
                minutes = float(item["data"]["min_duration_min"])
                door = int(round(minutes + policy.rail_station_access_minutes))
                options.append(
                    {
                        "mode": "rail",
                        "route": item["subject"],
                        "label": f"Amtrak {item['subject'].replace('-', ' to ')}",
                        "airports": [],
                        "door": door,
                        "delay": 0,
                        "level": "low",
                        "disrupted": False,
                        "based_on": [item["id"]],
                        "extra": {"train_min": int(round(minutes)), "station_access_min": int(policy.rail_station_access_minutes), "trains_per_day": item["data"].get("trains_per_day")},
                        "assumption": "Fastest direct train plus 60 min station access; live train delays not included.",
                    }
                )

        # ---- drive ---------------------------------------------------------
        drive = live.get_drive_route(origin, destination)
        if keep(drive, "drive"):
            item = drive["evidence"][0]
            door = int(round(float(item["data"]["duration_min"])))
            option = {
                "mode": "drive",
                "route": item["subject"],
                "label": f"Drive {item['subject'].replace('-', ' to ')}",
                "airports": [],
                "door": door,
                "delay": 0,
                "level": "low" if door <= 10 * 60 else "elevated",
                "disrupted": False,
                "based_on": [item["id"]],
                "extra": {"distance_km": item["data"].get("distance_km")},
                "assumption": "Route time without traffic or rest stops.",
            }
            if max_drive_h is not None and door > float(max_drive_h) * 60:
                excluded.append((f"drive:{item['subject']}", f"Drive {item['subject']} ({door} min) skipped: longer than the user's {float(max_drive_h):g} h drive limit", "max_drive_hours"))
            else:
                options.append(option)

        # ---- score every option ---------------------------------------------
        for option in options:
            arrival = start_at + timedelta(minutes=option["door"] + option["delay"])
            slack = _minutes_between(deadline_at, arrival)
            tolerance = _minutes_between(deadline_at, start_at + timedelta(minutes=option["door"]))
            latest = deadline_at - timedelta(minutes=option["door"] + option["delay"] + min_buffer)
            option["slack"] = slack
            option["tolerance"] = tolerance
            option["arrival"] = arrival
            option["latest"] = latest if latest >= start_at else None
            option["verdict"] = _verdict(slack, min_buffer, option["level"], option["disrupted"])
            notes = []
            _downgrade_if_delay_unknown(option, notes)
            _apply_history(option, notes)
            penalty = 0.0
            if option["mode"] == "flight" and avoid_redeye and option["latest"] is not None:
                local_latest = option["latest"].astimezone(ZoneInfo(origin_tz))
                window_h = (option["latest"] - start_at).total_seconds() / 3600
                if (local_latest.hour >= 21 or local_latest.hour < 5) and window_h < 4:
                    notes.append("only overnight departures fit the window; user avoids red-eyes")
                    penalty += 1.0
            if option["mode"] == "rail" and rail_pref_h is not None and option["door"] <= float(rail_pref_h) * 60 and option["verdict"] in {"comfortable", "tight"}:
                notes.append(f"user prefers rail under {float(rail_pref_h):g} h")
                penalty -= 0.6
            option["notes"] = notes
            option["rank_key"] = (_VERDICT_RANK[option["verdict"]] + penalty, _LEVEL_RANK[option["level"]], option["door"] + option["delay"])
        options.sort(key=lambda option: option["rank_key"])

        # ---- learned history used by any option ---------------------------------
        history_items: dict[str, Evidence] = {}
        for option in options:
            history = option.get("history")
            if history:
                if history["id"] not in history_items:
                    history_items[history["id"]] = gdp_history.history_evidence(history)
                option["based_on"] = option["based_on"] + [history["id"]]
        evidence.extend(history_items.values())

        # ---- option evidence --------------------------------------------------
        for rank, option in enumerate(options, start=1):
            option_id = f"plan:{option['mode']}:{option['route']}"
            option["id"] = option_id
            status = {
                "comfortable": f"arrives with {option['slack']} min to spare",
                "tight": f"arrives with only {option['slack']} min to spare",
                "at_risk": f"arrives with {option['slack']} min to spare but an FAA ground stop or closure is active",
                "likely_miss": f"misses the deadline by {-option['slack']} min",
            }[option["verdict"]]
            delay_text = f" plus {option['delay']} min announced delay" if option["delay"] else ""
            avoided = (option.get("extra") or {}).get("avoids_delays_at") or {}
            if avoided:
                delay_text += "; avoids the FAA delay at " + ", ".join(
                    f"{code} ({minutes} min)" if minutes else f"{code} (ground stop/closure)" for code, minutes in sorted(avoided.items()))
            latest_text = f"; latest safe departure {_fmt(option['latest'], origin_tz)}" if option["latest"] else "; no departure keeps the buffer"
            data = {
                "mode": option["mode"],
                "route": option["route"],
                "rank": rank,
                "verdict": option["verdict"],
                "level": option["level"],
                "door_to_door_min": option["door"],
                "expected_delay_min": option["delay"],
                "delay_unknown": bool(option.get("unknown_airports")),
                "history_margin_min": (option.get("history") or {}).get("minutes"),
                "history_id": (option.get("history") or {}).get("id"),
                "history_downgraded": bool(option.get("history_downgraded")),
                "slack_min": option["slack"],
                "shortfall_min": max(0, -option["slack"]),
                "delay_tolerance_min": option["tolerance"],
                "min_buffer_min": min_buffer,
                "earliest_departure_local": _fmt(start_at, origin_tz),
                "earliest_arrival_local": _fmt(option["arrival"], destination_tz),
                "latest_departure_local": _fmt(option["latest"], origin_tz),
                "deadline_local": _fmt(deadline_at, destination_tz),
                "based_on": option["based_on"],
                "preferences_applied": sorted(pref_ids) if option["notes"] else [],
                "notes": option["notes"],
                "assumption": option["assumption"],
                **option["extra"],
            }
            evidence.append(
                Evidence(
                    id=option_id,
                    source=SOURCE,
                    subject=option["route"],
                    observation=f"#{rank} {option['label']}: door to door {option['door']} min ({option['door'] / 60:.1f} h){delay_text}; {status} ({option['verdict']}){latest_text}",
                    data=data,
                    valid_from=start_at.isoformat(),
                    valid_to=deadline_at.isoformat(),
                    fetched_at=now.isoformat(),
                )
            )

        for suffix, observation, key in excluded:
            evidence.append(
                Evidence(
                    id=f"plan:excluded:{suffix}",
                    source=SOURCE,
                    subject=suffix.split(":", 1)[1],
                    observation=observation,
                    data={"excluded_by_preference": key, "preference_id": f"pref:{store.user_id}:{key}"},
                    fetched_at=now.isoformat(),
                )
            )

        # ---- contingency: Plan A, Plan B, trigger, decide-by ---------------------
        viable = [option for option in options if option["verdict"] != "likely_miss"]
        plan_a = viable[0] if viable else None
        plan_b = None
        if plan_a is not None:
            # Plan B must not fail for the same reason as Plan A: a different
            # mode, or (for flights) a route that shares no airport with Plan A.
            for option in viable[1:]:
                if option["mode"] != plan_a["mode"]:
                    plan_b = option
                    break
                if plan_a["mode"] == "flight" and not set(option["airports"]) & set(plan_a["airports"]):
                    plan_b = option
                    break
        if plan_a is None:
            observation = "No option reaches the destination before the deadline. Escalate to the human: consider moving the deadline or postponing."
            contingency = {"plan_a": None, "plan_b": None, "needs_human": True}
            subject = f"{origin_info.get('iata') or origin}-{destination_info.get('iata') or destination}"
        else:
            threshold = max(15, plan_a["slack"] - min_buffer)
            if plan_a["mode"] == "flight":
                o, d = plan_a["airports"]
                trigger = f"an FAA ground stop or closure at {o} or {d}, or new delays above {threshold} min"
            elif plan_a["mode"] == "rail":
                trigger = f"Amtrak reports a delay above {threshold} min on this route"
            else:
                trigger = f"traffic adds more than {threshold} min to the drive"
            decide_by = None
            if plan_b is not None and plan_b["latest"] is not None:
                decide_by = max(start_at, plan_b["latest"] - timedelta(minutes=DECIDE_MARGIN_MIN))
            subject = plan_a["route"] + (f" | {plan_b['route']}" if plan_b else "")
            decide_now = decide_by is not None and decide_by <= start_at
            if plan_b is not None:
                if decide_by is None:
                    decide_text = ""
                elif decide_now:
                    decide_text = "; Plan B's window is closing, so decide now"
                else:
                    decide_text = f"; decide by {_fmt(decide_by, origin_tz)}"
                observation = (
                    f"Plan A: {plan_a['label']} ({plan_a['verdict']}). Plan B: {plan_b['label']} ({plan_b['verdict']}). "
                    f"Switch to Plan B if {trigger}{decide_text}."
                )
            else:
                observation = f"Plan A: {plan_a['label']} ({plan_a['verdict']}). No independent Plan B arrives in time; watch for {trigger}."
            contingency = {
                "plan_a": plan_a["id"],
                "plan_b": plan_b["id"] if plan_b else None,
                "switch_trigger": trigger,
                "switch_delay_threshold_min": threshold,
                "decide_by_local": _fmt(decide_by, origin_tz) if decide_by else None,
                "decide_now": decide_now,
                "needs_human": plan_a["verdict"] != "comfortable",
            }
        evidence.append(
            Evidence(
                id="plan:contingency",
                source=SOURCE,
                subject=subject,
                observation=observation,
                data=contingency,
                valid_from=start_at.isoformat(),
                valid_to=deadline_at.isoformat(),
                fetched_at=now.isoformat(),
            )
        )
        evidence.append(
            Evidence(
                id="plan:data_gaps",
                source=SOURCE,
                subject="data_quality",
                observation=("Sources unavailable: " + "; ".join(gaps)) if gaps else "All planner sources answered",
                data={"gaps": gaps, "count": len(gaps)},
                fetched_at=now.isoformat(),
            )
        )

        all_evidence = [Evidence(**item) if isinstance(item, dict) else item for item in passthrough]
        all_evidence += preference_evidence(store)
        all_evidence += evidence
        return ToolResult(source=SOURCE, ok=True, evidence=all_evidence).to_dict()
    except Exception as exc:
        return ToolResult(source=SOURCE, ok=False, evidence=[], error=str(exc)).to_dict()
