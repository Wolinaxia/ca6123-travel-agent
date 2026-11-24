"""Select airport evidence that applies to the departure or estimated arrival window."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from travel_agent.core.engine.models import Policy, TripRequest
from travel_agent.core.reference import airports, bts

Window = tuple[datetime, datetime]


def airport_windows(goal: dict) -> dict[str, Window]:
    """Build half-open windows from the planner's local departure date and airports."""
    origin = goal.get("origin_airport")
    destination = goal.get("destination_airport")
    window_name = goal.get("departure_window", "whole_day")
    if not origin or not destination or window_name not in ("morning", "afternoon", "evening", "whole_day"):
        return {}

    try:
        day = date.fromisoformat(goal["departure_date"])
        origin_zone = ZoneInfo((airports.get(origin) or {})["timezone"])
        destination_zone = ZoneInfo((airports.get(destination) or {})["timezone"])
    except (KeyError, TypeError, ValueError, ZoneInfoNotFoundError):
        return {}

    trip = TripRequest(origin_text=origin, destination_text=destination,
                       departure_date=day, departure_window=window_name)
    first_hour, last_hour = trip.window_hours
    midnight = datetime.combine(day, time.min, origin_zone)
    departure = midnight + timedelta(hours=first_hour)
    departure_end = midnight + timedelta(hours=last_hour)

    minutes = bts.block_minutes(origin, destination)
    duration = timedelta(minutes=minutes if minutes is not None else Policy().default_block_minutes)
    # Add elapsed flight time in UTC, including on days when local clocks change.
    arrival = (departure.astimezone(timezone.utc) + duration).astimezone(destination_zone)
    arrival_end = (departure_end.astimezone(timezone.utc) + duration).astimezone(destination_zone)
    return {origin: (departure, departure_end), destination: (arrival, arrival_end)}


def _timestamp(value: str) -> datetime:
    """Require an explicit offset so the machine's local zone cannot affect filtering."""
    if not isinstance(value, str):
        raise ValueError("Missing timestamp")
    # Python 3.10's fromisoformat needs an explicit UTC offset in place of Z.
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None:
        raise ValueError("Missing time zone")
    return parsed.astimezone(timezone.utc)


def _airport_record(item: dict, window: Window) -> dict | None:
    start, end = window
    try:
        fetched = _timestamp(item.get("fetched_at"))
        valid_from = item.get("valid_from")
        valid_to = item.get("valid_to")
        current_faa = item.get("source") in ("mock_faa", "faa_nas")
        nws_alert = item.get("source") == "nws"
        if item.get("source") == "open_meteo":
            # The daily adapter uses timezone=auto. An endpoint window can touch
            # two local dates; an end at exactly midnight excludes that next day.
            day = date.fromisoformat(item["data"].get("date") or "")
            if start.date() <= day <= (end - timedelta(microseconds=1)).date():
                return {**item, "data": {**item["data"], "temporal_scope": "daily"}}
            return None
        if nws_alert and valid_from is None and valid_to is None and item["data"].get("active_alerts") == 0:
            if fetched >= end:
                return None
            data = {**item["data"], "as_of_local": fetched.astimezone(start.tzinfo).isoformat(),
                    "temporal_scope": "snapshot"}
            return {**item, "data": data}
        if valid_from is None and valid_to is None:
            if current_faa:
                observed = fetched
            elif item["source"] == "aviation_wx" and item["id"].endswith(":metar"):
                observed = _timestamp(item["data"].get("observed_at"))
            else:
                return None
            local = observed.astimezone(start.tzinfo)
            if observed > fetched or local.date() < start.date() or observed >= end:
                return None
            data = {**item.get("data", {}), "as_of_local": local.isoformat(), "temporal_scope": "snapshot"}
            return {**item, "data": data}
        if valid_from is None and (current_faa or nws_alert):
            since = fetched  # Active programs and alerts sometimes supply only an end.
        else:
            since = _timestamp(valid_from)
        until = _timestamp(valid_to)
    except (TypeError, ValueError, OverflowError):
        return None
    if since < until and since < end and start < until:
        return item
    return None


def filter_airport_result(result: dict, airport: str, window: Window) -> dict:
    """Restrict a single-airport lookup to its requested subject, source and time."""
    kept = []
    if result["ok"]:
        for item in result["evidence"]:
            if item["subject"] == airport and item["source"] == result["source"]:
                selected = _airport_record(item, window)
                if selected is not None:
                    kept.append(selected)
    return {**result, "evidence": kept}


def filter_tool_result(result: dict, trip_goal: dict) -> dict:
    """Return a ToolResult-shaped copy containing only eligible original evidence."""
    windows = airport_windows(trip_goal) if result["ok"] else {}
    kept = []
    for item in result["evidence"]:
        if item["source"] == "bts" and windows:
            origin = trip_goal["origin_airport"]
            destination = trip_goal["destination_airport"]
            data = item["data"]
            month = windows[origin][0].month
            basis = data.get("basis") or ""
            matching_period = basis == "annual average" or re.fullmatch(rf"[0-9]{{4}}-{month:02d}", basis)
            if (item["subject"] == f"{origin}-{destination}" and data.get("origin") == origin
                    and data.get("destination") == destination and data.get("month") == month and matching_period):
                kept.append(item)
            continue
        window = windows.get(item["subject"])
        if window:
            selected = _airport_record(item, window)
            if selected is not None:
                kept.append(selected)
    return {**result, "evidence": kept}
