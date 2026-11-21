"""FAA National Airspace System XML adapter."""

from datetime import datetime, timedelta, timezone
import re
import xml.etree.ElementTree as ET

from ..http import request, source_result
from ..models import (
    FaaArrivalDepartureDelay,
    FaaClosure,
    FaaGroundDelay,
    FaaGroundStop,
    FaaNasData,
    SourceResult,
)


URL = "https://nasstatus.faa.gov/api/airport-status-information"
RANGE_RE = re.compile(r"(\d{10})-(\d{10})")


def parse_duration_minutes(value: str | None) -> int:
    """Parse FAA duration prose such as '1 hour and 23 minutes'."""
    if not value:
        return 0
    hours = re.search(r"(\d+)\s*hour", value, re.IGNORECASE)
    minutes = re.search(r"(\d+)\s*minute", value, re.IGNORECASE)
    return (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)


def _range_times(reason: str) -> tuple[datetime, datetime] | None:
    match = RANGE_RE.search(reason)
    if not match:
        return None

    def parse_compact(value: str) -> datetime:
        return datetime.strptime(value, "%y%m%d%H%M").replace(tzinfo=timezone.utc)

    return parse_compact(match.group(1)), parse_compact(match.group(2))


def _parse_faa_time(value: str | None, now: datetime, fallback: datetime | None = None) -> datetime | None:
    if not value:
        return None
    match = re.search(r"([A-Z][a-z]{2})\s+(\d{1,2}) at (\d{2}):(\d{2}) UTC", value)
    if not match:
        return fallback
    parsed = datetime.strptime(
        f"{now.year} {match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}",
        "%Y %b %d %H %M",
    ).replace(tzinfo=timezone.utc)
    if parsed < now - timedelta(days=60):
        parsed = parsed.replace(year=parsed.year + 1)
    return parsed


def _closure_time(text: str | None, reason: str, now: datetime, is_reopen: bool) -> datetime | None:
    compact = _range_times(reason)
    if compact:
        return compact[1] if is_reopen else compact[0]
    return _parse_faa_time(text, now)


def parse(xml_text: str, now: datetime | None = None) -> FaaNasData:
    now = now or datetime.now(timezone.utc)
    root = ET.fromstring(xml_text)
    data = FaaNasData()
    known_names = {
        "ground delay programs",
        "ground stops",
        "general arrival/departure delay info",
        "airport closures",
    }
    for delay_type in root.findall("./Delay_type"):
        name = delay_type.findtext("Name", "")
        normalized_name = name.strip().lower()
        if normalized_name not in known_names:
            data.unparsed.append({"name": name, "raw_xml_snippet": ET.tostring(delay_type, encoding="unicode")})
    for item in root.findall(".//Ground_Delay"):
        data.ground_delays.append(
            FaaGroundDelay(
                airport=item.findtext("ARPT", ""),
                reason=item.findtext("Reason", ""),
                avg_minutes=parse_duration_minutes(item.findtext("Avg")),
                max_minutes=parse_duration_minutes(item.findtext("Max")),
            )
        )
    stop_nodes = list(root.findall(".//Ground_Stop_List/Ground_Stop"))
    stop_nodes.extend(root.findall(".//Ground_Stop_List/Program"))
    stop_nodes.extend(root.findall(".//Ground_Stop"))
    seen_stops: set[int] = set()
    for item in stop_nodes:
        if id(item) in seen_stops:
            continue
        seen_stops.add(id(item))
        data.ground_stops.append(
            FaaGroundStop(
                airport=item.findtext("ARPT", ""),
                reason=item.findtext("Reason", ""),
                end=_parse_faa_time(item.findtext("End_Time") or item.findtext("End"), now),
            )
        )
    for delay in root.findall(".//Arrival_Departure_Delay_List/Delay"):
        for item in delay.findall("Arrival_Departure"):
            data.arrival_departure_delays.append(
                FaaArrivalDepartureDelay(
                    airport=delay.findtext("ARPT", ""),
                    kind=item.get("Type", ""),
                    reason=delay.findtext("Reason", ""),
                    min_minutes=parse_duration_minutes(item.findtext("Min")),
                    max_minutes=parse_duration_minutes(item.findtext("Max")),
                    trend=item.findtext("Trend"),
                )
            )
    for airport in root.findall(".//Airport"):
        reason = airport.findtext("Reason", "")
        reopen = _closure_time(airport.findtext("Reopen"), reason, now, True)
        start = _closure_time(airport.findtext("Start"), reason, now, False)
        ga_only = "NON SKED" in reason.upper() and "GA" in reason.upper()
        data.closures.append(
            FaaClosure(
                airport=airport.findtext("ARPT", ""),
                reason=reason,
                start=start,
                reopen=reopen,
                affects_commercial=not ga_only,
                active=reopen is None or reopen > now,
            )
        )
    return data


def fetch(now: datetime | None = None) -> SourceResult[FaaNasData]:
    return source_result("faa_nas", URL, lambda: parse(request("GET", URL)[1], now=now))
