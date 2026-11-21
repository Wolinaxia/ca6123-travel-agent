"""Rule-based trip text parser and metro resolution."""

from datetime import date, datetime, timedelta
import re
from zoneinfo import ZoneInfo

from ..reference.airports import get, by_municipality
from ..reference.airports import airports_near
from ..reference.bts import route_base_rate
from ..sources.open_meteo import geocode
from .models import TripRequest


ALIASES = {
    "nyc": "New York", "new york city": "New York", "sf": "San Francisco", "sfo": "San Francisco",
    "la": "Los Angeles", "los angeles": "Los Angeles", "dc": "Washington DC", "washington": "Washington DC", "the big apple": "New York", "big apple": "New York", "frisco": "San Francisco",
}
KNOWN_CITIES = {
    "new york": ((40.7128, -74.0060), "America/New_York"),
    "san francisco": ((37.7749, -122.4194), "America/Los_Angeles"),
    "boston": ((42.3601, -71.0589), "America/New_York"),
    "washington dc": ((38.9072, -77.0369), "America/New_York"),
    "chicago": ((41.8781, -87.6298), "America/Chicago"),
}
CITY_PRIMARY_AIRPORT = {"new york": "JFK", "san francisco": "SFO", "boston": "BOS", "washington dc": "DCA", "chicago": "ORD"}
WEEKDAYS = {name.lower(): index for index, name in enumerate(("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))}
DATE_MARKER = r"(?:day after tomorrow|today|tomorrow|next\s+\w+day|this\s+\w+day|(?:on\s+)?(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?|(?:on\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|in\s+\d+\s+days?|20\d{2}[-/]\d{1,2}[-/]\d{1,2})"


def _date_from_text(text: str, origin_timezone: str, now: datetime) -> tuple[date | None, str, str | None]:
    local_now = now.astimezone(ZoneInfo(origin_timezone))
    lowered = text.lower()
    if "day after tomorrow" in lowered:
        return local_now.date() + timedelta(days=2), "day after tomorrow", None
    in_days = re.search(r"\bin\s+(\d+)\s+days?\b", lowered)
    if in_days:
        return local_now.date() + timedelta(days=int(in_days.group(1))), in_days.group(0), None
    if "today" in lowered:
        return local_now.date(), "today", None
    if "tomorrow" in lowered:
        return local_now.date() + timedelta(days=1), "tomorrow", None
    aliases = {**WEEKDAYS, "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    for weekday, number in aliases.items():
        if weekday in lowered:
            delta = (number - local_now.weekday()) % 7
            if re.search(rf"\bnext\s+{weekday}", lowered):
                delta = delta or 7
            if re.search(rf"\bthis\s+{weekday}", lowered) and delta == 0 and local_now.hour >= 18:
                delta = 7
            if delta == 0 and local_now.hour >= 18:
                delta = 7
            return local_now.date() + timedelta(days=delta), weekday, None
    match = re.search(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))), "explicit", None
        except ValueError:
            return None, "", "invalid calendar date"
    return None, "", "missing departure date"


def _window(text: str) -> str:
    lowered = text.lower()
    for value in ("morning", "afternoon", "evening"):
        if value in lowered:
            return value
    return "whole_day"


def _split_route(text: str) -> tuple[str, str] | None:
    match = re.search(rf"(.+?)\s*(?:→|->|\bto\b|-)\s*(.+?)(?=,?\s+(?:{DATE_MARKER})\b|\s+(?:morning|afternoon|evening)\b|$)", text, re.IGNORECASE)
    return (match.group(1).strip(), match.group(2).strip()) if match else None


def _resolve_place(value: str) -> tuple[str, str | None, tuple[float, float] | None, str, str | None]:
    normalized_value = value.strip()
    normalized_value = re.sub(r",\s*[A-Za-z]{2}$", "", normalized_value)
    normalized = ALIASES.get(normalized_value.lower(), normalized_value)
    airport = get(normalized.upper())
    if airport:
        return airport["municipality"], airport["iata"], (float(airport["lat"]), float(airport["lon"])), airport.get("timezone") or "America/New_York", None
    known = KNOWN_CITIES.get(normalized.lower())
    if known:
        return normalized, None, known[0], known[1], None
    result = geocode(normalized)
    if not result.ok or not result.data:
        reference = by_municipality(normalized)
        if reference:
            return reference["municipality"], reference.get("iata"), (float(reference["lat"]), float(reference["lon"])), reference.get("timezone") or "America/New_York", None
        return normalized, None, None, "America/New_York", result.error or "unknown US place"
    data = result.data
    return normalized, None, (float(data["latitude"]), float(data["longitude"])), str(data.get("timezone") or "America/New_York"), None


def _trip_from_resolved(
    origin_text: str,
    destination_text: str,
    departure_date: date,
    departure_window: str,
    now: datetime,
    origin: tuple[str, str | None, tuple[float, float] | None, str, str | None],
    destination: tuple[str, str | None, tuple[float, float] | None, str, str | None],
) -> TripRequest | dict[str, str]:
    origin_city, origin_airport, origin_latlon, origin_tz, origin_error = origin
    destination_city, destination_airport, destination_latlon, destination_tz, destination_error = destination
    if origin_error or destination_error:
        return {"error": origin_error or destination_error or "Could not resolve a US place"}
    origin_city_latlon = origin_latlon
    destination_city_latlon = destination_latlon
    if origin_airport is None and origin_latlon:
        nearby = airports_near(*origin_latlon, 100)
        if nearby:
            origin_airport = str(nearby[0]["iata"])
            origin_latlon = (float(nearby[0]["lat"]), float(nearby[0]["lon"]))
    if destination_airport is None and destination_latlon:
        nearby = airports_near(*destination_latlon, 100)
        if nearby:
            destination_airport = str(nearby[0]["iata"])
            destination_latlon = (float(nearby[0]["lat"]), float(nearby[0]["lon"]))
    if origin_airport is None or destination_airport is None:
        return {"error": "Could not resolve a metro airport for the origin or destination"}
    if departure_date < now.astimezone(ZoneInfo(origin_tz)).date():
        return {"error": "departure date is in the past (origin local date)"}
    explicit_origin = get(origin_text.upper()) is not None
    explicit_destination = get(destination_text.upper()) is not None
    if not explicit_origin and not explicit_destination and origin_latlon and destination_latlon:
        origin_candidates = airports_near(*origin_latlon, 70)
        destination_candidates = airports_near(*destination_latlon, 80)
        pairs = []
        for origin_candidate in origin_candidates:
            for destination_candidate in destination_candidates:
                rate = route_base_rate(origin_candidate.get("iata", ""), destination_candidate.get("iata", ""), departure_date.month)
                if rate:
                    pairs.append((rate.get("flights", 0), origin_candidate, destination_candidate))
        if pairs:
            _, selected_origin, selected_destination = max(pairs, key=lambda item: item[0])
            origin_airport = str(selected_origin["iata"])
            origin_latlon = (float(selected_origin["lat"]), float(selected_origin["lon"]))
            destination_airport = str(selected_destination["iata"])
            destination_latlon = (float(selected_destination["lat"]), float(selected_destination["lon"]))
    return TripRequest(
        origin_text=origin_text,
        destination_text=destination_text,
        departure_date=departure_date,
        departure_window=departure_window,
        origin_city=origin_city,
        destination_city=destination_city,
        origin_airport=origin_airport,
        destination_airport=destination_airport,
        origin_latlon=origin_latlon,
        destination_latlon=destination_latlon,
        origin_city_latlon=origin_city_latlon,
        destination_city_latlon=destination_city_latlon,
        origin_timezone=origin_tz,
        destination_timezone=destination_tz,
    )


def _metro_airport_error(
    origin: tuple[str, str | None, tuple[float, float] | None, str, str | None],
    destination: tuple[str, str | None, tuple[float, float] | None, str, str | None],
) -> str | None:
    """Return an error when either endpoint has no airport within 100 km."""
    for _, airport, latlon, _, place_error in (origin, destination):
        if place_error:
            return place_error
        if airport is None and latlon:
            if not airports_near(*latlon, 100):
                return "Could not resolve a metro airport for the origin or destination"
    return None


def build_trip(
    origin_text: str,
    destination_text: str,
    departure_date: date,
    departure_window: str = "whole_day",
    now: datetime | None = None,
) -> TripRequest | dict[str, str]:
    """Resolve two place names and build a TripRequest for a known date/window."""
    now = now or datetime.now().astimezone()
    origin = _resolve_place(origin_text)
    destination = _resolve_place(destination_text)
    return _trip_from_resolved(origin_text, destination_text, departure_date, departure_window, now, origin, destination)


def parse(text: str, now: datetime | None = None) -> TripRequest | dict[str, str]:
    now = now or datetime.now().astimezone()
    route = _split_route(text)
    if not route:
        return {"error": "Could not parse an origin and destination"}
    origin_text, destination_text = route
    origin = _resolve_place(origin_text)
    destination = _resolve_place(destination_text)
    origin_error = origin[4]
    destination_error = destination[4]
    if origin_error or destination_error:
        return {"error": origin_error or destination_error or "Could not resolve a US place"}
    metro_error = _metro_airport_error(origin, destination)
    if metro_error:
        return {"error": metro_error}
    departure_date, _, date_error = _date_from_text(text, origin[3], now)
    if date_error or departure_date is None:
        return {"error": date_error or "Could not resolve departure date"}
    return _trip_from_resolved(origin_text, destination_text, departure_date, _window(text), now, origin, destination)
