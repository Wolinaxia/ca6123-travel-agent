"""Pure, horizon-aware deterministic assessment."""

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from ..models import AirportWeather, FaaNasData, NwsAlert
from ..reference.bts import route_base_rate, block_minutes
from ..reference.airports import get
from ..reference.holidays import is_peak_travel
from .alternatives import build_alternatives
from .models import Assessment, Evidence, LegAssessment, Policy, SourceInputs, SourceStatus, TripRequest
from .weather_rules import open_meteo_level


LEVELS = {"low": 0, "elevated": 1, "high": 2, "severe": 3, "info": -1}


def _max_level(current: str, candidate: str) -> str:
    return candidate if LEVELS.get(candidate, -1) > LEVELS.get(current, -1) else current


def _overlaps(start: datetime | None, end: datetime | None, window_start: datetime, window_end: datetime) -> bool:
    return (start is None or start < window_end) and (end is None or end > window_start)


def _window(trip: TripRequest, block: float) -> tuple[datetime, datetime, datetime, datetime]:
    start_hour, end_hour = trip.window_hours
    origin_zone = ZoneInfo(trip.origin_timezone)
    destination_zone = ZoneInfo(trip.destination_timezone)
    departure = datetime.combine(trip.departure_date, datetime.min.time(), tzinfo=origin_zone).replace(hour=start_hour)
    departure_end = datetime.combine(trip.departure_date, datetime.min.time(), tzinfo=origin_zone).replace(hour=end_hour % 24)
    if end_hour == 24:
        departure_end += timedelta(days=1)
    arrival = departure + timedelta(minutes=block)
    return departure, departure_end, arrival.astimezone(destination_zone), (departure_end + timedelta(minutes=block)).astimezone(destination_zone)


def _add(evidence: list[Evidence], leg: str, category: str, level: str, rule: str, title: str, detail: str, source: str, result: Any = None, valid_from: datetime | None = None, valid_to: datetime | None = None) -> None:
    evidence.append(Evidence(id=f"E{len(evidence) + 1:03d}", leg=leg, category=category, level=level, rule_id=rule, title=title, detail=detail, source=source, url=getattr(result, "url", "") if result else "", fetched_at=getattr(result, "fetched_at", None), valid_from=valid_from, valid_to=valid_to, is_snapshot="snapshot" in (getattr(result, "url", "") or "")))


def _weather_level(evidence: list[Evidence], weather: AirportWeather, result: Any, start: datetime, end: datetime, policy: Policy, current: bool = True) -> str:
    level = "low"
    if current and weather.flight_category == "LIFR":
        _add(evidence, "flight", "delay", "high", "TAF_LIFR", "LIFR airport category", f"{weather.icao} is LIFR", "aviation_wx", result, start, end)
        level = "high"
    elif current and weather.flight_category == "IFR":
        _add(evidence, "flight", "delay", "elevated", "TAF_IFR", "IFR airport category", f"{weather.icao} is IFR", "aviation_wx", result, start, end)
        level = "elevated"
    for period in weather.taf:
        if period.start and period.end and (period.end < start or period.start > end):
            continue
        category = None
        if period.ceiling_ft is not None and period.ceiling_ft < policy.taf_lifr_ceiling_ft or period.visibility_sm is not None and period.visibility_sm < policy.taf_lifr_visibility_sm:
            category = ("LIFR", "high", "TAF_LIFR")
        elif period.ceiling_ft is not None and period.ceiling_ft < policy.taf_ifr_ceiling_ft or period.visibility_sm is not None and period.visibility_sm < policy.taf_ifr_visibility_sm:
            category = ("IFR", "elevated", "TAF_IFR")
        if category:
            name, candidate, rule = category
            _add(evidence, "flight", "delay", candidate, rule, f"TAF {name} conditions", f"ceiling {period.ceiling_ft or 'unknown'} ft, visibility {period.visibility_sm or 'unknown'} sm", "aviation_wx", result, period.start, period.end)
            level = _max_level(level, candidate)
        if period.gust_kt is not None:
            rule_level = "high" if period.gust_kt >= policy.gust_high_kt else "elevated" if period.gust_kt >= policy.gust_elevated_kt else "low"
            if rule_level != "low":
                rule = "TAF_GUST_45" if rule_level == "high" else "TAF_GUST_35"
                _add(evidence, "flight", "delay", rule_level, rule, "TAF wind gusts", f"{period.gust_kt} kt", "aviation_wx", result, period.start, period.end)
                level = _max_level(level, rule_level)
        for code, rule, candidate, title in (("TS", "TAF_TS", "elevated", "TAF thunderstorms"), ("SN", "TAF_SNOW", "high", "TAF snow"), ("FZRA", "TAF_FZRA", "high", "TAF freezing rain")):
            if code in period.weather_codes:
                _add(evidence, "flight", "delay", candidate, rule, title, code, "aviation_wx", result, period.start, period.end)
                level = _max_level(level, candidate)
    return level


def _alert_level(alert: NwsAlert, policy: Policy) -> tuple[str, str, bool]:
    mapped = policy.nws_events.get(alert.event)
    if mapped:
        category, level = mapped
        if (alert.severity or "").lower() == "extreme":
            level = "severe"
        tier = "SEVERE" if level == "severe" else "WARNING" if level == "high" else "ADVISORY"
        return level, f"NWS_{'SAFETY' if category == 'safety' else 'ALERT'}_{tier}", category == "safety"
    text = alert.event.lower()
    safety = any(word in text for word in policy.safety_keywords)
    if text.endswith(" watch"):
        return "elevated", "NWS_SAFETY_ADVISORY" if safety else "NWS_ALERT_ADVISORY", safety
    if alert.event in policy.nws_severe_events or (alert.severity or "").lower() == "extreme":
        return "severe", "NWS_SAFETY_SEVERE" if safety else "NWS_ALERT_SEVERE", safety
    if alert.event in policy.nws_warning_events or (alert.severity or "").lower() in {"severe", "extreme"} or alert.event.lower().endswith("warning"):
        return "high", "NWS_SAFETY_WARNING" if safety else "NWS_ALERT_WARNING", safety
    return "elevated", "NWS_SAFETY_ADVISORY" if safety else "NWS_ALERT_ADVISORY", safety


def _alerts(evidence: list[Evidence], alerts: list[NwsAlert], result: Any, start: datetime, end: datetime, policy: Policy, leg: str, filtered: list[dict[str, Any]]) -> tuple[str, bool]:
    level = "low"
    safety = False
    for alert in alerts:
        if alert.ends and alert.ends < start or alert.onset and alert.onset > end:
            continue
        if alert.event.lower().endswith("statement"):
            filtered.append({"source": "NWS", "count": 1, "reason": "informational statement, below advisory level", "examples": [alert.event]})
            continue
        if alert.event not in policy.nws_events and not any(word in alert.event.lower() for word in policy.travel_alert_keywords):
            filtered.append({"source": "NWS", "count": 1, "reason": "not travel-relevant", "examples": [alert.event]})
            continue
        candidate, rule, is_safety = _alert_level(alert, policy)
        _add(evidence, leg, "safety" if is_safety else "delay", candidate, rule, alert.event, alert.headline or alert.area_desc or alert.event, "nws", result, alert.onset, alert.ends or alert.expires)
        level = _max_level(level, candidate)
        safety = safety or is_safety
    return level, safety


def _forecast(evidence: list[Evidence], periods: list[Any], result: Any, start: datetime, end: datetime) -> str:
    level = "low"
    for period in periods:
        if period.start > end or period.end < start:
            continue
        text = f"{period.short_forecast} {period.detailed_forecast}".lower()
        if "snow" in text or "freezing rain" in text:
            _add(evidence, "flight", "delay", "high", "NWS_FCST_WINTER", "NWS winter forecast", period.detailed_forecast, "nws", result, period.start, period.end)
            level = _max_level(level, "high")
        elif "thunder" in text or "storm" in text:
            _add(evidence, "flight", "delay", "elevated", "NWS_FCST_THUNDER", "NWS thunderstorm forecast", period.detailed_forecast, "nws", result, period.start, period.end)
            level = _max_level(level, "elevated")
    return level


def _source_status(inputs: SourceInputs) -> list[SourceStatus]:
    statuses = []
    for name, value in inputs.__dict__.items():
        if hasattr(value, "ok"):
            error = value.error
            state = "ok" if value.ok else "not_captured" if error and ("replay snapshot miss" in error or "not captured" in error) else "failed"
            statuses.append(SourceStatus(source=name, ok=value.ok, state=state, error=error, latency_ms=inputs.source_latency.get(name)))
    return statuses


def evaluate(trip: TripRequest, inputs: SourceInputs, now: datetime, policy: Policy | None = None) -> Assessment:
    policy = policy or Policy()
    days = (trip.departure_date - now.astimezone(ZoneInfo(trip.origin_timezone)).date()).days
    block = block_minutes(trip.origin_airport, trip.destination_airport) if trip.origin_airport and trip.destination_airport else None
    block = block or 180.0
    departure, departure_end, arrival, arrival_end = _window(trip, block)
    evidence: list[Evidence] = []
    filtered: list[dict[str, Any] | str] = []
    levels = {"access": "low", "flight": "low", "egress": "low"}
    duty_of_care = False
    airports = {trip.origin_airport, trip.destination_airport}
    if inputs.faa and inputs.faa.ok:
        faa: FaaNasData = inputs.faa.data
        ignored = []
        for delay in faa.ground_delays:
            if delay.airport not in airports:
                ignored.append(delay.airport)
                continue
            if days > 1:
                continue
            if days != 0:
                _add(evidence, "flight", "info", "info", "FAA_GDP_CONTEXT", "FAA program outside trip day", f"Active today at {delay.airport} ({delay.reason}); programs are issued for today's hours — not rated for {trip.departure_date}.", "faa_nas", inputs.faa)
                continue
            if departure_end <= now:
                _add(evidence, "flight", "info", "info", "FAA_GDP_CONTEXT", "FAA program window ended", f"Active today at {delay.airport} but the requested departure window ended before the current time.", "faa_nas", inputs.faa)
                continue
            if delay.airport == trip.origin_airport:
                _add(evidence, "flight", "info", "info", "FAA_GDP_CONTEXT", "Origin FAA program context", f"Active today at {delay.airport} ({delay.reason}); origin GDPs hold inbound flights, not departures.", "faa_nas", inputs.faa)
                continue
            candidate = "high" if delay.avg_minutes >= policy.gdp_high_avg_minutes else "elevated" if delay.avg_minutes >= policy.gdp_elevated_avg_minutes else "low"
            if candidate != "low":
                rule = "FAA_GDP_AVG_60" if candidate == "high" else "FAA_GDP_AVG_15"
                _add(evidence, "flight", "delay", candidate, rule, "FAA ground delay program", f"Ground delay program at {delay.airport} ({delay.reason}): avg {delay.avg_minutes} min, max {delay.max_minutes // 60} h {delay.max_minutes % 60:02d} min. Programs at the destination hold {delay.airport}-bound flights at the departure airport ({trip.origin_airport}).", "faa_nas", inputs.faa, departure, departure_end)
                levels["flight"] = _max_level(levels["flight"], candidate)
        for stop in faa.ground_stops:
            if days > 1:
                continue
            if days != 0:
                _add(evidence, "flight", "info", "info", "FAA_GDP_CONTEXT", "FAA ground stop outside trip day", f"Active today at {stop.airport}; ground stops are not rated for {trip.departure_date}.", "faa_nas", inputs.faa)
                continue
            window_start, window_end = (departure, departure_end) if stop.airport == trip.origin_airport else (arrival, arrival_end)
            if stop.airport in airports and _overlaps(None, stop.end, window_start, window_end):
                _add(evidence, "flight", "delay", "high", "FAA_GROUND_STOP", "FAA ground stop", f"{stop.airport}: {stop.reason}", "faa_nas", inputs.faa, departure, stop.end)
                levels["flight"] = _max_level(levels["flight"], "high")
            else:
                ignored.append(stop.airport)
        for closure in faa.closures:
            if closure.airport not in airports:
                ignored.append(closure.airport)
                continue
            if not closure.affects_commercial:
                filtered.append({"source": "FAA", "count": 1, "reason": "GA-only commercial closure filtered", "examples": [closure.airport]})
            else:
                window_start, window_end = (departure, departure_end) if closure.airport == trip.origin_airport else (arrival, arrival_end)
                if not _overlaps(closure.start, closure.reopen, window_start, window_end):
                    filtered.append({"source": "FAA", "count": 1, "reason": "closure does not overlap trip window", "examples": [closure.airport]})
                    continue
                _add(evidence, "flight", "delay", "severe", "FAA_CLOSURE", "Commercial airport closure", closure.reason, "faa_nas", inputs.faa, closure.start, closure.reopen)
                levels["flight"] = "severe"
        for delay in faa.arrival_departure_delays if days == 0 else []:
            applies = delay.airport == trip.origin_airport and "departure" in delay.kind.lower() or delay.airport == trip.destination_airport and "arrival" in delay.kind.lower()
            if not applies or delay.max_minutes < 15:
                continue
            candidate = "high" if delay.max_minutes >= 60 else "elevated"
            _add(evidence, "flight", "delay", candidate, "FAA_ARRDEP_DELAY", "FAA arrival/departure delay", f"{delay.airport}: {delay.min_minutes}–{delay.max_minutes} min, {delay.reason}", "faa_nas", inputs.faa, departure, arrival_end)
        if ignored:
            filtered.append({"source": "FAA", "count": len(ignored), "reason": "notice does not concern trip airports", "examples": ignored[:3]})
        if days > 1:
            filtered.append({"source": "FAA", "reason": "source not used for this date (outside horizon)", "examples": []})
    elif inputs.faa:
        filtered.append({"source": "FAA", "reason": "source not used for this date (outside horizon)", "examples": []})
    weather_allowed = days <= 1
    for result, airport, start, end in ((inputs.taf_origin, trip.origin_airport, departure, departure_end), (inputs.taf_destination, trip.destination_airport, arrival, arrival_end)):
        airport_ref = get(airport) if airport else None
        icao = airport_ref.get("icao") if airport_ref else airport
        weather = result.data.get(icao) if result and result.ok and result.data and icao else None
        if weather_allowed and result and result.ok and weather:
            current = start <= now.astimezone(start.tzinfo) <= min(end, start + timedelta(hours=3))
            levels["flight"] = _max_level(levels["flight"], _weather_level(evidence, weather, result, start, end, policy, current=current))
        elif result:
            filtered.append({"source": "TAF/METAR", "reason": "source not used for this date" if not weather_allowed else "airport data unavailable", "examples": [airport or "airport"]})
    if days <= 7:
        for result, start, end, leg in ((inputs.nws_origin_alerts, departure, departure_end, "access"), (inputs.nws_destination_alerts, arrival, arrival_end, "egress")):
            if result and result.ok and result.data:
                level, safety = _alerts(evidence, result.data, result, start, end, policy, leg, filtered)
                levels[leg] = _max_level(levels[leg], level)
                duty_of_care = duty_of_care or safety
        for result, start, end in ((inputs.nws_origin_forecast, departure, departure_end), (inputs.nws_destination_forecast, arrival, arrival_end)):
            if result and result.ok and result.data:
                levels["flight"] = _max_level(levels["flight"], _forecast(evidence, result.data, result, start, end))
    else:
        filtered.append({"source": "NWS", "reason": "source not used for this date", "examples": []})
    if 8 <= days <= 16:
        for result, leg in ((inputs.open_meteo_origin, "access"), (inputs.open_meteo_destination, "egress")):
            if result and result.ok and result.data:
                level = open_meteo_level(result.data)
                if level != "low":
                    rule = "OM_SNOW" if level == "high" else "OM_THUNDER"
                    _add(evidence, leg, "delay", level, rule, "Open-Meteo weather", f"{level} daily weather signal", "open_meteo", result)
                    levels[leg] = _max_level(levels[leg], level)
    elif inputs.open_meteo_origin or inputs.open_meteo_destination:
        filtered.append({"source": "Open-Meteo", "reason": "source not used for this date", "examples": []})
    if days == 0:
        for result, leg in ((inputs.tomtom_origin, "access"), (inputs.tomtom_destination, "egress")):
            if result and result.ok:
                seen: set[tuple[str, str]] = set()
                for incident in result.data or []:
                    if incident.distance_km is not None and incident.distance_km > policy.tomtom_radius_km:
                        continue
                    if not incident.road_numbers:
                        continue
                    if (incident.magnitude_of_delay or 0) < policy.tomtom_magnitude and "clos" not in (incident.description or "").lower():
                        continue
                    endpoints = tuple(sorted((incident.from_location or "", incident.to_location or "")))
                    key = (incident.road_numbers[0], endpoints)
                    if key in seen:
                        continue
                    seen.add(key)
                    level = "high" if "clos" in (incident.description or "").lower() else "elevated"
                    _add(evidence, leg, "delay", level, "TOMTOM_MAJOR", "Major traffic incident", incident.description or "Traffic disruption", "tomtom", result)
                    levels[leg] = _max_level(levels[leg], level)
        if inputs.mta and inputs.mta.ok:
            mta_legs = []
            if trip.origin_airport in {"JFK", "LGA", "EWR"}:
                mta_legs.append("access")
            if trip.destination_airport in {"JFK", "LGA", "EWR"}:
                mta_legs.append("egress")
            for alert in inputs.mta.data or []:
                if any(route in {"A", "E", "J", "Z"} for route in alert.affected_routes):
                    for leg in mta_legs:
                        _add(evidence, leg, "delay", "elevated", "MTA_DISRUPTION", "Airport transit disruption", alert.header_text, "mta", inputs.mta)
                        levels[leg] = _max_level(levels[leg], "elevated")
    else:
        filtered.append({"source": "TomTom/MTA", "reason": "source not used for this date", "examples": []})
    if 0 <= days <= 1 and inputs.amtraker:
        filtered.append({"source": "Amtraker", "count": 1, "reason": "attached only to matching rail alternatives", "examples": []})
    statuses = _source_status(inputs)
    core_names = {"faa", "taf_origin", "taf_destination", "nws_origin_alerts", "nws_destination_alerts"}
    status_by_name = {status.source: status for status in statuses}
    core_ok = all(name in status_by_name and (status_by_name[name].ok or (status_by_name[name].error or "").startswith("replay snapshot miss")) for name in core_names)
    confidence = "high" if days <= 1 and core_ok else "medium" if days <= 7 else "low"
    if sum(not status_by_name[name].ok and not (status_by_name[name].error or "").startswith("replay snapshot miss") for name in core_names if name in status_by_name) >= 2:
        confidence = "low"
    if inputs.faa and not inputs.faa.ok:
        _add(evidence, "flight", "info", "info", "SOURCE_UNAVAILABLE", "faa unavailable", inputs.faa.error or "source failed", "faa", inputs.faa)
    base = route_base_rate(trip.origin_airport, trip.destination_airport, trip.departure_date.month) if trip.origin_airport and trip.destination_airport else None
    if base and (base.get("dep_del15", 0) >= policy.bts_delay_threshold or base.get("cancel_rate", 0) >= policy.bts_cancel_threshold):
        _add(evidence, "flight", "delay", "elevated", "BTS_BASE_CAP", "Historical route disruption rate", f"Departure delay {base.get('dep_del15', 0):.1%} ({base.get('basis', 'monthly')})", "bts")
        levels["flight"] = _max_level(levels["flight"], "elevated")
    if is_peak_travel(trip.departure_date):
        _add(evidence, "flight", "delay", "elevated", "HOLIDAY_PEAK", "Peak travel window", str(trip.departure_date), "holidays")
        levels["flight"] = _max_level(levels["flight"], "elevated")
    for leg_name in levels:
        leg_evidence = [item for item in evidence if item.leg == leg_name and item.level != "info"]
        levels[leg_name] = max((item.level for item in leg_evidence), key=lambda value: LEVELS[value], default="low")
    alternatives = build_alternatives(trip, inputs, block, policy, days, levels["flight"])
    worst = max(levels.values(), key=lambda value: LEVELS[value])
    feasible = next((item for item in alternatives if item.feasible), None)
    action = "no_action" if worst == "low" else "monitor" if worst == "elevated" else "prepare"
    action_detail = None
    if worst == "severe" or (worst == "high" and confidence == "high"):
        if feasible:
            action = "change_plan"
            action_detail = f"Take {feasible.name}"
        else:
            action = "change_plan" if worst == "severe" else "prepare"
            action_detail = "Change date/time — no practical alternative mode" if worst == "severe" else "no practical alternative; consider changing date"
    recheck = trip.departure_date - timedelta(days=7) if days > 16 else None
    if duty_of_care:
        action_detail = f"{action_detail}; duty of care" if action_detail else "duty of care"
    top = sorted((item for item in evidence if item.level != "info"), key=lambda item: LEVELS[item.level], reverse=True)[:3]
    legs = [LegAssessment(name=name, level=level, evidence_ids=[item.id for item in evidence if item.leg == name]) for name, level in levels.items()]
    driving_leg = None if worst == "low" else next((leg.name for leg in legs if leg.level == worst and leg.evidence_ids), None)
    news = []
    news_fallback_reason = None
    if inputs.news and inputs.news.ok:
        us_terms = {trip.origin_city, trip.destination_city, trip.origin_airport, trip.destination_airport, "united states", "new york", "san francisco", "washington", "boston"}
        disruption_terms = {"delay", "cancel", "strike", "outage", "disruption", "atc"}
        cutoff = now.astimezone(timezone.utc) - timedelta(hours=policy.news_max_age_hours)
        candidates = []
        stale_count = 0
        stale_examples = []
        for article in inputs.news.data or []:
            published = article.published_at
            if published is None or (published.replace(tzinfo=timezone.utc) if published.tzinfo is None else published.astimezone(timezone.utc)) < cutoff:
                stale_count += 1
                if len(stale_examples) < 3:
                    stale_examples.append(article.title)
                continue
            title = article.title.lower()
            if any(term and term.lower() in title for term in us_terms) and any(term in title for term in disruption_terms):
                candidates.append(article)
        news = candidates
        if stale_count:
            filtered.append({"source": "News", "count": stale_count, "reason": "older than 72 h / undated", "examples": stale_examples})
    return Assessment(trip=trip, level=worst, confidence=confidence, action=action, action_detail=action_detail, recheck_on=recheck, duty_of_care=duty_of_care, driving_leg=driving_leg, top_evidence_ids=[item.id for item in top], evidence=evidence, source_status=statuses, legs=legs, alternatives=alternatives, news=news, news_fallback_reason=news_fallback_reason, filtered=filtered)
