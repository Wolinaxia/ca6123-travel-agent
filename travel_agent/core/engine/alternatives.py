"""Alternative mode evaluation and feasibility gates."""

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any

from ..reference import amtrak
from ..reference.airports import airports_near, get
from ..reference.bts import route_base_rate
from .models import Alternative, Policy, SourceInputs, TripRequest


def _distance_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lat1, lon1 = origin
    lat2, lon2 = destination
    return math.sqrt((lat2 - lat1) ** 2 + ((lon2 - lon1) * 0.8) ** 2) * 111


def build_alternatives(trip: TripRequest, inputs: SourceInputs, block_minutes: float, policy: Policy, days: int, primary_level: str) -> list[Alternative]:
    flying_minutes = policy.airport_buffer_hours * 60 + policy.egress_hours * 60 + block_minutes
    alternatives = [Alternative(mode="bus", name="Bus", not_practical=True, reason="no open national data (Greyhound/FlixBus have no public API)")]
    origin_alternates = airports_near(*(trip.origin_city_latlon or trip.origin_latlon or (0, 0)), policy.origin_metro_radius_km)
    destination_alternates = airports_near(*(trip.destination_city_latlon or trip.destination_latlon or (0, 0)), policy.destination_metro_radius_km)
    start_hour, end_hour = trip.window_hours
    departure = datetime.combine(trip.departure_date, datetime.min.time(), tzinfo=ZoneInfo(trip.origin_timezone)).replace(hour=start_hour)
    departure_end = datetime.combine(trip.departure_date, datetime.min.time(), tzinfo=ZoneInfo(trip.origin_timezone)) + timedelta(days=1 if end_hour == 24 else 0, hours=end_hour % 24)
    for origin in origin_alternates:
        for destination in destination_alternates:
            if origin.get("iata") == trip.origin_airport and destination.get("iata") == trip.destination_airport:
                continue
            route = route_base_rate(str(origin["iata"]), str(destination["iata"]), trip.departure_date.month)
            if not route:
                continue
            blocked = False
            delayed = False
            if inputs.faa and inputs.faa.ok and inputs.faa.data:
                blocked = any(item.airport in {origin["iata"], destination["iata"]} and item.affects_commercial and (item.start is None or item.start < departure_end) and (item.reopen is None or item.reopen > departure) for item in inputs.faa.data.closures)
                delayed = any(item.airport == destination["iata"] for item in inputs.faa.data.ground_delays)
            weather_bad = False
            for result in (inputs.taf_origin, inputs.taf_destination):
                if result and result.ok and result.data:
                    origin_icao = (get(str(origin["iata"])) or {}).get("icao", str(origin["iata"]))
                    destination_icao = (get(str(destination["iata"])) or {}).get("icao", str(destination["iata"]))
                    weather = result.data.get(origin_icao) or result.data.get(destination_icao)
                    weather_bad = weather_bad or bool(weather and weather.flight_category in {"IFR", "LIFR"})
            destination_hazard = any(result and result.ok and result.data for result in (inputs.nws_origin_alerts, inputs.nws_destination_alerts))
            better = not blocked and not delayed and not weather_bad and not destination_hazard and primary_level in {"high", "severe"} and route.get("dep_del15", 1) < 0.30
            if origin["iata"] == trip.origin_airport:
                name = f"{origin['iata']} → {destination['iata']}"
            elif destination["iata"] == trip.destination_airport:
                name = f"{origin['iata']} → {destination['iata']}"
            else:
                name = f"{origin['iata']} → {destination['iata']}"
            reason = None if better else "Same destination delay program applies" if delayed else "Alternative route is not strictly better"
            alternatives.append(Alternative(mode="alternate_airport", name=name, feasible=better, not_practical=not better, better_than_primary=better, risk_level="high" if destination_hazard else ("elevated" if not better else "low"), reason=reason))
    if inputs.osrm_drive and inputs.osrm_drive.ok and inputs.osrm_drive.data:
        drive_minutes = inputs.osrm_drive.data.duration_s / 60
    else:
        drive_minutes = _distance_km(trip.origin_city_latlon or trip.origin_latlon or (0, 0), trip.destination_city_latlon or trip.destination_latlon or (0, 0)) / policy.drive_fallback_kmh * 60
    access_minutes = inputs.osrm_access.data.duration_s / 60 if inputs.osrm_access and inputs.osrm_access.ok and inputs.osrm_access.data else policy.access_fallback_minutes
    egress_minutes = inputs.osrm_egress.data.duration_s / 60 if inputs.osrm_egress and inputs.osrm_egress.ok and inputs.osrm_egress.data else policy.access_fallback_minutes
    flying_door_to_door = access_minutes + policy.airport_buffer_hours * 60 + block_minutes + policy.egress_hours * 60
    extra = drive_minutes - flying_door_to_door
    route_hazard = any(result and result.ok and result.data for result in (inputs.nws_origin_alerts, inputs.nws_destination_alerts))
    alternatives.append(Alternative(mode="drive", name="Drive", duration_min=drive_minutes, extra_minutes=extra, feasible=extra <= policy.max_extra_hours * 60 and not route_hazard, not_practical=extra > policy.max_extra_hours * 60 or route_hazard, reason="origin/destination safety alert applies to all modes" if route_hazard else (None if extra <= policy.max_extra_hours * 60 else f"+{extra / 60:.1f}h vs flying"), risk_level="high" if route_hazard else "low"))
    if alternatives[-1].risk_level == "low" and days <= 7:
        if any(result.ok and result.data for result in inputs.drive_alerts):
            alternatives[-1].risk_level = "elevated"
    services = amtrak.direct_services(trip.origin_city_latlon or trip.origin_latlon or (0, 0), trip.destination_city_latlon or trip.destination_latlon or (0, 0))
    if services:
        rail = services[0]
        rail_minutes = float(rail["min_duration_min"]) + policy.rail_station_access_minutes
        rail_extra = rail_minutes - flying_door_to_door
        rail_names = rail["route_names"]
        rail_risk = "high" if route_hazard else "low"
        rail_reason = None if rail_extra <= policy.max_extra_hours * 60 else f"+{rail_extra / 60:.1f}h vs flying"
        if days <= 2 and inputs.amtraker and inputs.amtraker.ok:
            matching_delays = [train.delay_minutes for train in inputs.amtraker.data or [] if train.route in rail_names and train.delay_minutes is not None and train.delay_minutes >= policy.amtraker_delay_minutes]
            if matching_delays:
                rail_risk = "elevated"
                rail_reason = f"matching train delay {max(matching_delays)} minutes"
        from_code = rail["from_station"]["id"]
        to_code = rail["to_station"]["id"]
        alternatives.append(Alternative(mode="rail", name=f"Amtrak {from_code} → {to_code} · {rail.get('trains_per_day', 0)} trains/day · ~{round(rail_minutes / 60)} h", route_names=rail_names, duration_min=rail_minutes, extra_minutes=rail_extra, feasible=rail_extra <= policy.max_extra_hours * 60 and not route_hazard, not_practical=rail_extra > policy.max_extra_hours * 60 or route_hazard, reason="origin/destination safety alert applies to all modes" if route_hazard else rail_reason, risk_level=rail_risk))
    else:
        alternatives.append(Alternative(mode="rail", name="Amtrak", reason="No direct service"))
    return alternatives
