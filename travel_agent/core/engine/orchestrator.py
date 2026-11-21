"""Concurrent source orchestration around the pure evaluator."""

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
import time
from zoneinfo import ZoneInfo

from ..sources import amtraker, aviation_wx, faa_nas, mta, news, nws, open_meteo, osrm, tomtom
from ..http import http_mode, snapshot_dir
from ..models import SourceResult
from ..reference import amtrak
from ..reference.airports import airports_near
from .evaluate import evaluate
from .models import Assessment, Policy, SourceInputs, TripRequest


def assess(trip: TripRequest, now: datetime | None = None, policy: Policy | None = None, mode: str | None = None, snapshot: str | None = None) -> Assessment:
    now = now or datetime.now(timezone.utc)
    policy = policy or Policy()
    origin = trip.origin_city_latlon or trip.origin_latlon or (0.0, 0.0)
    destination = trip.destination_city_latlon or trip.destination_latlon or (0.0, 0.0)
    origin_airport = trip.origin_latlon or origin
    destination_airport = trip.destination_latlon or destination
    local_date = now.astimezone(ZoneInfo(trip.origin_timezone)).date()
    days = (trip.departure_date - local_date).days
    jobs: dict[str, callable] = {
        "faa": lambda: faa_nas.fetch(now=now),
        "taf_origin": lambda: aviation_wx.fetch([str(item["icao"]) for item in airports_near(*origin, 60) if item.get("icao")]),
        "taf_destination": lambda: aviation_wx.fetch([str(item["icao"]) for item in airports_near(*destination, 70) if item.get("icao")]),
        "nws_origin_alerts": lambda: nws.alerts(*origin),
        "nws_destination_alerts": lambda: nws.alerts(*destination),
        "nws_origin_forecast": lambda: nws.forecast(*origin),
        "nws_destination_forecast": lambda: nws.forecast(*destination),
        "news": lambda: news.search(f"{trip.origin_airport or trip.origin_city} {trip.destination_airport or trip.destination_city} flight disruption ATC airline"),
        "amtrak": lambda: amtrak.direct_services(origin, destination),
    }
    if days <= 1:
        jobs.update({
            "osrm_access": lambda: osrm.route(origin, origin_airport),
            "osrm_egress": lambda: osrm.route(destination_airport, destination),
            "osrm_drive": lambda: osrm.route(origin, destination),
            "amtraker": amtraker.trains,
        })
        if days == 0:
            jobs.update({
                "tomtom_origin": lambda: tomtom.incidents(*origin, policy.tomtom_radius_km),
                "tomtom_destination": lambda: tomtom.incidents(*destination, policy.tomtom_radius_km),
                "mta": mta.alerts,
            })
    elif days <= 7:
        jobs = {key: value for key, value in jobs.items() if not key.startswith("taf_") and not key.startswith("tomtom") and key not in {"mta", "amtraker"}}
        jobs["osrm_drive"] = lambda: osrm.route(origin, destination)
    elif days <= 16:
        jobs = {"open_meteo_origin": lambda: open_meteo.forecast(*origin), "open_meteo_destination": lambda: open_meteo.forecast(*destination), "news": jobs["news"], "amtrak": jobs["amtrak"]}
    else:
        jobs = {"news": jobs["news"], "amtrak": jobs["amtrak"]}
    mode_token = http_mode.set(mode) if mode else None
    snapshot_token = snapshot_dir.set(snapshot) if snapshot else None
    outputs = {}
    latencies = {}
    executor = ThreadPoolExecutor(max_workers=min(12, len(jobs)))
    submitted = time.monotonic()
    future_started = {}
    futures = {}
    def run_job(job):
        mode_token_inner = http_mode.set(mode) if mode else None
        snapshot_token_inner = snapshot_dir.set(snapshot) if snapshot else None
        try:
            return job()
        finally:
            if snapshot_token_inner:
                snapshot_dir.reset(snapshot_token_inner)
            if mode_token_inner:
                http_mode.reset(mode_token_inner)
    for name, job in jobs.items():
        future = executor.submit(run_job, job)
        futures[future] = name
        future_started[future] = submitted
    done, pending = wait(futures, timeout=1.5 if mode == "replay" else 8)
    for future in done:
            name = futures[future]
            try:
                outputs[name] = future.result(timeout=10)
                if name == "amtrak" and not isinstance(outputs[name], SourceResult):
                    outputs[name] = SourceResult(source="amtrak", ok=True, fetched_at=now, url="offline://amtrak", data=outputs[name])
            except Exception as exc:
                outputs[name] = SourceResult(source=name, ok=False, fetched_at=now, url="", error=str(exc))
            latencies[name] = (time.monotonic() - future_started[future]) * 1000
    for future in pending:
        name = futures[future]
        outputs[name] = SourceResult(source=name, ok=False, fetched_at=now, url="", error="source timeout")
        latencies[name] = 1500 if mode == "replay" else 8000
    executor.shutdown(wait=False, cancel_futures=True)
    if days <= 7 and outputs.get("osrm_drive") and outputs["osrm_drive"].ok and hasattr(outputs["osrm_drive"].data, "coordinates"):
        coordinates = outputs["osrm_drive"].data.coordinates
        step = max(1, len(coordinates) // 10)
        points = coordinates[::step][:10]
        sampler = ThreadPoolExecutor(max_workers=min(10, len(points)))
        sampled = [sampler.submit(run_job, lambda coordinate=coordinate: nws.alerts(coordinate[1], coordinate[0])) for coordinate in points]
        done, _ = wait(sampled, timeout=1.5)
        for future in done:
            try:
                outputs.setdefault("drive_alerts", []).append(future.result())
            except Exception:
                continue
        sampler.shutdown(wait=False, cancel_futures=True)
    source_inputs = SourceInputs(**outputs, source_latency=latencies)
    result = evaluate(trip, source_inputs, now, policy)
    if source_inputs.news and source_inputs.news.ok:
        candidates = list(result.news)[:policy.max_news_items]
        result.news = [item.model_copy(update={"extracted_by": "keywords"}) for item in candidates]
    if snapshot_token:
        snapshot_dir.reset(snapshot_token)
    if mode_token:
        http_mode.reset(mode_token)
    return result
