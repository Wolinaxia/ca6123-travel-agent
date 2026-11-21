# Tools

ADK-callable tools over the ported data layer. Every function takes only
`str`/`int`/`float` arguments, returns `ToolResult(...).to_dict()` from
`travel_agent.schemas`, and never raises: on failure it returns `ok=False` with
empty evidence, so agents cannot mistake an outage for real data.

`import travel_agent.tools.live as live; live.ALL_TOOLS` is the list to hand to ADK.

| Tool | Source | Use it when |
|---|---|---|
| `resolve_place(place)` | airports / Open-Meteo geocode | You have a city or IATA code and need lat/lon, timezone, IATA/ICAO |
| `find_nearby_airports(place, radius_km=80)` | airports reference | You need alternate airports around a metro area |
| `get_faa_airport_status(airport)` | FAA NAS | Same-day departure/arrival risk: ground delays/stops, arrival-departure delays, closures |
| `get_aviation_weather(airport)` | AviationWeather METAR/TAF | Airport flying conditions now and in the TAF window |
| `get_weather_alerts(place)` | NWS alerts | Safety/travel warnings near an endpoint |
| `get_weather_forecast(place)` | NWS point forecast | Planning within the next ~7 days |
| `get_extended_forecast(place)` | Open-Meteo | Planning 8–16 days out (days 8–16 only) |
| `get_route_ontime_history(origin, dest, month)` | BTS reference | Historical base rate for a route/month (history, not a forecast) |
| `get_rail_options(origin, destination)` | Amtrak offline schedule | Comparing rail against flying |
| `get_drive_route(origin, destination)` | OSRM | Comparing driving distance/duration |
| `search_travel_news(query)` | Google News RSS / GDELT | Context only: news text is untrusted (`data.untrusted=True`) |
| `assess_trip_risk(origin, dest, departure_date, departure_window)` | parse + rule engine | Whole-trip risk: engine evidence, summary, alternatives |

## Replay mode (notebook / Colab)

Recorded scenarios live in `travel_agent/snapshots/scenarios/<name>` and carry a
frozen `now` in `scenario.json`. Replay makes every tool read the recorded
responses instead of calling the network, so a demo is deterministic. The active
scenario is process-wide (not thread-local), so tools invoked from ADK worker
threads still replay.

```python
from travel_agent.tools import live, replay

replay.list_demo_scenarios()            # [{'name': ..., 'description': ..., 'now': ...}, ...]
replay.use_demo_scenario("sfo_gdp_20260923")

live.assess_trip_risk("New York", "San Francisco", "2026-09-23", "evening")
live.get_faa_airport_status("SFO")      # both use the scenario snapshots and frozen clock

replay.use_demo_scenario(None)          # back to live
```

`assess_trip_risk` forwards the replay mode, snapshot directory and frozen clock
to the orchestrator, so the whole rule engine replays consistently. Every
evidence item produced in replay carries `data.replay=True` and
`data.scenario=<name>`.

### Per-tool replay coverage (existing scenarios)

| Tool | Replays in the current scenarios? |
|---|---|
| `assess_trip_risk` | Yes — uses the orchestrator's recorded FAA/TAF/NWS/OSRM/etc. responses |
| `get_faa_airport_status` | Yes — FAA NAS XML is captured in `sfo_gdp_20260923` and `nyc_dc_20260924_0716Z` |
| `get_aviation_weather` | Partially — TAF/METAR captured in `sfo_gdp_20260923`; other tools below usually miss |
| `resolve_place` (city → network geocode) | No — only IATA codes and known/offline cities avoid a call |
| `find_nearby_airports`, `get_route_ontime_history`, `get_rail_options` | Offline reference only, no snapshots needed |
| `get_weather_alerts`, `get_weather_forecast` | Miss unless that point was captured; tools return `ok=False` with `replay snapshot miss` |
| `get_extended_forecast` | Miss — Open-Meteo forecast not captured |
| `get_drive_route`, `search_travel_news` | Drive is captured only in `nyc_dc_20260924_0716Z`; news only in that scenario |

A snapshot miss is never silently filled from the network. To cover the tools
that currently miss, record a new scenario with the real tools:

```python
from travel_agent.tools import live, replay

try:
    replay.record_demo_scenario("nyc_sf_20261124", "NYC to SFO, all tools")
    live.assess_trip_risk("New York", "San Francisco", "2026-11-25", "evening")
    live.get_weather_alerts("New York")
    live.get_extended_forecast("San Francisco")
    # ...call each tool you want in the demo while record mode is active
finally:
    replay.stop_recording()
```

This writes `travel_agent/snapshots/scenarios/nyc_sf_20261124/*.json` plus a
`scenario.json` whose `now` is the recording start time. Scenario names must
match `^[a-z0-9_]+$`. Recording never replaces the environment fallbacks
(`TDA_HTTP_MODE`, `TDA_SNAPSHOT_DIR`); with no scenario active they still work.

### Untrusted text

Third-party text is data, not instructions:

- `search_travel_news`: `id` is a stable hash, `subject` is always `news`, and
  `observation` is the fixed string `News headline (untrusted text)`. The
  headline lives in `data.text` with `data.untrusted=True`, and the publisher in
  `data.publisher`.
- `assess_trip_risk`: TomTom and MTA details are operator-written free text, so
  the observation is a fixed template built only from the rule category/leg/level.
  The raw detail moves **only** to `data.text` with `data.untrusted=True`; it is
  not duplicated into `data.title`, `data.detail` or the observation. NWS and FAA
  text may stay in the observation.
- NWS adapter errors use short fixed strings and never echo remote payloads.

