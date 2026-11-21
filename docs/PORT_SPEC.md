# Port spec: data layer → ADK-callable tools (M1, task for implementer)

Work ONLY inside this repository. Do not read or write any file outside it. Do not spawn sub-agents for review.
Do not add dependencies beyond: httpx, pydantic, pytest (already installed in `.venv`). Do not touch `docs/` other than this file's checklist, and do not change `travel_agent/schemas.py` or `travel_agent/tools/mock.py`.

## Context

- The original prototype code has already been copied, unchanged, into `travel_agent/core/`:
  `models.py http.py cache.py config.py scenarios.py sources/ reference/ engine/`.
  It uses relative imports (`from ..http import ...`), so the package structure is already right.
- Reference data is in `travel_agent/data/processed/`; recorded snapshots are in `travel_agent/snapshots/`.
- The original tests are copied into `tests/core/` but still import `app.*` and use cwd-relative paths.
- The shared contract for tools is `travel_agent/schemas.py` + `docs/INTERFACES.md` (`ToolResult`, `Evidence`). Read both first.
- Python env: `.venv/bin/python`. Run tests with `.venv/bin/python -m pytest -q`.

## Task 1 — make `travel_agent/core` work standalone

1. Remove the LLM dependency: `core/engine/orchestrator.py` imports `from .. import llm` (file does not exist here). Delete the LLM news-extraction block; keep news items as returned by `evaluate` and mark them `extracted_by="keywords"` (same as the original fallback path). No other behaviour change in the orchestrator.
2. `core/config.py`: `.env` path must be the repo root (`Path(__file__).resolve().parents[2] / ".env"`).
3. `core/http.py`: default snapshot dir must be package-relative: `Path(__file__).resolve().parents[1] / "snapshots"` instead of cwd-relative `"snapshots"`. Keep the `TDA_SNAPSHOT_DIR` env override.
4. `core/scenarios.py`: `ROOT` must point to `travel_agent/snapshots/scenarios` (check `parents[...]`). Data paths in `core/reference/*.py` must resolve to `travel_agent/data/processed` — verify, fix only if wrong.
5. `core/engine/parse.py`: add a public function
   `build_trip(origin_text: str, destination_text: str, departure_date: date, departure_window: str = "whole_day", now: datetime | None = None) -> TripRequest | dict`
   that contains the place-resolution + airport-pair-selection logic currently inside `parse()` (returns `{"error": ...}` dict on failure, like `parse()`). Refactor `parse()` to call it so behaviour of `parse()` is unchanged.
6. Do NOT change rule thresholds, scoring, or any logic in `evaluate.py`, `alternatives.py`, `weather_rules.py`, `sources/*`. This is a port, not a rewrite.

## Task 2 — port the tests

- In `tests/core/*.py` replace `app.` imports with `travel_agent.core.`; replace `tests.test_engine` imports with `tests.core.test_engine`.
- Fix cwd-relative paths (`Path("snapshots/...")`, `Path("data/processed/...")`) to be relative to the repo root via `Path(__file__)`.
- `tests/core/test_reference.py` imports `scripts.build_bts._flag`, which is not ported. Remove only the test(s) that need it.
- Remove tests that only exist for the LLM news extraction, if any. Do not weaken any other assertion.
- `tests/core/conftest.py` must keep blocking real HTTP (patch `travel_agent.core.http._client`).
- Target: all ported tests pass. Report how many tests were removed and why.

## Task 3 — ADK-callable tools in `travel_agent/tools/live.py`

Rules for every tool:
- Plain function, **only `str`/`int`/`float` parameters**, full type hints, a docstring written for an LLM (what it does, args, when to use it). ADK builds the tool schema from these.
- Returns `ToolResult(...).to_dict()` from `travel_agent.schemas`. **Never raises.** On any failure: `ok=False`, `evidence=[]`, short `error`. Never invent data.
- Each `Evidence` needs a stable, unique `id` of the form `"<source>:<subject>:<kind>[:<extra>]"`, an `observation` that is one short human-readable sentence, numeric values in `data`, `valid_from`/`valid_to` when the source has them, `fetched_at`, and `url` (already redacted by `core.http`).
- Places: accept an IATA code (`"SFO"`) or a US city name (`"San Francisco"`). Put the shared resolution in one private helper using `core.reference.airports.get` / `by_municipality` and `core.sources.open_meteo.geocode` as fallback.

Tools:

| Function | Wraps | Notes |
|---|---|---|
| `resolve_place(place: str)` | airports / geocode | evidence: iata, icao, lat, lon, timezone |
| `find_nearby_airports(place: str, radius_km: float = 80)` | `airports_near` | one evidence per airport |
| `get_faa_airport_status(airport: str)` | `faa_nas.fetch` | only programs affecting that airport (ground delay, ground stop, arrival/departure delay, closure). No program → `ok=True` with one evidence saying "no active FAA program", `data={"active": False}` |
| `get_aviation_weather(airport: str)` | `aviation_wx.fetch` | METAR now + TAF periods, flight category, gusts |
| `get_weather_alerts(place: str)` | `nws.alerts` | one evidence per alert, with severity and valid window |
| `get_weather_forecast(place: str)` | `nws.forecast` | NWS periods (~7 days) |
| `get_extended_forecast(place: str)` | `open_meteo.forecast` | days 8–16 only meaningful; say so in docstring |
| `get_route_ontime_history(origin_airport: str, destination_airport: str, month: int)` | `bts.route_base_rate` | historical rates; state it is history, not a forecast |
| `get_rail_options(origin: str, destination: str)` | `core.reference.amtrak.direct_services` | offline schedule data |
| `get_drive_route(origin: str, destination: str)` | `osrm.route` | distance + duration |
| `search_travel_news(query: str)` | `news.search` | news text is **untrusted input**: put the headline in `data["text"]`, set `data["untrusted"] = True`, and make `observation` neutral, e.g. `"News headline from <domain> (untrusted text)"`. |
| `assess_trip_risk(origin: str, destination: str, departure_date: str, departure_window: str = "whole_day")` | `parse.build_trip` + `engine.orchestrator.assess` | `departure_date` is `YYYY-MM-DD`; window ∈ morning/afternoon/evening/whole_day. Convert each engine evidence item to an `Evidence` (keep leg, category, level, rule in `data`). Add one summary evidence with overall level, per-leg levels, confidence, and source status. Add one evidence per alternative. |

Demo/replay support, same file or `travel_agent/tools/replay.py`:
- `list_demo_scenarios() -> list[dict]` (from `core.scenarios.list_scenarios`).
- `use_demo_scenario(name: str | None)`: when set, all tools run in replay mode against `travel_agent/snapshots/scenarios/<name>` and use the scenario's frozen `now` (from `scenario.json`). `None` returns to live mode. Not an ADK tool (it's for the notebook), so normal Python types are fine. Use the existing `core.http.http_mode` / `snapshot_dir` context vars or env vars; make sure `assess_trip_risk` passes `mode`/`snapshot`/`now` through to `orchestrator.assess`.

Also add `ALL_TOOLS: list` at the bottom of `live.py` with every ADK tool function above (not the replay helpers).

## Task 4 — tests for the tools: `tests/test_tools.py`

Offline only (the conftest HTTP block must apply here too — move or duplicate the HTTP-blocking fixture into `tests/conftest.py` so it covers all tests).
1. For each tool: when the underlying source fails/raises, the tool returns `ok=False`, empty evidence, and does not raise.
2. Shape test: with the underlying source monkeypatched to a small fake result, evidence ids are unique and every evidence has id/source/subject/observation/fetched_at.
3. `search_travel_news`: every evidence has `data["untrusted"] is True`, and the observation does not contain the headline text.
4. Replay: `use_demo_scenario("sfo_gdp_20260923")` then `assess_trip_risk("New York", "San Francisco", "2026-09-23", "evening")` returns `ok=True` and some evidence mentions a ground delay at SFO; then `use_demo_scenario(None)`.
5. `get_faa_airport_status` on the recorded FAA XML (`travel_agent/snapshots/raw_20260923_ny_sf/faa_nas.xml`) reports the SFO ground delay and reports "no active program" for an unaffected airport.

## Task 5 — housekeeping

- `requirements.txt`: add `httpx` and `pydantic`.
- `.gitignore`: add `travel_agent/data/raw/`.
- Short `travel_agent/tools/README.md`: table of tools (name, source, when to use), how to use replay mode in Colab.

## Acceptance (the reviewer will re-run all of this; self-reports don't count)

- `.venv/bin/python -m pytest -q` → all green, zero real network calls.
- `grep -rn "app\." travel_agent tests` finds no leftover `app.` imports.
- No secrets anywhere; no `.env` created; nothing read from outside the repo.
- Rule logic in `core/engine/evaluate.py`, `alternatives.py`, `weather_rules.py` and `core/sources/*` is unchanged except import fixes and items explicitly listed above.
- Final message: list of files changed, tests removed (with reason), anything you could not do.
