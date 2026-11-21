"""Demo/replay helpers for the live tools.

`use_demo_scenario` flips the shared HTTP layer into replay mode and pins the
clock to the scenario's recorded `now`, so every tool returns the frozen data
from `travel_agent/snapshots/scenarios/<name>` instead of making live calls.

The active scenario is stored in module-level state (guarded by a lock) as well
as in context variables: ADK may run sync tools in worker threads that do not
copy context, and those threads must still see the replay configuration.

`record_demo_scenario` is the inverse: it points the HTTP layer at a scenario
directory in record mode and stamps a `scenario.json` with the recording start
time, so a new deterministic demo can be captured with the real tools.
"""

from __future__ import annotations

import json
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import RLock

from travel_agent.core import http, scenarios
from travel_agent.core.http import http_mode, snapshot_dir


_SCENARIO_NAME_RE = re.compile(r"^[a-z0-9_]+$")

_lock = RLock()
_demo_now: ContextVar[datetime | None] = ContextVar("tda_demo_now", default=None)
_demo_snapshot: ContextVar[str | None] = ContextVar("tda_demo_snapshot", default=None)
_active: dict[str, str | datetime | None] = {"now": None, "snapshot": None, "scenario": None}


def _clear_state() -> None:
    with _lock:
        _active["now"] = None
        _active["snapshot"] = None
        _active["scenario"] = None
    _demo_now.set(None)
    _demo_snapshot.set(None)
    http_mode.set(None)
    snapshot_dir.set(None)
    http.set_active_state(None, None)


def list_demo_scenarios() -> list[dict]:
    """Return the recorded demo scenarios (name, description, frozen `now`, default query)."""
    return scenarios.list_scenarios()


def use_demo_scenario(name: str | None) -> dict | None:
    """Switch tools to a recorded scenario, or back to live mode with None."""
    if name is None:
        _clear_state()
        return None
    scenario = next((item for item in scenarios.list_scenarios() if item.get("name") == name), None)
    if scenario is None:
        raise ValueError(f"unknown demo scenario: {name}")
    now = datetime.fromisoformat(str(scenario["now"]).replace("Z", "+00:00"))
    directory = str(scenarios.ROOT / scenario.get("name", name))
    with _lock:
        _active["now"] = now
        _active["snapshot"] = directory
        _active["scenario"] = scenario.get("name", name)
    _demo_now.set(now)
    _demo_snapshot.set(directory)
    http_mode.set("replay")
    snapshot_dir.set(directory)
    http.set_active_state("replay", directory)
    return scenario


def record_demo_scenario(name: str, description: str = "", default_query: str = "") -> dict:
    """Start recording live responses into a new scenario directory.

    `name` must match ``^[a-z0-9_]+$``. The scenario's frozen `now` is the
    recording start time. Call `stop_recording` when the recording run is done.
    """
    if not _SCENARIO_NAME_RE.match(name or ""):
        raise ValueError("scenario name must match ^[a-z0-9_]+$")
    now = datetime.now(timezone.utc)
    directory = scenarios.ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": name,
        "description": description,
        "now": now.isoformat().replace("+00:00", "Z"),
        "recorded_at": now.isoformat().replace("+00:00", "Z"),
        "default_query": default_query,
    }
    (directory / "scenario.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with _lock:
        _active["now"] = now
        _active["snapshot"] = str(directory)
        _active["scenario"] = name
    _demo_now.set(now)
    _demo_snapshot.set(str(directory))
    http_mode.set("record")
    snapshot_dir.set(str(directory))
    http.set_active_state("record", str(directory))
    return metadata


def stop_recording() -> None:
    """Stop recording and return the HTTP layer to live mode."""
    _clear_state()


def demo_now() -> datetime | None:
    """The frozen scenario clock, or None when running live."""
    with _lock:
        value = _active["now"]
    return value if isinstance(value, datetime) else None


def replay_mode() -> str | None:
    """'replay' while a demo scenario is active, otherwise None."""
    with _lock:
        return "replay" if _active["snapshot"] and http.active_mode() == "replay" else None


def replay_snapshot() -> str | None:
    """Snapshot directory for the active demo scenario, otherwise None."""
    with _lock:
        value = _active["snapshot"]
    return str(value) if value else None


def scenario_name() -> str | None:
    """Name of the active demo scenario, or None when running live."""
    with _lock:
        value = _active["scenario"]
    return str(value) if value else None
