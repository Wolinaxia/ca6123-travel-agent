# Rule fix 1 — MTA noise and GA-only FAA closures

Work ONLY inside this repository; do not read `.env`; do not spawn sub-agents. No new deps.
This round MAY change `travel_agent/core/engine/evaluate.py` (MTA block only), `travel_agent/core/sources/mta.py` (timestamps only) and `travel_agent/tools/live.py` (`get_faa_airport_status` only). Nothing else in `core/`. Do not touch anything under `travel_agent/snapshots/`.
Each change: a test that fails before the fix. Offline `pytest -q` must pass.

## Problem (measured on recorded scenario `calm_baseline_20261121`)
The MTA subway feed had 224 alerts; 45 mention A/E/J/Z (32 unique headers), only 6 active at the recording time. The engine adds one `elevated` evidence per (alert × leg) with no time or relevance filter, so every NYC trip is `elevated` even on a calm day.

## R1 — MTA timestamps are timezone-wrong
`core/sources/mta.py::_time` uses `datetime.fromtimestamp(int(value))` (naive, machine-local tz). Use `datetime.fromtimestamp(int(value), tz=timezone.utc)`. Test with a fixed epoch.

## R2 — MTA rule in `evaluate.py` (only the `if inputs.mta and inputs.mta.ok:` block)
1. **Active in the trip window:** count an alert only if one of its `active_periods` overlaps the leg's time window (access = departure window at origin, egress = arrival window at destination; reuse the windows the function already computes for other sources). Alerts with no periods count as active.
2. **Dedupe** by `header_text`.
3. **Relevance → level:** `elevated` only if the header mentions the airport path: any of `JFK`, `Howard Beach`, `Jamaica`, `Sutphin`, `Far Rockaway`, `AirTrain`, `Kennedy`, `Newark Airport`, `LaGuardia` (case-insensitive). Other active A/E/J/Z alerts → level `info` (shown as context, does not raise the leg level).
4. **Aggregate:** emit at most one `info` evidence per leg summarising the count ("N active service changes on A/E/J/Z, none on the airport branch"), plus one `elevated` evidence per relevant (deduped) alert.
5. Record filtered counts in `filtered` (reason: "MTA alert outside trip window" / "MTA alert not on airport branch") like other sources do.
Tests: (a) future planned work outside the window → not counted; (b) 10 duplicate headers → one item; (c) "No [A] between 168 St and 207 St" active → info, leg stays low; (d) "No [A] trains to Howard Beach-JFK Airport" active → elevated; (e) existing tests still pass.

## R3 — GA-only FAA closures in `get_faa_airport_status`
When a closure has `affects_commercial=False`: observation `"General-aviation restriction at <IATA> (commercial flights not affected)"`, `data["program"]="ga_restriction"`, and it must not hide the "no active FAA program" evidence: the tool then returns both the GA note and the `"No active FAA program at <IATA>"` evidence (`data.active=False`). The raw NOTAM text goes to `data["reason"]` only. Commercial closures unchanged. Test with a recorded/fixture XML or a constructed `FaaNasData`.

## Final report
Per item done/not done, files, tests. Then, offline, run the replay check and report the result:
```
from travel_agent.tools import live, replay
replay.use_demo_scenario("calm_baseline_20261121")
for o,d,w in [("New York","Boston","evening"),("New York","Washington","morning"),("New York","San Francisco","whole_day")]:
    r = live.assess_trip_risk(o, d, "2026-11-21", w)
    print(o, d, w, [e["data"] for e in r["evidence"] if e["id"].endswith(":summary")][0]["legs"])
replay.use_demo_scenario(None)
```
Also confirm the SFO scenario (`sfo_gdp_20260923`, New York → San Francisco 2026-09-23 evening) still reports the SFO ground delay program and overall `elevated`.
