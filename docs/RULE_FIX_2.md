# Rule fix 2 — background priors must not raise the risk level on their own

Work ONLY inside this repository; do not read `.env`; do not touch `travel_agent/snapshots/`; do not spawn sub-agents. No new deps.
May change: `travel_agent/core/engine/evaluate.py`, `travel_agent/core/engine/models.py` (Policy.rules text only), tests. Nothing else in `core/`.
Each change: a test that fails before the fix. Offline `pytest -q` must pass. Do not weaken assertions unrelated to this change; if an existing test asserted the old behaviour, update it and say which and why in the final report.

## Design decision (approved)
Live signals (FAA programs/closures, NWS alerts, TAF/METAR weather, NWS/Open-Meteo forecasts, TomTom, MTA airport-branch alerts) decide the risk level.
**Background priors** — historical BTS route rates (`BTS_BASE_CAP`) and the holiday peak window (`HOLIDAY_PEAK`) — are context: they are still emitted as evidence (so the agent can mention them), but with level `info`, so they never raise a leg or the overall level by themselves.

## P1 — `evaluate.py`
- `BTS_BASE_CAP` and `HOLIDAY_PEAK` evidence: level `info` (category stays `delay`), and do not touch `levels[...]`.
- Keep their title/detail text; for BTS keep the rate in the detail.
- **Amplifier:** if the flight leg already has at least one live `elevated`-or-higher item, and a prior applies (BTS threshold met or holiday peak), add one `info` note `"Background priors (historical rate / holiday peak) make recovery slower"`; do not change the level. (Keep it simple; no level bumps from priors.)
- Confidence logic unchanged.

## P2 — `models.py`
Update `Policy.rules` descriptions for `BTS_BASE_CAP` and `HOLIDAY_PEAK` to say "context only, does not raise level", and add an entry for `MTA_SERVICE_CONTEXT` ("context: active non-airport-branch MTA service changes").

## Tests
1. Route meeting the BTS threshold, no live signals → overall `low`, BTS evidence present with level `info`.
2. Holiday-peak date, no live signals → overall `low`, HOLIDAY_PEAK evidence `info`.
3. Live elevated signal + prior → overall `elevated` (not higher), amplifier note present.
4. Recorded scenarios (offline replay):
   - `calm_baseline_20261121`: New York→Boston 2026-11-21 evening → overall `low`.
   - `sfo_gdp_20260923`: New York→San Francisco 2026-09-23 evening → overall `elevated`, SFO ground delay evidence present.

## Final report
Per item done/not done, files, tests changed (with reasons), and the replay output for both scenarios plus New York→Washington morning and New York→San Francisco whole_day on the calm baseline.
