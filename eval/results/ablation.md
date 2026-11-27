# Verifier ablation (M1)

Labels: Gemini via Antigravity (an LLM, blind to Verifier output), not a human  

Generated: 2026-11-27T16:01:28+00:00  
Git: `968e3cc` · model: `gemini-3.5-flash-lite` · records: 18 · claims: 260

Turns without claims (excluded from escalation rates): a05, a07.

## Headline

| Measure | Verifier ON | Verifier OFF |
|---|---|---|
| Bad claims shown | 7/250 (3%) | 7/260 (3%) |
| Escalation rate | 10/16 (62%) | 7/16 (44%) |

## Arm A: real claims

| Measure | Verifier ON | Verifier OFF |
|---|---|---|
| Bad claims shown | 7/250 (3%) | 7/260 (3%) |
| Catch rate (wrong claims held back) | 0/7 (0%) | 0/7 (0%) |
| False hold rate (correct claims held back) | 10/253 (4%) | 0/253 (0%) |
| Escalation rate | 10/16 (62%) | 7/16 (44%) |

### By scenario

| Scenario | False holds ON | False holds OFF | Escalation ON | Escalation OFF |
|---|---|---|---|---|
| nyc_rain_20261122_1708z | 10/139 (7%) | 0/139 (0%) | 8/9 (89%) | 5/9 (56%) |
| nyc_rain_20261122_1550z | 0/69 (0%) | 0/69 (0%) | 2/4 (50%) | 2/4 (50%) |
| calm_baseline_20261121 | 0/45 (0%) | 0/45 (0%) | 0/3 (0%) | 0/3 (0%) |

### Headline shift

None.

## Held claims by Verifier reason

| Status | Reason | Correct | Wrong | Total |
|---|---|---|---|---|
| contradicted | downplays rule-engine risk | 10 | 0 | 10 |

## Arm B: injected faults

Base claims: 243. Correct claims excluded for not being supported: 10. Controls supported: 243/243.

| Fault | In scope | Applicable | Caught | Catch rate |
|---|---|---|---|---|
| wrong_number | yes | 108 | 103 | 95% |
| wrong_airport | yes | 220 | 217 | 99% |
| fabricated_id | yes | 243 | 243 | 100% |
| no_citation | yes | 243 | 243 | 100% |
| risk_downplayed | yes | 58 | 41 | 71% |
| normal_vs_program | yes | 46 | 0 | 0% |
| wrong_time_window | no | 1 | 0 | 0% |

Overall catch rate over the six in-scope faults: 847/918 (92%).
Out-of-scope (expected to get through): 0/1 (0%).

### Missed in-scope mutations

- `a01:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a01:flight_airport:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a01:weather:3` risk_downplayed: "The NWS forecast for BOS during the evening window indicates heavy rain and patchy fog with northeast winds of 6 to 14 mph and a 100% chance of precipitation."
- `a01:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a01:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:3` wrong_airport: "Flight from EWR to PVD avoids the 148-minute LGA delay and arrives with 136 min to spare."
- `a02:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a03:flight_airport:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:flight_airport:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:flight_airport:6` risk_downplayed: "BOS TAF forecasts rain and lower ceilings (down to 700 ft with 2 SM visibility) later tonight (02:00-10:00 UTC)."
- `a03:weather:3` risk_downplayed: "The BOS forecast for the evening window (2026-11-22T19:17:54-05:00 to 2026-11-23T01:17:54-05:00) indicates heavy rain and patchy fog, a low around 53, northeast wind 6 to 14 mph, a 100% chance of precipitation, and 1 to 2 inches of new rainfall possible."
- `a03:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:alternatives:1` wrong_number: "The direct flight from EWR to BOS is rated tight, arriving with only 77 minutes to spare after accounting for the 115-minute average FAA delay."
- `a03:alternatives:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:alternatives:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a04:weather:1` risk_downplayed: "The NWS forecast for JFK during the 2026-11-22 evening window calls for light rain, cloudy skies, low around 55, north wind 12 to 15 mph with gusts up to 26 mph, and 90% chance of precipitation."
- `a04:weather:3` risk_downplayed: "The NWS forecast for BOS during the 2026-11-22 evening window calls for heavy rain and patchy fog, cloudy skies, low around 53, northeast wind 6 to 14 mph, 100% chance of precipitation, and 1 to 2 inches of possible rainfall."
- `a04:alternatives:1` wrong_number: "Flight JFK to PVD avoids the FAA delay at EWR (365 min) and LGA (148 min) and arrives with 143 min to spare."
- `a04:alternatives:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a04:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p01:flight_airport:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:flight_airport:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:weather:5` risk_downplayed: "The weather forecast for BOS from 2026-11-22T18:00:00-05:00 to 2026-11-23T06:00:00-05:00 indicates rain likely before 8pm, then rain and patchy fog, a low around 53, and NE winds of 6 to 14 mph with heavy rainfall amounts between 1 and 2 inches possible."
- `p01:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p01:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:alternatives:4` wrong_airport: "LGA has no active FAA delay programs, making JFK flights a strong backup option."
- `p02:flight_airport:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p02:flight_airport:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p02:weather:5` risk_downplayed: "BOS forecast for 2026-11-22 from 18:00 to 2026-11-23 06:00 indicates heavy rain and patchy fog, cloudy with a low around 53, and northeast wind 6 to 14 mph."
- `p02:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p03:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p03:flight_airport:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p03:weather:5` risk_downplayed: "BOS forecast for the evening/night of 2026-11-22 indicates rain likely before 8pm, then heavy rain and patchy fog with a low around 53, 100% chance of precipitation, and 1 to 2 inches of possible rainfall."
- `p03:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p03:alternatives:2` wrong_airport: "LGA has no active FAA programs and offers a comfortable flight backup into PVD."
- `r01:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:weather:3` risk_downplayed: "The BOS weather forecast for the evening window on 2026-11-22 indicates heavy rain, patchy fog, a low around 53, northeast winds at 6 to 14 mph, and a 100% chance of precipitation with 1 to 2 inches of possible rainfall."
- `r01:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r02:flight_airport:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `r02:flight_airport:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `r02:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `r02:alternatives:2` wrong_number: "Flight JFK to PVD avoids the EWR delay of 365 min and the LGA delay of 148 min."
- `r02:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a06:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a06:weather:3` risk_downplayed: "BOS forecast for the evening window indicates rain likely before 8pm, then rain and patchy fog, cloudy with a low around 53, northeast wind 6 to 14 mph, 100% chance of precipitation, and 1 to 2 inches of new rainfall possible."
- `a06:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a06:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a08:flight_airport:4` risk_downplayed: "JFK TAF forecast from 2026-11-22T23:00:00+00:00 to 2026-11-23T02:00:00+00:00 calls for RA, BR, ceiling 800 ft, visibility 4.0 sm."
- `a08:flight_airport:5` risk_downplayed: "JFK TAF forecast from 2026-11-23T02:00:00+00:00 to 2026-11-23T09:00:00+00:00 calls for RA, BR, ceiling 800 ft, visibility 4.0 sm."
- `a08:flight_airport:6` risk_downplayed: "BOS TAF forecast from 2026-11-23T00:00:00+00:00 to 2026-11-23T02:00:00+00:00 calls for RA, ceiling 2500 ft, visibility 5.0 sm."
- `a08:flight_airport:7` risk_downplayed: "BOS TAF forecast from 2026-11-23T02:00:00+00:00 to 2026-11-23T10:00:00+00:00 calls for RA, BR, ceiling 700 ft, visibility 2.0 sm."
- `a08:weather:1` risk_downplayed: "For the period overlapping 2026-11-22 18:00 to 24:00 at JFK, the forecast calls for light rain, a low around 54, north wind 12 to 15 mph with gusts up to 26 mph, and 88-90% probability of precipitation."
- `a08:weather:3` risk_downplayed: "For the period overlapping the BOS window on 2026-11-22 evening, the forecast calls for heavy rain, patchy fog, a low around 53, northeast wind 6 to 14 mph, and a 100% chance of precipitation with 1 to 2 inches of possible rainfall."
- `a08:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:weather:5` risk_downplayed: "For BOS between 18:00 on 2026-11-22 and 06:00 on 2026-11-23, the forecast calls for rain likely before 8pm then rain and patchy fog, a low around 53, and northeast wind 6 to 14 mph (100% chance of precipitation with 1 to 2 inches possible)."
- `p04:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:alternatives:2` wrong_number: "Taking flight EWR to PVD avoids the 203 minute delay at LGA."
- `p04:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r03:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r03:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r03:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r03:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p05:alternatives:0` wrong_number: "Amtrak from New York Moynihan Train Hall at Penn Station to Route 128 Amtrak Station has a door-to-door duration of 821 min and arrives with 854 min of slack before the deadline."

## Limitations

- Single run per request.
- One labeller (Gemini via Antigravity (an LLM, blind to Verifier output), not a human); Arm A's "wrong" means wrong according to that labeller.
- M4's two extra approval-card checks are excluded.
- Three recorded scenarios on one route (New York ↔ Boston).
