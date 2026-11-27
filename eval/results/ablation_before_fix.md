# Verifier ablation (M1)

Labels: Gemini via Antigravity (an LLM, blind to Verifier output), not a human  

Generated: 2026-11-27T14:48:33+00:00  
Git: `968e3cc` · model: `gemini-3.5-flash-lite` · records: 18 · claims: 260

Turns without claims (excluded from escalation rates): a05, a07.

## Headline

| Measure | Verifier ON | Verifier OFF |
|---|---|---|
| Bad claims shown | 5/191 (3%) | 7/260 (3%) |
| Escalation rate | 14/16 (88%) | 7/16 (44%) |

## Arm A: real claims

| Measure | Verifier ON | Verifier OFF |
|---|---|---|
| Bad claims shown | 5/191 (3%) | 7/260 (3%) |
| Catch rate (wrong claims held back) | 2/7 (29%) | 0/7 (0%) |
| False hold rate (correct claims held back) | 67/253 (26%) | 0/253 (0%) |
| Escalation rate | 14/16 (88%) | 7/16 (44%) |

### By scenario

| Scenario | False holds ON | False holds OFF | Escalation ON | Escalation OFF |
|---|---|---|---|---|
| nyc_rain_20261122_1708z | 53/139 (38%) | 0/139 (0%) | 9/9 (100%) | 5/9 (56%) |
| nyc_rain_20261122_1550z | 12/69 (17%) | 0/69 (0%) | 4/4 (100%) | 2/4 (50%) |
| calm_baseline_20261121 | 2/45 (4%) | 0/45 (0%) | 1/3 (33%) | 0/3 (0%) |

### Headline shift

None.

## Held claims by Verifier reason

| Status | Reason | Correct | Wrong | Total |
|---|---|---|---|---|
| contradicted | downplays rule-engine risk | 40 | 0 | 40 |
| unsupported | states a percent value but cited evidence has none | 17 | 1 | 18 |
| contradicted | states N mph but cited evidence does not | 10 | 1 | 11 |

## Arm B: injected faults

Base claims: 186. Correct claims excluded for not being supported: 67. Controls supported: 186/186.

| Fault | In scope | Applicable | Caught | Catch rate |
|---|---|---|---|---|
| wrong_number | yes | 81 | 76 | 94% |
| wrong_airport | yes | 163 | 160 | 98% |
| fabricated_id | yes | 186 | 186 | 100% |
| no_citation | yes | 186 | 186 | 100% |
| risk_downplayed | yes | 25 | 17 | 68% |
| normal_vs_program | yes | 39 | 0 | 0% |
| wrong_time_window | no | 0 | 0 | n/a |

Overall catch rate over the six in-scope faults: 625/680 (92%).
Out-of-scope (expected to get through): n/a.

### Missed in-scope mutations

- `a01:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a01:flight_airport:3` risk_downplayed: "EWR has a Ground Delay Program averaging 115 minutes due to low ceilings, plus additional departure delays of 16-30 minutes due to airspace sharing with N90."
- `a01:flight_airport:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a01:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a01:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a02:alternatives:3` wrong_airport: "Flight from EWR to PVD avoids the 148-minute LGA delay and arrives with 136 min to spare."
- `a02:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a03:flight_airport:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:alternatives:1` wrong_number: "The direct flight from EWR to BOS is rated tight, arriving with only 77 minutes to spare after accounting for the 115-minute average FAA delay."
- `a03:alternatives:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a03:alternatives:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a04:alternatives:1` wrong_number: "Flight JFK to PVD avoids the FAA delay at EWR (365 min) and LGA (148 min) and arrives with 143 min to spare."
- `a04:alternatives:1` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a04:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:flight_airport:2` risk_downplayed: "EWR has a Ground Delay Program averaging 115 minutes due to low ceilings as of 12:08 PM local time."
- `p01:flight_airport:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:flight_airport:3` risk_downplayed: "EWR also has departure delays ranging from 16 to 30 minutes due to airspace sharing with N90 as of 12:08 PM local time."
- `p01:flight_airport:3` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:flight_airport:6` risk_downplayed: "EWR METAR observation reports IFR conditions with 9 kt winds and 2 statute miles visibility."
- `p01:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p01:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p01:alternatives:4` wrong_airport: "LGA has no active FAA delay programs, making JFK flights a strong backup option."
- `p02:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p03:flight_airport:2` risk_downplayed: "EWR has a Ground Delay Program averaging 115 minutes due to low ceilings, plus departure delays of 16-30 minutes due to airspace sharing with N90."
- `p03:flight_airport:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `p03:flight_airport:5` risk_downplayed: "EWR METAR observation indicates IFR conditions with 9 kt winds and 2 statute miles visibility."
- `p03:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p03:alternatives:2` wrong_airport: "LGA has no active FAA programs and offers a comfortable flight backup into PVD."
- `r01:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:alternatives:2` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r01:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `r02:flight_airport:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `r02:alternatives:0` normal_vs_program: "EWR is operating normally with no FAA delays."
- `r02:alternatives:2` wrong_number: "Flight JFK to PVD avoids the EWR delay of 365 min and the LGA delay of 148 min."
- `r02:alternatives:2` normal_vs_program: "EWR is operating normally with no FAA delays."
- `a06:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a06:flight_airport:6` risk_downplayed: "EWR current weather is LIFR with 1/2SM visibility in mist and overcast clouds at 300 ft."
- `a06:alternatives:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a06:alternatives:3` normal_vs_program: "LGA is operating normally with no FAA delays."
- `a08:alternatives:1` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:flight_airport:0` normal_vs_program: "LGA is operating normally with no FAA delays."
- `p04:flight_airport:6` risk_downplayed: "EWR METAR reports LIFR conditions with 1/2 SM visibility and mist at 14:51 UTC."
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
