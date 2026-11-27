# M4 history back-test: do the past-case lessons hold on real data?

Generated: 2026-11-27T03:37:12+00:00

Data: real FAA ground delay program (GDP) advisories (https://www.fly.faa.gov/adv/advADB) joined with BTS
per-flight arrival records (https://transtats.bts.gov). Learned on programs that started in **2025**, tested
on programs that started in **2026** (Jan-Aug, never used for learning). Built by `eval/backtest_gdp_history.py`.

## In short

1. **The past-case lesson holds on real data.** Case `sfo_fog_gdp_oak` says a neighbouring airport usually avoids a delay program. In 2026 the neighbour did better in **185 of 197 programs (93.9%)**; in 2025, 7.9% of flights into program airports were cancelled or diverted vs 3.1% at neighbours.
2. **The announced average is not the whole story.** In 2025, 1 in 10 flights into a program airport arrived 78 min or more after the announced average delay.
3. **The history-aware planner keeps its promises more often.** On 2026 programs (not used for learning), travellers told "comfortable" who still missed fell from 8.3% to 7.5%; the options it downgraded really missed 31.2% of the time. It rarely changes which airport is chosen, because the planner already prefers the safer neighbour.

Data: 334 programs and 67,806 flights in the 2026 test alone.

## 1. What real flights did during announced delay programs (learned, 2025)

| Kind of program | programs | flights at program airport | real delay beyond announced avg (median / 90th pct) | cancelled or diverted | neighbour airport delay (median / 90th pct) | neighbour cancelled |
|---|---|---|---|---|---|---|
| thunderstorms | 111 | 15991 | -3 / 114 min | 11.5% | 16 / 122 min | 11.3% |
| low_visibility | 92 | 10502 | -16 / 58 min | 4.2% | 0 / 60 min | 1.9% |
| wind | 61 | 8289 | -15 / 59 min | 3.9% | 0 / 54 min | 1.3% |
| snow_ice | 19 | 4782 | -12 / 94 min | 10.5% | 5 / 94 min | 17.8% |
| other | 196 | 25771 | -17 / 64 min | 8.1% | 0 / 53 min | 1.1% |
| all | 479 | 65335 | -14 / 78 min | 7.9% | 0 / 64 min | 3.1% |

How often the FAA later **raised** the announced average (all saved advisories, 2024-2025):

* thunderstorms: 42/193 programs (21.8%)
* low_visibility: 17/146 programs (11.6%)
* wind: 20/105 programs (19.0%)
* snow_ice: 12/54 programs (22.2%)
* other: 54/429 programs (12.6%)
* all: 145/927 programs (15.6%)

## 2. Lesson check: was the neighbouring airport really safer? (case `sfo_fog_gdp_oak`)

Per program: share of flights more than 60 min late or cancelled, program airport vs same-metro neighbours
that had no program of their own that day.

| Kind of program | 2025: neighbour better | 2026: neighbour better |
|---|---|---|
| thunderstorms | 60/67 (89.6%) | 49/50 (98.0%) |
| low_visibility | 52/55 (94.5%) | 32/34 (94.1%) |
| wind | 39/41 (95.1%) | 14/15 (93.3%) |
| snow_ice | 7/8 (87.5%) | 9/9 (100.0%) |
| other | 88/95 (92.6%) | 81/89 (91.0%) |
| all | 246/266 (92.5%) | 185/197 (93.9%) |

## 3. Back-test: does the planner keep its promises?

A traveller scenario = one real flight into the program airport x one deadline (0-240 min of slack after
the announced delay). *Missed* = the real flight arrived after the deadline, or was cancelled / diverted.
History rule: an option is "comfortable" only if it still makes the deadline at the 90th-percentile
delay that real flights had in 2025 programs of the same kind.

**TEST: 2026 programs (not used for learning)** (334 programs, 48752 flights at the program airport, 19054 at neighbours)

| | current planner (trusts the announced average) | history-aware planner |
|---|---|---|
| travellers told "comfortable" | 259446 | 250551 |
| ...of whom actually missed the deadline | 21580 (8.3%) | 18803 (7.5%) |
| downgraded to "tight" by history | - | 8895 (of whom 31.2% really missed) |

Choice between the program airport and a same-metro neighbour (16745 traveller scenarios: deadlines 0-240 min of slack x 0-60 min extra ground time from the neighbour):

* the history rule changed the choice in 40 scenarios (0.2%);
* in those scenarios the real miss rate was 40.9% with the current choice and 32.9% with the history-aware choice;
* over all scenarios: 11.6% vs 11.6%.

**For reference, 2025 (the learning year, so optimistic)** (479 programs, 65335 flights at the program airport, 28446 at neighbours)

| | current planner (trusts the announced average) | history-aware planner |
|---|---|---|
| travellers told "comfortable" | 380622 | 366036 |
| ...of whom actually missed the deadline | 26082 (6.9%) | 23631 (6.5%) |
| downgraded to "tight" by history | - | 14586 (of whom 16.8% really missed) |

Choice between the program airport and a same-metro neighbour (22610 traveller scenarios: deadlines 0-240 min of slack x 0-60 min extra ground time from the neighbour):

* the history rule changed the choice in 50 scenarios (0.2%);
* in those scenarios the real miss rate was 24.4% with the current choice and 29.4% with the history-aware choice;
* over all scenarios: 11.0% vs 11.0%.

## Versions (what changed and why)

* **Rule v1** also downgraded options landing at a same-metro *neighbour* (using the neighbour statistic).
  On 2026 it cut wrong "comfortable" promises from 8.3% to 7.5%, but where it changed the airport choice
  (80 of 16,745 scenarios) the real miss rate got slightly worse (33.4% -> 35.6%), because it penalised the
  neighbour, which section 2 shows was safer in 94% of programs.
* **Rule v2** (current): only the program airport itself is downgraded, because section 2 shows the
  neighbour was the safer airport. Checked on a split inside 2025 (learn Jan-Jun, check Jul-Dec): the
  "comfortable" promise improved the same way (6.1% -> 5.7% missed), but in the few scenarios where the
  airport choice changed (27 for v2, 52 for v1), both versions did slightly worse there. On 2026, v2 changed
  40 choices and did better there (40.9% -> 32.9%). These counts are too small to call: **the rule improves
  the honesty of "comfortable"; it does not measurably improve which airport is chosen.**
  Since v1's 2026 result had been seen before v2 was chosen, 2026 is not a perfectly untouched test for v2.
* Also tried on the 2025 split, not adopted: ranking options by estimated chance of missing (from the same
  history). It lowered the miss rate only from 11.78% to 11.68% overall, because the planner already prefers
  the neighbour airport during a program, which the data confirms; not worth changing the ranking before freeze.

## Honest limits

* Flights are sampled every 3rd day; programs crossing midnight UTC are cut at the day boundary.
* A traveller's deadline is simulated (0-240 min of slack); real travellers' deadlines are unknown.
* Delays are arrival delays reported by airlines to BTS; connections and ground transport are not included.
* Neighbour airports count only when they had no program of their own that day.
* Only ground delay programs are covered; ground stops, airline outages and rail are not.
