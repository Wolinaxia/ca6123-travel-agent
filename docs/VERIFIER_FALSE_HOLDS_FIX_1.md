# Fix round 1 for docs/VERIFIER_FALSE_HOLDS_FIX.md

Only work inside this repository directory; do not read `.env`; no network; do not run
`python -m eval.run_eval` without `--help`. The first round is in the working tree (uncommitted). Change
only `travel_agent/verification/checks.py`, `tests/test_verifier.py`, `eval/verifier_cases.json`.

A safety review of round 1 showed the loosening went too far, and a rescoring on the real claims showed
the new FAA floor creates its own false holds. This round replaces the round-1 rule for the risk
comparison with a narrower exemption, and narrows the number change.

## What round 1 let through (must be held again)
With an LGA-BOS route summary at `elevated`/`high` in the store, all with `risk: "low"`:
1. Trip-level reassurance without two upper-case codes:
   "Your flight tonight should be fine." (cites the clear BOS FAA status), "The trip looks low risk."
   (cites BOS no-alerts), "New York to Boston looks fine tonight.", "LGA to the destination looks fine
   tonight." (cites the LGA METAR), "lga to bos looks fine tonight.", "No disruption is expected for this
   journey."
2. A disrupted airport declared normal by citing a benign item for it, while an active LGA ground stop /
   GDP is in the store but not cited: "LGA operations are running normally." (cites the LGA METAR),
   "LGA has no active FAA programs." (cites the METAR), "LGA looks fine for departures tonight." (cites
   the LGA forecast).
3. Severe weather labelled low: "The thunderstorm warning near LGA will not affect flights." (cites a
   Severe NWS alert), "LGA is reporting LIFR conditions with gusts of 38 kt." (cites the LIFR METAR).

## What round 1 newly held (must be supported again)
Alternatives claims that say an option **avoids** a delayed airport, risk `low`, citing a planner option
and the FAA program being avoided, e.g. "Taking flight EWR to PVD avoids the 61 minute delay at LGA."
citing `["plan:flight:EWR-PVD", "faa_nas:LGA:GDP:2"]`. The `low` describes the alternative, not LGA.

## Task

### 1. Route-summary comparison: default back to the old behaviour, with one exemption
Restore the pre-round-1 `same_airport` behaviour as the default (airports from the claim text and from
trusted cited subjects pull in the route summary).

Exempt a claim from the `same_airport` path only when it is a **single-airport observation**, i.e. all of:
- its text names exactly one distinct upper-case airport code (`_claim_places`), call it A;
- every trusted, unexpired cited evidence is about A: its `subject` equals A, or A is a `:`-separated
  segment of its id (e.g. `nws:LGA:no_alerts`, `aviation_wx:LGA:metar`, `faa_nas:LGA:GDP:2`);
- it cites no engine evidence and no `plan:` evidence (those keep their existing paths);
- it has at least one trusted cited evidence.
`same_route` and `same_subject` matching and cited engine levels are unchanged for every claim.

### 2. Floor from cited evidence (replaces round 1's FAA floor)
Applies when `risk` is set and the claim cites **no `plan:` evidence**. Computed over trusted, unexpired
cited evidence; the claim's risk must not be below the highest floor:
- FAA program, read from `data.program` after `str(...).strip().lower()` with spaces and hyphens turned
  into underscores: `gdp` or `arrival_departure` -> `moderate`; `ground_stop`, `gs` or `closure` -> `high`.
  Do not test the evidence `source` for this; any cited evidence with such a `program` counts. Skip the
  evidence only when `data.affects_commercial is False`.
- NWS alert severity: `data.severity` (case-insensitive) `severe` or `extreme` -> `moderate`.
- METAR flight category: `data.flight_category` (case-insensitive) `ifr` or `lifr` -> `moderate`.
Below the floor -> `contradicted`, reason `downplays cited evidence`, check name `risk_consistent`.
(Use the single reason `downplays cited evidence` for all three kinds; update round-1 tests accordingly.)

### 3. Airport declared normal while a program is active elsewhere in the store
`_airport_status_conflict` only looks at cited evidence. Extend the check: for each airport code named
in the claim text, if the (trusted, unexpired) **store** holds an FAA evidence about that airport whose
normalised `data.program` is one of the values in section 2 and `affects_commercial` is not `False`, and
the claim text matches the existing `_NORMAL_AIRPORT` pattern **or** a reassurance pattern
(`\b(looks?|is|are|should be)\s+(fine|ok|okay|normal|unaffected|on time)\b`, `\bno (delays?|disruptions?|
problems?|issues?)\b`, `\bwill not affect\b`, case-insensitive), the claim is `contradicted` with the
existing reason of the airport-status check. A claim that itself cites that FAA evidence and reports the
program truthfully ("LGA has a ground delay program averaging 148 min") must not trigger this: it does not
match the patterns.

### 4. Numbers (narrow round 1)
- Keep: a key containing `probability`, or equal to `pop`, with a numeric value is `percent`.
- Free-text mining of unit-bearing numbers is limited to string values under the keys
  `detailed_forecast`, `short_forecast`, `wind`, `wind_speed`, `wind_gust` (at any nesting level). All
  other string values under keys with no unit family are ignored again, as before round 1.

### 5. Tests
- Every sentence in "What round 1 let through" is NOT `supported` (use a store with an `elevated` LGA-BOS
  summary, an active LGA ground stop and GDP, a Severe NWS alert at LGA, an LIFR METAR at LGA, a clear BOS
  FAA status, BOS no-alerts, an LGA forecast). Give each its own parametrised case.
- The four round-0 examples stay `supported`: no NWS alerts near LGA (low); BOS has no active FAA delay
  programs (low); BOS weather is currently VFR ... (low, VFR METAR); LGA has a Ground Delay Program
  averaging 148 minutes (moderate, trip `high`).
- "Taking flight EWR to PVD avoids the 61 minute delay at LGA." (low) citing a plan option and the LGA
  GDP: `supported` (plan evidence in the store with the numbers it needs).
- Floor: FAA `GDP` low -> contradicted, moderate -> supported; `Ground Stop` (odd casing) moderate ->
  contradicted, high -> supported; `closure` with `affects_commercial: False`, low -> not contradicted by
  the floor; Severe alert low -> contradicted, moderate -> supported; LIFR METAR low -> contradicted,
  moderate -> supported; VFR METAR low -> supported.
- Numbers: with the forecast evidence, "gusts up to 26 mph" and "80% chance of precipitation" supported;
  "gusts up to 46 mph", "30% chance" not supported; a number that appears only inside an FAA `reason`
  string or an engine `action_detail` string does NOT support a claim.
- List in your final message every pre-existing test or case (before round 1) whose expectation differs
  from the original, with old and new values.

## Acceptance
`.venv/bin/python -m pytest -q` fully green. No new dependencies, no sub-agents, do not commit.

## Additions from the correctness review (these override the text above where they differ)
- **Section 1:** "restore the old behaviour" includes the old `has_route_citation` gate that round 1
  deleted. A claim that cites engine evidence for a route must not inherit another route's summary through
  airports named in its text. Test: store with a JFK-SFO summary `low` and a JFK-OAK summary `high`;
  "JFK to SFO is low risk, better than OAK." (risk `low`, citing an `engine:JFK-SFO:...` finding) is
  `supported`, as before round 1.
- **Section 2, scope of the floor:** the floor applies only to a **single-airport observation** as defined
  in section 1 (so only evidence about airport A is considered). Claims naming several airports or citing
  `plan:`/engine evidence get no floor; they are handled by the route-summary comparison as before round 1.
  Tests (no route summary in the store, risk `low`, all must be `supported`):
  "Flying into OAK avoids the SFO delay." citing `faa_nas:OAK:status` and `faa_nas:SFO:GDP:0`;
  "The OAK option avoids the SFO ground delay program." citing a `plan:` option and `faa_nas:SFO:GDP:0`.
- **Section 4:** the percent rule is: key contains `probability`, or key equals `pop` exactly. Not a
  `_pop` suffix (`metro_pop` must not become a percent).
- Add tests for: one airport named in the text while another airport's evidence is also cited (not a
  single-airport observation -> old behaviour); a number under a skipped key such as `raw` does not
  support a claim.
