# Verifier false holds: final design (rounds 3 and 4)

This file describes what was merged. `VERIFIER_FALSE_HOLDS_FIX.md` (round 1) and
`VERIFIER_FALSE_HOLDS_FIX_1.md` (round 2) are kept as history; both were rejected in review for
loosening too much, and where they differ from this file, this file wins.

The change was made **after** seeing the ablation result (67 of 253 correct claims held back). It was
developed against the same 260 labelled claims, so the "after" numbers in `eval/results/` are not a
held-out measurement.

## Rule

`verify_claim` behaves exactly as before, except for a **single-airport observation**:

- the claim text names exactly one upper-case airport code;
- it has no trip or judgement wording (`trip`, `journey`, `itinerary`, `route`, `travel…`, `you`, `your`,
  `flying`, `flight` except "flight category/rules/conditions", `risk…`, `safe`, `smooth`, `easy`, `worry`,
  `concern`, `threat…`, `good to go`, `go ahead`, `as planned`, `on time`, `no problem`, `no issue`);
- every trusted, current citation has that airport as its subject, and none is engine or planner evidence.

For such a claim:

1. **Floor from its own evidence** (new check). A cited active FAA program requires at least `moderate`
   (GDP, arrival/departure delays) or `high` (ground stop, closure); a cited NWS alert of severity Severe or
   Extreme, or a METAR with flight category IFR or LIFR, requires at least `moderate`. Below that:
   `contradicted: downplays cited evidence`. General-aviation-only restrictions do not count.
2. **Trip-level summary.**
   - If the store holds no active FAA program for that airport, the route summary is not compared. "No
     active NWS alerts near BOS" (low) stands on a trip rated high.
   - If the store does hold one (cited or not), the route summary is compared as before, but a risk at or
     above that program's level is accepted (`supported: consistent with the airport's FAA program`).
     "LGA is reporting LIFR conditions" (moderate) stands when LGA has a ground delay program; "no alerts
     near LGA" (low) does not. A sentence that says the airport is normal ("operations are running
     normally", "no active FAA programs") is never accepted this way.

Everything that is not a single-airport observation (no code, city names, lower-case codes, two airports,
a route or engine citation, trip wording) keeps the original comparison.

Numbers: a key containing `probability`, or equal to `pop`, is a percent; unit-bearing numbers are also
read from the string values of `detailed_forecast`, `short_forecast`, `wind`, `wind_speed`, `wind_gust`.
No other free text is mined.

## Property the design is built on

Apart from the floor in point 1, the change only **removes** holds the original code made. It adds no new
reason to hold a claim, so it cannot create false holds of its own, and anything it newly shows is a
single-airport observation at an airport with no recognised active FAA program, or one whose risk is at
least that program's level.

## Checks made on the final code

- A reviewer's generated test (89,040 combinations of claim text, citations, risk and store) found no case
  where the new code holds a claim the original showed, other than through the floor in point 1.
- The sentences earlier reviews got through (trip-level reassurance without two codes, a disrupted airport
  declared normal by citing something benign, severe weather labelled low) behave as in the original code.

## Known limits (shown now, held before)

- **Numbers in forecast prose.** Any unit-bearing number in the forecast paragraph supports a claim of the
  same unit: "gusts only reach 10 mph" matches "gusts 10 mph after midnight" even if the paragraph also
  says 46 mph; "20% chance tonight" matches Saturday's 20%.
- **Floor coverage.** No floor for TAF forecast ceilings or visibility, for NWS alerts below Severe, for
  MVFR, or for FAA `program` values outside GDP / arrival_departure / ground_stop / closure (the live tools
  emit only these four). A factual sentence about such evidence can carry a `low` label on a trip rated
  higher.
- **Wording list.** A trip-level judgement that names one airport and avoids every listed word is
  treated as an observation of that airport. The list was extended once in review; it is a list, not a
  parser, and will have gaps.
- In all three cases the answer's overall risk is still the higher of the rule engine's level and the
  shown claims, so the headline cannot be talked down.

## Result on the labelled ablation set (same 260 claims, Verifier ON)

| | Before | After |
|---|---|---|
| Correct claims held back | 67/253 (26%) | 10/253 (4%) |
| Wrong claims held back | 2/7 | 0/7 |
| Turns flagged for a human | 14/16 | 10/16 |
| Injected faults caught (six in-scope types) | 625/680 (92%) | 847/918 (92%) |
| of which `risk_downplayed` | 17/25 (68%) | 41/58 (71%) |

The 10 remaining holds are all low-risk observations (no alerts, a mild forecast) at an airport that has
an active FAA program, on a trip rated higher. The two wrong claims held before were held for unrelated
number reasons that the fix removed; none of the seven wrong claims (semantic errors such as overstated
case lessons or loose terminology) is something these checks detect.
