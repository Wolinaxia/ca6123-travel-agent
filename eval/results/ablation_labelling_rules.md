# Labelling rules for the Verifier ablation

These are the instructions the labeller was given, together with the blank sheet
(`ablation_labels.csv` before the `label` and `note` columns were filled). The labeller was Gemini,
run by hand through Antigravity. It had no access to this repository, to the Verifier's code or to any
Verifier verdict, and was told not to look for them. The instructions were written in Chinese; this is
a translation of the rules part, made by the person who wrote them.

## What was asked

Fill `label` for every row (260 rows, no sampling) and `note` where needed.

Each row: `claim_id`, `scenario`, `request` (what the traveller asked), `agent`, `claim` (the sentence to
judge), `risk` (the agent's risk label for the claim: low / moderate / high, empty = plain statement of
fact), `confidence` (ignore), `cited_ids`, `evidence` (the cited evidence: for each id a one-line summary,
then `data: {...}`, then `valid: from..to` when present), `label`, `note`.

The data is a recording replayed with a frozen clock. "Today", "tonight" and "as of" in a claim refer to:

- `nyc_rain_20261122_1708z`: 2026-11-22 17:08 UTC = 12:08 in New York (Sunday)
- `nyc_rain_20261122_1550z`: 2026-11-22 15:50 UTC = 10:50 in New York (Sunday)
- `calm_baseline_20261121`: 2026-11-21 07:18 UTC = 02:18 in New York (Saturday)

## How to judge

**Check the `claim` only against the `evidence` column of the same row.** Do not use your own knowledge,
evidence from other rows, or a web search to add to it or to contradict it. The only question is: is this
sentence supported by the evidence it cites?

One label out of seven:

- `correct`: every fact in the claim is in the cited evidence, or is a direct conversion or rewording of
  it. Rounding, unit conversion, time-zone conversion and spelling out an abbreviation count as consistent.
- `wrong_number`: a number in the claim (minutes, percent, visibility, wind speed, time, flight count...)
  does not agree with the evidence.
- `wrong_place`: an airport, city or route direction does not agree with the evidence.
- `risk_downplayed`: the `risk` column is clearly lighter than what the evidence shows. Mainly: the
  evidence shows an active ground delay program, ground stop or closure at that airport and the claim is
  labelled `low`; or the evidence is a severe weather alert and the claim is labelled `low`. Only when
  `risk` is not empty.
- `stale`: the claim is about a time period that the cited evidence's `valid:` window does not cover, or
  the evidence is clearly older than the scenario's "now" and the claim treats it as current. Not to be
  used when the evidence has no `valid:` part.
- `unsupported`: the claim says something the cited evidence does not contain, including: a fact the
  evidence does not mention; a prediction or guarantee beyond the evidence (evidence: "no delay program
  now"; claim: "no delays tonight"); a historical statistic presented as a prediction for this flight; an
  empty `cited_ids` with a factual statement.
- `other_wrong`: wrong, but none of the above. Say what is wrong in `note`.

Details:

1. When several errors apply, take the first that applies in this order: `wrong_number`, `wrong_place`,
   `risk_downplayed`, `stale`, `unsupported`, `other_wrong`. Put the others in `note`.
2. If any part of the sentence does not stand, the whole claim is not `correct`.
3. Truncated evidence: `data` longer than 1200 characters is cut and ends with `…` (10 rows). If the fact
   you need may be in the cut part, judge on what is visible and write `evidence truncated` in `note`.
4. A claim that states its own limits ("historical data, not a prediction", "a snapshot, may change") does
   not go beyond the evidence.
5. Claims from the alternatives agent often cite `plan:` evidence (the planner's arrival times, slack and
   verdict per option) and `case:` evidence (past cases). Treat both as evidence. A past case supports
   "this happened before", not "this will happen today".
6. When unsure: choose the most likely label and start `note` with `UNSURE:` and one sentence of reason.
   Do not label everything `correct` to save effort, and do not look for errors to appear strict.

## Output asked for

The filled sheet with the same columns and rows, nothing changed except `label` and `note` (this was
checked cell by cell on return: no other cell differs), and a short report: counts per label, the rows
marked `UNSURE:` with the five hardest, anything ambiguous in the rules, and any row left unlabelled.

## What came back

253 `correct`, 3 `wrong_number`, 3 `unsupported`, 1 `other_wrong`; 6 rows marked `UNSURE:`; no row
left unlabelled. Ambiguities the labeller reported:

1. A forecast claim that quotes the traveller's sub-window (18:00–00:00) inside a longer `valid:` window
   (18:00–06:00): counted as `correct` when written "from ... to ...", as `wrong_number` (with `UNSURE:`)
   when written "(valid ... to ...)".
2. `risk_downplayed` gives no rule for `moderate` on a large delay (a ground delay program averaging 148
   minutes, maximum 362, labelled `moderate` was left `correct`).
3. No threshold for loose terminology (scattered cloud layers called "ceilings" was labelled
   `other_wrong` with `UNSURE:`).
4. How far a past case may be generalised ("affects rail far less" turned into "avoids ... entirely" was
   labelled `unsupported`).

No human labelled a sample, so there is no agreement figure.
