# Flight claims lose their citations: the model writes `exact_evidence_ids`

## Context
- Repo: ca6123-travel-agent, Python, pytest, Google ADK. Only work inside this repository; do not read
  `.env`; no network.
- Symptom (real Gemini, 18 requests on recorded scenarios): in 10 of 15 turns with flight claims, **every**
  flight claim had an empty `evidence_ids`. The raw model output is fine except for the key name:
  ```json
  {"confidence": "high", "exact_evidence_ids": ["faa_nas:LGA:GDP:2"], "risk": null, "text": "..."}
  ```
- Cause: `travel_agent/agents/flight_airport.py` (prompt, near line 95) says
  "Each Claim has text, exact evidence_ids, confidence ...". The model reads "exact evidence_ids" as the key
  name. `travel_agent/agents/specialists.py::parse_agent_result` only reads `evidence_ids`, so the claims
  arrive uncited and the Verifier marks all of them unsupported.

## Task
1. `travel_agent/agents/flight_airport.py` prompt: make the output keys unambiguous. Replace that sentence
   with wording that names the JSON keys literally and adds one short example claim, e.g.
   `Each claim is a JSON object with exactly these keys: "text", "evidence_ids" (a list of evidence ids
   copied exactly from the tool results), "confidence" ("low", "medium" or "high") and "risk" ("low",
   "moderate", "high", or null for factual observations).` followed by an example like
   `{"text": "LGA has a ground delay program averaging 50 min.", "evidence_ids": ["faa_nas:LGA:GDP:0"],
   "confidence": "high", "risk": "moderate"}`. Keep every other instruction of the prompt unchanged.
2. `travel_agent/agents/weather.py` prompt: it currently says "Each claim has text, evidence_ids,
   confidence (...) and risk (...)". Quote the key names the same way (`"text"`, `"evidence_ids"`,
   `"confidence"`, `"risk"`); keep the rest unchanged.
3. `parse_agent_result` in `travel_agent/agents/specialists.py`: when a claim has no `evidence_ids` key
   (or it is null), fall back to the first present key among `exact_evidence_ids`, `evidence_id`,
   `evidence`, `citations`. The value is handled exactly like `evidence_ids` today (list of strings, or a
   single string). If `evidence_ids` is present and non-null it always wins, even when empty.
4. Tests:
   - `tests/test_orchestrator.py`: each alias key is accepted (list and single-string forms);
     `evidence_ids` wins over an alias; `evidence_ids: []` with an alias present stays empty; a claim with
     none of the keys has `evidence_ids == []`.
   - `tests/test_flight_airport.py` or `tests/test_m3_flight.py` (whichever already inspects the flight
     prompt or builds the agent): the prompt text contains `"evidence_ids"` in quotes and does not contain
     the phrase `exact evidence_ids`. Same quoted-key assertion for the weather prompt in
     `tests/test_m3_weather.py`.
   - An end-to-end style test with the existing scripted-model helpers where the flight model answers with
     `exact_evidence_ids`: the parsed flight claims carry the ids.

## Constraints
- Do not touch `travel_agent/verification/`, guardrails, `travel_agent/hitl/`, `alternatives.py`, eval files.
- No new dependencies. Do not spawn sub-agents. Do not commit.

## Acceptance
- New tests fail on the current code and pass after the change.
- `.venv/bin/python -m pytest -q` fully green.
