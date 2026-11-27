# Weather claims are dropped when the model writes `confidence` as a number

## Context
- Repo: ca6123-travel-agent, Python, pytest, Google ADK. Only work inside this repository (this directory);
  do not read files outside it, do not read `.env`, no network.
- Symptom (real Gemini runs, scenario `nyc_rain_20261122_1708z`, 4 of 4 runs): the Weather agent calls its
  tools successfully and answers with valid claims, but writes `"confidence": 1.0` / `0.9`.
  `travel_agent/agents/specialists.py::parse_agent_result` requires `confidence` to be a string or null;
  on the first non-string it returns `{"claims": [], "needs_human": True, "error": "could not parse weather
  output (invalid claim fields)"}`, so every weather claim is lost.
- Cause 1: the parser discards the whole result for one mistyped field.
- Cause 2: `travel_agent/agents/weather.py` (prompt near line 68) lists the claim fields but never says
  which values `confidence` and `risk` may take. `flight_airport.py` (line ~95) and `alternatives.py` do.

## Task
1. `parse_agent_result` in `travel_agent/agents/specialists.py`:
   - A numeric `confidence` (int or float, not bool) is mapped to a level: `>= 0.8` -> `"high"`,
     `>= 0.5` -> `"medium"`, otherwise `"low"`. Values above 1 are treated as percentages when `<= 100`
     (divide by 100); anything else numeric -> `"low"`.
   - A string `confidence` is matched case-insensitively after `.strip()` (`"High"` -> `"high"`); unknown
     strings stay `"low"` as today.
   - Any other type (list, dict, bool) -> `"low"`. A mistyped `confidence` must never discard the claim or
     the result.
   - `risk`: a string is matched case-insensitively after `.strip()`; a non-string, non-null `risk` becomes
     `None` for that claim instead of discarding the whole result. (Unknown strings already become `None`.)
   - Nothing else about the function changes (text/evidence_ids/unavailable/needs_human handling).
2. `travel_agent/agents/weather.py`: in the prompt, state the allowed values the same way the flight agent
   does: confidence is one of `low`, `medium`, `high` (a word, not a number); risk is one of `low`,
   `moderate`, `high`, or null. Keep the rest of the prompt unchanged.
3. Tests:
   - `tests/test_orchestrator.py` (or the file that already tests `parse_agent_result`; find it): numeric
     confidences `1.0`, `0.9`, `0.5`, `0.2`, `85`, `True`, `[1]` map as specified and the claims are kept;
     `"High"` -> `"high"`; a non-string risk (`2`) gives `risk None` with the claim kept; a result where one
     claim has a numeric confidence and another is fine keeps both.
   - `tests/test_m3_weather.py`: an end-to-end style test with the existing scripted-model helpers where
     the weather model returns claims with `"confidence": 1.0`: the parsed weather result has the claims
     and no `error`. Also assert the weather prompt text mentions `low`, `medium` and `high`.
   - If an existing test asserts the old "invalid claim fields" behaviour for mistyped confidence/risk,
     update that test to the new behaviour and say so in your final message.

## Constraints
- Do not touch `travel_agent/verification/`, guardrails, `travel_agent/hitl/`, `alternatives.py`, eval files.
- No new dependencies. Do not spawn sub-agents. Do not commit.

## Acceptance
- New tests fail on the current code and pass after the change.
- `.venv/bin/python -m pytest -q` fully green.
