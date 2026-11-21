# Verification

The Verifier asks one deterministic question per claim: **is this supported by
the evidence the agent cited?**

## Pieces

| Module | Purpose |
|---|---|
| `store.py` | Evidence store in `ctx.state["evidence_store"]`: `{id: {"evidence", "tool_ok", "source"}}`. JSON-serialisable so it survives ADK state copies. |
| `checks.py` | `verify_claim` / `verify_claims`. Ordered, unit-aware checks, no LLM. |
| `../agents/verifier.py` | `verify_claims_tool` (ADK function tool) and `build_verifier_agent()`. |

## Checks (in order)

1. **Citation present** -- no `evidence_ids` -> `unsupported`.
2. **Citations exist** -- an unknown id -> `unsupported` (fabricated citation).
3. **Trusted tool call** -- evidence from a `tool_ok=False` result cannot support
   anything, exactly like a missing id -> `unsupported`.
4. **Not untrusted-only** -- every cited item is untrusted news -> `unsupported`.
   Untrusted items still count as "cited", but never as numeric/place support.
5. **Numbers match** -- claims and trusted evidence are compared by unit family:
   `minutes` (`min`, `h`, `"1 h 23 min"` -> 83, `90-minute`, number words
   one-twenty plus tens), `percent`, `knots`, `mph`, `km`, `temperature`.
   Evidence numbers come from `observation` and unit-implying `data` keys
   (`*_min`, `*minutes*`, `*_pct`, `*rate*`, `*_kt`/`gust*`, `*_km`, `*_mph`,
   `temp*`); ids, urls, timestamps, `rule`, coordinates and `source_status` are
   ignored. Tolerance `max(0.5, 5%)`. No same-family candidate -> `unsupported`;
   a candidate exists but none matches -> `contradicted`.
6. **Places match** -- airport codes match on word boundaries, are checked
   against the airport reference data, and common abbreviations (USA, PDT, ETA,
   ...) are ignored. A substring such as `SFOO` does not satisfy `SFO`.
7. **Risk consistent** -- if the claim carries `risk`, compare it with the
   highest `data.level` among cited findings **and** every summary for the same
   route in the store (even uncited). If the citation carries no route, also
   match summaries whose route contains an airport from the claim text or from
   a cited trusted subject (FAA `SFO` -> `JFK-SFO`). A lower claim is
   `contradicted` ("downplays rule-engine risk"); a higher one is `supported`
   with a "more cautious" note.

`verify_claims` returns `{"verdicts", "needs_human", "summary", "counts"}`.
`needs_human` is true if any verdict is not `supported`. If claims cite ids while
the store is empty, the summary names the likely wiring mistake.
`verify_claims_tool` persists the result in `state["last_verification"]`; the
HITL step must read that key, not the Verifier's prose.

## Wiring

The Verifier agent reads the store from `tool_context.state`, so evidence must
already be there. Wire `record_and_sanitize` on every agent that calls tools:

```python
from travel_agent.agents.verifier import build_verifier_agent
from travel_agent.guardrails.callbacks import record_and_sanitize

verifier = build_verifier_agent()   # no API key needed to construct

agent = LlmAgent(
    ...,
    after_tool_callback=record_and_sanitize,
)
```

The tool is called once per review with the specialists' claims:

```json
[{"text": "SFO arrivals are delayed about 34 min",
  "evidence_ids": ["faa_nas:SFO:GDP:0"],
  "confidence": "high",
  "risk": "moderate"}]
```

A semantic LLM judge may be added later at `checks.SEMANTIC_JUDGE_HOOK`; it is
deliberately empty for now.

## State keys

`evidence_store` (written by `record_and_sanitize`), `approved_actions` (HITL,
single-use), `completed_actions`, `last_verification` and `trace_run_id` are
documented in `docs/INTERFACES.md`.
