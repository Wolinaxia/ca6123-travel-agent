# Travel Disruption Agent — CA6123 Group Project

A goal-driven, multi-agent assistant for **U.S. domestic trips**. Instead of answering
"what's the weather at JFK?", it takes a goal like:

> *"I need to be in San Francisco by Friday 6pm, leaving from NYC.
> Find the least-risky option and suggest a rebooking if my flight looks bad."*

…and plans, checks live disruption signals, compares alternatives, and proposes an action.
**Nothing is booked without human approval.**

Built with **Google ADK + Gemini** (course stack). Deadline: **Fri 4 Dec 2026, 12:00 noon**.

---

## Architecture (one screen)

```
User goal
   │
   ▼
Orchestrator (intent routing + task breakdown)          ← Member 2
   │
   ├──► Flight & Airport agent ─┐                        ← Member 3
   ├──► Weather agent ──────────┤  each returns Claims   ← Member 3
   └──► Alternatives agent ─────┘  that cite Evidence    ← Member 4
                 │
                 ▼
        Verifier agent  ── unsupported claim? ──► escalate to human   ← Member 1
                 │
                 ▼
        Plan + recommended action ──► Human approval (HITL) ──► mock booking   ← Member 4
```

Every tool returns **Evidence** with an id. Every agent statement is a **Claim** that must
cite evidence ids. The Verifier checks claims against the actual tool outputs. This is the
contract that holds the whole system together. See [`docs/INTERFACES.md`](docs/INTERFACES.md).

## Mapping to the brief's 4-stage cycle

| Stage | What we show | Owner |
|---|---|---|
| Perceive | System prompts, context engineering (time-window evidence filtering), agentic RAG over airport/route reference data | M3 |
| Reason | Intent classification + routing, task breakdown for goal-driven requests | M2 |
| Action | Tool calling over 9+ free live data sources (FAA, NWS, aviation weather, BTS, Amtrak…), mock booking tool, optional MCP server | M3 / M4 |
| Learn | Few-shot from historical disruption cases, user preference memory | M4 |
| Human–AI interaction | Approval gate before any booking; escalation on unsupported claims | M4 / M1 |
| Responsible AI | Claim–evidence verification, two-layer prompt-injection filter on user/news text, PII redaction, deny-by-default booking gate, guardrail test set | M1 |

Bonus targets: **Agent Evaluation**, **Observability** (ADK callbacks → trace logs),
**Advanced guardrails**, optional **MCP**.

## Team & tasks

See [`docs/TASKS.md`](docs/TASKS.md) for the split, timeline, and video plan.

| Member | Owns | Presents in video |
|---|---|---|
| M1 (repo owner) | Data-layer tools (ported from prior prototype), Verifier agent, guardrails, evaluation | Responsible AI, Bonus, Conclusions |
| M2 | Orchestrator, intent routing, task breakdown | Overview, Reason |
| M3 | Flight & Airport agent, Weather agent, context engineering | Perceive, Action |
| M4 | Alternatives agent, HITL approval flow, few-shot + memory | Learn, Human–AI interaction |

## Repo layout

```
travel_agent/
  schemas.py        # Evidence / Claim / AgentResult — the shared contract (start here)
  tools/            # data-source tools (M1 ports these)
  agents/           # orchestrator, specialists, verifier
  guardrails/       # input/output filters, PII redaction, injection filter (M1)
  memory/           # user preferences + few-shot case store (M4)
eval/               # evaluation set + scripts (M1)
notebooks/          # the demo notebook we submit
tests/
docs/
```

## Setup

```bash
git clone https://github.com/Wolinaxia/ca6123-travel-agent.git
cd ca6123-travel-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # put YOUR OWN Gemini key in .env, never commit it
pytest
```

In Colab: put `GOOGLE_API_KEY` in Colab Secrets (same as the course notebooks).

## Rules

- **Never commit API keys.** This repo is public. `.env` is git-ignored.
- Use `gemini-2.5-flash-lite` while developing — our token budget is shared and limited.
- One branch per task, open a PR, at least one teammate reviews. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

- The data-source layer is adapted from a personal prototype by M1, built before this course
  (reuse cleared with the course instructor). The agentic architecture is new for this project.
- AI tools used in development will be listed in the final slide of the video, as the brief requires.
