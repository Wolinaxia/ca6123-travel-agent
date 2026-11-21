# Tasks, Timeline, Video Plan

Today: Sat 21 Nov. Submission: **Fri 4 Dec, 12:00 noon**. Peer evaluation: Tue 8 Dec, 12:00 noon.

## Who does what

### M1 — Data layer, Verifier, Guardrails, Evaluation (repo owner)
- [ ] Port the data sources into `travel_agent/tools/` as plain functions returning `ToolResult` (see INTERFACES.md). **Target: Tue 24 Nov**, so others can switch from mocks to real tools.
- [ ] Verifier agent: checks every `Claim` against the cited `Evidence`; unsupported → escalate to human
- [ ] Guardrails: prompt-injection filter on news text, PII redaction on user input/output, off-topic blocking
- [ ] Guardrail test set (attacks that should be blocked + benign inputs that must pass)
- [ ] Evaluation: historical disruption cases (with negative controls) → scored runs
- [ ] Observability: ADK callbacks write a trace log per run
- [ ] **Ablation: with vs without Verifier** (our main novelty evidence for the video)
  - Same fixed task set (~15–20 goal-driven requests, run on recorded demo scenarios so results are reproducible and cheap)
  - Run the full system twice: Verifier ON vs OFF (everything else identical, same model and seed/temperature 0)
  - Per run, count: total claims, claims with no evidence, claims citing non-existent evidence, wrong numbers/airports, risk downplayed vs the rule engine
  - Report: how many bad claims reach the user without the Verifier vs with it, plus how often the Verifier escalates to a human (and false escalations on correct claims)
  - Output one table + one chart for the Responsible AI / Novelty segment; script in `eval/ablation.py`, results in `eval/results/`
  - Depends on: orchestrator + specialists working end-to-end (M2–M4), so run it Fri 27 – Sun 29 Nov

### M2 — Orchestrator & Reasoning
- [ ] Root agent in ADK; intent classes: `assess_trip` / `plan_trip` / `rebook` / `out_of_scope`
- [ ] Task breakdown for goal-driven requests (deadline, origin, destination, constraints → sub-tasks)
- [ ] Routing to specialist agents (sequential vs parallel — decide and justify)
- [ ] Final answer assembly from specialists' `AgentResult`s
- [ ] Start with **mock tools** from `tools/mock.py` until M1's real tools land

### M3 — Flight & Airport agent, Weather agent, Perceive stage
- [ ] Flight & Airport agent: delay programs, on-time history, multi-airport comparison (e.g. JFK / LGA / EWR)
- [ ] Weather agent: forecasts, alerts, aviation weather
- [ ] Context engineering: only pass evidence that is valid for the trip's time window (reduces hallucination)
- [ ] System prompts for both agents; every output as `Claim`s citing evidence ids
- [ ] (Bonus, optional) agentic RAG over airport/route reference data

### M4 — Alternatives, Human-in-the-loop, Learn stage
- [ ] Alternatives agent: other airports, driving, rail (when the flight option looks bad)
- [ ] Mock booking/rebooking tool
- [ ] HITL approval gate: the agent proposes, the human approves or rejects, then the mock booking runs
- [ ] Few-shot examples from past disruption cases in prompts
- [ ] User preference memory (e.g. "no red-eyes", "prefers rail under 4h")

### Everyone
- [ ] Set up Colab + own Gemini key, run the course's API key test notebook
- [ ] Write your section of the demo notebook with clear markdown (graded!)
- [ ] Your video segment: slides + script, **timed with a stopwatch**

## Timeline

| Dates | Milestone |
|---|---|
| Sat 21 – Mon 23 Nov | Everyone: ADK hello-world in Colab; agents built against mock tools |
| Tue 24 Nov | M1: real tools merged |
| Wed 25 – Fri 27 Nov | Integration: orchestrator ↔ specialists ↔ verifier; HITL; guardrails |
| Fri 27 – Sun 29 Nov | M1: ablation runs (Verifier ON vs OFF), needs the full pipeline working |
| **Sun 29 Nov** | **Feature freeze.** Only bug fixes after this |
| Mon 30 Nov – Tue 1 Dec | Evaluation runs, record demo snapshots, slides, scripts |
| Wed 2 – Thu 3 Dec | Record + edit video; final notebook clean-up |
| **Fri 4 Dec, before 12:00** | Upload video (YouTube) + code zip (group number in filename). **Submit once.** |
| Tue 8 Dec, before 12:00 | Everyone does peer evaluation (or you get 0 for it) |

## Video plan (hard limit 12:00, aim for 11:30)

Everyone must appear on screen and speak in their own voice (no TTS).

| # | Section | Who | Time |
|---|---|---|---|
| 1 | Overview: motivation, goal, workflow diagram | M2 | 1:00 |
| 2 | Perceive | M3 | 1:30 |
| 3 | Reason | M2 | 1:30 |
| 4 | Action | M3 | 1:30 |
| 5 | Learn | M4 | 1:15 |
| 6 | Human–AI interaction | M4 | 1:15 |
| 7 | Responsible AI (guardrails + tests + evaluation + Verifier ablation table) | M1 | 2:00 |
| 8 | Bonus features + conclusions + contributions + AI-tool acknowledgement | M1 + all | 1:30 |

Slides: large fonts, little text (wordy slides are penalised under Clarity). Show live demos where possible.

## Grading reminder

Technical 25 · Correctness & Clarity 25 · **Responsible AI 25** · Novelty 10 · **Bonus 15**.
Code is graded on readability too: clear headings, comments, markdown explanations.
