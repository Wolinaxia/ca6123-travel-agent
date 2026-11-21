# How we work

1. Pick your task in `docs/TASKS.md` (or the GitHub issue for it).
2. Branch from `main`: `git checkout -b m3/weather-agent`
3. Commit small, push, open a Pull Request into `main`.
4. One teammate reviews. Run `pytest` before asking for review.
5. Merge only when tests pass.

Conventions
- Agents and tools follow `docs/INTERFACES.md`. If you need to change the contract, raise it in the group first: everyone depends on it.
- Docstrings on every tool function: ADK uses them to tell the model what the tool does.
- Comments explain **why**, not what. The code is graded on clarity.
- No API keys in code, notebooks, or outputs. Clear notebook outputs that print keys before committing.
- Use `gemini-3.5-flash-lite` while developing.
