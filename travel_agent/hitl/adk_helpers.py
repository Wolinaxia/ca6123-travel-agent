"""Small helpers to run an ADK agent turn and move state in and out of a session.

Used by the M4 demo notebook. Kept separate from ``approval.py`` so the
approval logic stays a plain-Python, dict-based module that is easy to test.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.genai import types


def load_api_key() -> bool:
    """Make ``GOOGLE_API_KEY`` available without ever printing or hard-coding it.

    Order: already in the environment -> Colab Secrets -> a local ``.env`` file
    at the repo root (git-ignored). Returns True when a key is available.
    """
    if os.environ.get("GOOGLE_API_KEY"):
        return True
    try:  # Google Colab
        from google.colab import userdata  # type: ignore

        value = userdata.get("GOOGLE_API_KEY")
        if value:
            os.environ["GOOGLE_API_KEY"] = value
            return True
    except Exception:
        pass
    env_file = Path(__file__).parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
    return bool(os.environ.get("GOOGLE_API_KEY"))


async def run_turn(runner: Runner, user_id: str, session_id: str, text: str, verbose: bool = True) -> str:
    """Send one user message; print tool calls as they happen; return final text."""
    message = types.Content(role="user", parts=[types.Part(text=text)])
    final = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=message):
        for part in (event.content.parts if event.content and event.content.parts else []):
            if part.function_call and verbose:
                args = dict(part.function_call.args or {})
                print(f"  -> tool call: {part.function_call.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
            if part.text and event.is_final_response():
                final = part.text
    return final


async def get_state(runner: Runner, user_id: str, session_id: str) -> dict[str, Any]:
    """A plain-dict copy of the session state."""
    session = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
    return dict(session.state) if session else {}


async def push_state(runner: Runner, user_id: str, session_id: str, state: dict[str, Any], keys: list[str]) -> None:
    """Write selected keys back to the session (ADK's supported way: a state_delta event)."""
    session = await runner.session_service.get_session(app_name=runner.app_name, user_id=user_id, session_id=session_id)
    delta = {key: state.get(key) for key in keys}
    event = Event(author="hitl", actions=EventActions(state_delta=delta))
    await runner.session_service.append_event(session, event)
