"""User preference memory with consent (M4, Learn stage).

Design rules (these are the Responsible AI points for the video):

1. **Ask before saving.** Nothing learned from behaviour is stored silently.
   ``observe_rejection`` only produces a *suggestion*; it becomes a preference
   when the user confirms it (``confirm``).
2. **Transparent.** Every preference records where it came from
   (``user_stated`` / ``learned_from_feedback``), why, and when. The agent must
   say when a preference changed its recommendation.
3. **User control.** ``describe`` lists everything remembered; ``forget`` and
   ``forget_all`` delete it.
4. **Structured only.** Only a fixed set of typed keys is stored. Free text is
   never stored as a preference, so a stored preference cannot smuggle
   instructions into the prompt.

Storage is one JSON file per user in the git-ignored memory directory
(``TDA_MEMORY_DIR`` or ``<repo>/.memory``).
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent.memory.text_safety import looks_like_instruction
from travel_agent.schemas import Evidence, ToolResult

from .cases import memory_dir

SOURCE = "user_preferences"
DEFAULT_USER = "demo_user"
SIGNALS_NEEDED = 2  # repeated feedback needed before we *suggest* a preference
_LOCK = threading.Lock()

# The only keys that can be stored, with a short human description.
PREFERENCE_SPECS: dict[str, str] = {
    "avoid_redeye": "Avoid overnight (red-eye) flights",
    "max_drive_hours": "Longest acceptable drive, in hours",
    "prefer_rail_under_hours": "Prefer rail when it takes less than this many hours",
    "avoid_airports": "Airports to avoid (IATA codes)",
    "min_buffer_minutes": "Minimum arrival buffer before the deadline, in minutes",
}

_AIRPORT_NAMES = {
    "newark": "EWR", "laguardia": "LGA", "la guardia": "LGA", "kennedy": "JFK",
    "oakland": "OAK", "san jose": "SJC", "dulles": "IAD", "reagan": "DCA",
    "midway": "MDW", "o'hare": "ORD", "ohare": "ORD", "logan": "BOS",
}
_KNOWN_CODES = {"JFK", "LGA", "EWR", "SFO", "OAK", "SJC", "DCA", "IAD", "BWI", "ORD", "MDW",
                "BOS", "LAX", "BUR", "LGB", "SNA", "ONT", "SEA", "PDX", "DEN", "ATL", "DFW",
                "DAL", "IAH", "HOU", "MIA", "FLL", "PBI", "PHL", "HPN", "ISP"}

_NEG = r"(?:no|not|never|avoid|hate|don'?t(?:\s+want)?|do\s+not(?:\s+want)?|without|skip|rather\s+not)"
_EXPLICIT_RE = re.compile(r"\b(?:always|never|from now on|remember|in future|every time|all the time)\b", re.I)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr|hrs|hour|hours)", text, re.I)
    if match:
        return float(match.group(1))
    match = re.search(r"(\d+)\s*(?:min|mins|minutes)\b", text, re.I)
    if match:
        return round(float(match.group(1)) / 60.0, 2)
    return None


def _minutes(text: str) -> int | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hr|hrs|hour|hours)", text, re.I)
    if match:
        return int(round(float(match.group(1)) * 60))
    match = re.search(r"(\d+)\s*(?:min|mins|minutes)\b", text, re.I)
    if match:
        return int(match.group(1))
    return None


def parse_preference_statement(text: str, option: dict[str, Any] | None = None) -> list[tuple[str, Any, str]]:
    """Extract ``(key, value, why)`` candidates from a user sentence.

    Pure rules, no model call: cheap, predictable and testable. ``option`` is
    the option the user was reacting to (used for "too long", "too tight").
    """
    lowered = (text or "").lower()
    found: list[tuple[str, Any, str]] = []

    if re.search(r"red[\s-]?eye|overnight flight|fly overnight|night flight", lowered):
        if re.search(_NEG + r".{0,30}(?:red[\s-]?eye|overnight|night flight)", lowered) or re.search(
            r"(?:red[\s-]?eye|overnight|night flight).{0,20}(?:is|are)?\s*(?:awful|terrible|too tiring|exhausting|bad)",
            lowered,
        ):
            found.append(("avoid_redeye", True, "you said you avoid red-eye flights"))

    if re.search(r"driv", lowered):
        hours = _hours(lowered)
        if re.search(_NEG + r".{0,20}driv", lowered) and hours is None:
            found.append(("max_drive_hours", 0.0, "you said you do not want to drive"))
        elif hours is not None and re.search(r"driv.{0,40}(?:max|at most|under|no more than|less than|up to)|(?:max|at most|under|no more than|less than|up to).{0,40}driv", lowered):
            found.append(("max_drive_hours", hours, f"you said drives should be at most {hours:g} h"))

    if re.search(r"prefer.{0,15}(?:train|rail|amtrak)|(?:train|rail|amtrak).{0,15}(?:is|are)?\s*(?:better|preferred|nicer)", lowered):
        hours = _hours(lowered)
        found.append(("prefer_rail_under_hours", hours if hours is not None else 5.0, "you said you prefer rail"))

    codes: set[str] = set()
    for name, code in _AIRPORT_NAMES.items():
        if re.search(_NEG + r".{0,25}\b" + re.escape(name) + r"\b", lowered):
            codes.add(code)
    for code in re.findall(r"\b([A-Z]{3})\b", text or ""):
        if code in _KNOWN_CODES and re.search(_NEG + r".{0,25}\b" + code.lower() + r"\b", lowered):
            codes.add(code)
    if codes:
        found.append(("avoid_airports", sorted(codes), f"you said you avoid {', '.join(sorted(codes))}"))

    buffer_match = re.search(r"(?:at least|minimum|min\.?)\s+(\d+(?:\.\d+)?\s*(?:h\b|hr|hrs|hours?|min|mins|minutes))", lowered)
    if buffer_match and re.search(r"buffer|spare|early|before", lowered):
        minutes = _minutes(buffer_match.group(1))
        if minutes:
            found.append(("min_buffer_minutes", minutes, f"you asked for at least {minutes} min buffer"))
    elif re.search(r"too tight|cutting it close|too close|not enough (?:time|buffer)|too risky", lowered):
        found.append(("min_buffer_minutes", 120, "you rejected an option as too tight"))

    if option and re.search(r"too long|takes forever|too slow", lowered):
        mode = str(option.get("mode") or "")
        door = float(option.get("door_to_door_min") or 0)
        if mode == "drive" and door:
            found.append(("max_drive_hours", max(1.0, round(door / 60.0 - 1, 1)), "you rejected a drive as too long"))
        if mode == "rail" and door:
            found.append(("prefer_rail_under_hours", max(1.0, round(door / 60.0 - 1, 1)), "you rejected a train as too long"))
    return found


class PreferenceStore:
    """JSON-backed preference memory for one user."""

    def __init__(self, user_id: str = DEFAULT_USER, path: Path | None = None) -> None:
        raw = str(user_id or DEFAULT_USER)
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:40] or DEFAULT_USER
        if safe != raw:
            # Changed or shortened ids get a short hash of the ORIGINAL id, so
            # "alice.smith" and "alice_smith" never share one memory file.
            safe = f"{safe[:31]}-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:8]}"
        self.user_id = safe
        self.path = path or memory_dir() / f"preferences_{safe}.json"

    # -- storage ----------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"preferences": {}, "signals": {}, "pending": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        for key in ("preferences", "signals", "pending"):
            data.setdefault(key, {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -- validation -------------------------------------------------------
    @staticmethod
    def _validate(key: str, value: Any) -> Any:
        if key not in PREFERENCE_SPECS:
            raise ValueError(f"unknown preference '{key}'; allowed: {sorted(PREFERENCE_SPECS)}")
        if key == "avoid_redeye":
            return bool(value)
        if key in {"max_drive_hours", "prefer_rail_under_hours"}:
            number = float(value)
            if not 0 <= number <= 24:
                raise ValueError(f"{key} must be between 0 and 24 hours")
            return round(number, 2)
        if key == "min_buffer_minutes":
            number = int(value)
            if not 0 <= number <= 720:
                raise ValueError("min_buffer_minutes must be between 0 and 720")
            return number
        if key == "avoid_airports":
            items = value if isinstance(value, (list, tuple, set)) else [value]
            bad = [str(item) for item in items if not re.fullmatch(r"[A-Za-z]{3}", str(item).strip())]
            if bad or not items:
                raise ValueError(f"avoid_airports needs 3-letter airport codes, got {bad or 'nothing'}")
            return sorted({str(item).strip().upper() for item in items})
        raise ValueError(key)  # pragma: no cover

    # -- public API -------------------------------------------------------
    def get_all(self) -> dict[str, Any]:
        """``{key: value}`` of confirmed preferences only."""
        return {key: record["value"] for key, record in self._load()["preferences"].items()}

    def records(self) -> dict[str, dict[str, Any]]:
        """Confirmed preferences with provenance."""
        return dict(self._load()["preferences"])

    def set(self, key: str, value: Any, source: str = "user_stated", reason: str = "") -> dict[str, Any]:
        """Store a preference the user explicitly stated or confirmed."""
        clean = self._validate(key, value)
        with _LOCK:
            data = self._load()
            if key == "avoid_airports":
                previous = data["preferences"].get(key, {}).get("value") or []
                clean = sorted(set(previous) | set(clean))
            record = {"value": clean, "source": source, "reason": reason[:200], "saved_at": _now()}
            data["preferences"][key] = record
            data["signals"].pop(key, None)
            self._save(data)
        return record

    def forget(self, key: str) -> bool:
        """Delete one preference (and any pending suggestion / signals for it)."""
        with _LOCK:
            data = self._load()
            existed = data["preferences"].pop(key, None) is not None
            data["signals"].pop(key, None)
            data["pending"] = {sid: s for sid, s in data["pending"].items() if s.get("key") != key}
            self._save(data)
        return existed

    def forget_all(self) -> int:
        """Delete everything remembered about this user. Returns count removed."""
        with _LOCK:
            count = len(self._load()["preferences"])
            if self.path.exists():
                self.path.unlink()
        return count

    def pending(self) -> dict[str, dict[str, Any]]:
        return dict(self._load()["pending"])

    def remember_statement(self, text: str) -> list[dict[str, Any]]:
        """User said something like "remember I never want red-eyes".

        Returns suggestions (not saved yet) for the UI to confirm.
        """
        if looks_like_instruction(text or ""):
            return []
        return [self._suggest(key, value, why, explicit=True) for key, value, why in parse_preference_statement(text)]

    def observe_rejection(self, reason: str, option: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Learn from a rejected proposal.

        Each candidate preference found in the reason is counted. A suggestion
        is produced only when the same signal was seen ``SIGNALS_NEEDED`` times,
        or immediately when the user used an explicit word ("never", "always",
        "remember"). Nothing is saved until ``confirm`` is called.
        """
        if looks_like_instruction(reason or ""):
            return []
        explicit = bool(_EXPLICIT_RE.search(reason or ""))
        suggestions = []
        with _LOCK:
            data = self._load()
            current = data["preferences"]
            to_suggest = []
            for key, value, why in parse_preference_statement(reason, option):
                if key in current and current[key]["value"] == value:
                    continue
                signal = data["signals"].setdefault(key, {"count": 0, "value": value, "why": why})
                signal["count"] += 1
                signal["value"] = value
                signal["why"] = why
                if explicit or signal["count"] >= SIGNALS_NEEDED:
                    to_suggest.append((key, value, why, signal["count"]))
            self._save(data)
        for key, value, why, count in to_suggest:
            note = why if count < 2 else f"{why} ({count} times)"
            suggestions.append(self._suggest(key, value, note, explicit=explicit))
        return suggestions

    def _suggest(self, key: str, value: Any, why: str, explicit: bool) -> dict[str, Any]:
        clean = self._validate(key, value)
        suggestion = {
            "id": uuid.uuid4().hex[:8],
            "key": key,
            "value": clean,
            "why": why,
            "explicit": explicit,
            "question": f"Should I remember this preference? {PREFERENCE_SPECS[key]}: {clean} (because {why}).",
            "created_at": _now(),
        }
        with _LOCK:
            data = self._load()
            data["pending"] = {sid: s for sid, s in data["pending"].items() if s.get("key") != key}
            data["pending"][suggestion["id"]] = suggestion
            self._save(data)
        return suggestion

    def confirm(self, suggestion_id: str) -> dict[str, Any] | None:
        """User said yes: turn a pending suggestion into a stored preference."""
        with _LOCK:
            data = self._load()
            suggestion = data["pending"].pop(suggestion_id, None)
            self._save(data)
        if suggestion is None:
            return None
        source = "user_stated" if suggestion.get("explicit") else "learned_from_feedback"
        return self.set(suggestion["key"], suggestion["value"], source=source, reason=suggestion["why"])

    def dismiss(self, suggestion_id: str) -> bool:
        """User said no: drop the suggestion and reset its signal counter."""
        with _LOCK:
            data = self._load()
            suggestion = data["pending"].pop(suggestion_id, None)
            if suggestion:
                data["signals"].pop(suggestion["key"], None)
            self._save(data)
        return suggestion is not None

    def describe(self) -> list[str]:
        """Human-readable list for "show my preferences"."""
        lines = []
        for key, record in self.records().items():
            how = "you told me" if record["source"] == "user_stated" else "learned from your feedback, confirmed by you"
            lines.append(f"{PREFERENCE_SPECS[key]}: {record['value']} ({how}, {record['saved_at'][:10]})")
        return lines or ["No preferences remembered."]

    def prompt_summary(self) -> str:
        """Short, structured text for the agent's instruction (typed values only)."""
        prefs = self.get_all()
        if not prefs:
            return "No saved user preferences."
        return "; ".join(f"{key}={value}" for key, value in sorted(prefs.items()))


# --------------------------------------------------------------------------
# ADK tool
# --------------------------------------------------------------------------

def _user_id_from(tool_context: Any) -> str:
    try:
        state = getattr(tool_context, "state", None)
        if state is not None and state.get("user_id"):
            return str(state.get("user_id"))
    except Exception:
        pass
    try:
        user_id = getattr(tool_context, "user_id", None)
        if user_id:
            return str(user_id)
    except Exception:
        pass
    return DEFAULT_USER


def preference_evidence(store: PreferenceStore) -> list[Evidence]:
    """One evidence item per confirmed preference (cite these in claims)."""
    evidence = []
    fetched = _now()
    for key, record in store.records().items():
        value = record["value"]
        data: dict[str, Any] = {"key": key, "value": value, "pref_source": record["source"], "saved_at": record["saved_at"]}
        # Unit-bearing keys let the Verifier check numbers quoted in claims.
        if key == "max_drive_hours":
            data["max_drive_min"] = round(float(value) * 60)
        elif key == "prefer_rail_under_hours":
            data["rail_preferred_under_min"] = round(float(value) * 60)
        elif key == "min_buffer_minutes":
            data["min_buffer_min"] = int(value)
        evidence.append(
            Evidence(
                id=f"pref:{store.user_id}:{key}",
                source=SOURCE,
                subject="user_preferences",
                observation=f"User preference: {PREFERENCE_SPECS[key]} = {value}",
                data=data,
                fetched_at=fetched,
            )
        )
    return evidence


def get_user_preferences(tool_context: Any = None) -> dict:
    """Saved travel preferences of the current user (confirmed by the user).

    Call this before ranking options. Each evidence item is one preference,
    e.g. avoid_redeye=True or max_drive_hours=4. If a preference changes your
    recommendation, say so in a claim that cites the preference evidence id.
    An empty evidence list means no preferences are saved.
    """
    try:
        store = PreferenceStore(_user_id_from(tool_context))
        return ToolResult(source=SOURCE, ok=True, evidence=preference_evidence(store)).to_dict()
    except Exception as exc:
        return ToolResult(source=SOURCE, ok=False, evidence=[], error=str(exc)).to_dict()
