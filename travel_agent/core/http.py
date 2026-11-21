"""Shared HTTP client with deterministic record/replay support."""

import hashlib
import json
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Callable, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from .config import load_dotenv
from .models import SourceResult


load_dotenv()

PayloadT = TypeVar("PayloadT")
_client = httpx.Client(headers={"User-Agent": f"travel-risk-proto ({os.getenv('TDA_CONTACT', 'set TDA_CONTACT')})"})
http_mode: ContextVar[str | None] = ContextVar("tda_http_mode", default=None)
snapshot_dir: ContextVar[str | None] = ContextVar("tda_snapshot_dir", default=None)

_state_lock = RLock()
_active_state: dict[str, str | None] = {"mode": None, "snapshot": None}


def set_active_state(mode: str | None, snapshot: str | None) -> None:
    """Process-wide record/replay state, visible to worker threads."""
    with _state_lock:
        _active_state["mode"] = mode
        _active_state["snapshot"] = snapshot


def active_mode() -> str | None:
    with _state_lock:
        return _active_state["mode"]


def active_snapshot() -> str | None:
    with _state_lock:
        return _active_state["snapshot"]


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"key", "api_key", "apikey", "token"}:
            value = "REDACTED"
        query.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _snapshot_path(method: str, url: str) -> Path:
    default_dir = Path(__file__).resolve().parents[1] / "snapshots"
    directory = Path(snapshot_dir.get() or active_snapshot() or os.getenv("TDA_SNAPSHOT_DIR", str(default_dir)))
    stable_key = hashlib.sha256(f"{method.upper()} {redact_url(url)}".encode()).hexdigest()
    return directory / f"{stable_key}.json"


def request(method: str, url: str, timeout: float = 10.0, **kwargs: object) -> tuple[int, str, str]:
    clean_url = redact_url(url)
    mode = (http_mode.get() or active_mode() or os.getenv("TDA_HTTP_MODE", "live")).lower()
    snapshot = _snapshot_path(method, url)
    if mode == "replay":
        if not snapshot.exists():
            raise RuntimeError(f"replay snapshot miss: {method.upper()} {clean_url}")
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        return int(payload["status"]), str(payload["body"]), clean_url
    try:
        response = _client.request(method, url, timeout=timeout, **kwargs)
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"request timed out for {clean_url}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"request failed for {clean_url}: {type(exc).__name__}") from exc
    if response.status_code >= 500:
        try:
            response = _client.request(method, url, timeout=timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"request retry failed for {clean_url}: {type(exc).__name__}") from exc
    if mode == "record":
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps({"status": response.status_code, "body": response.text, "url": clean_url}), encoding="utf-8")
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} for {clean_url}")
    return response.status_code, response.text, clean_url


def source_result(source: str, url: str, parser: Callable[[], PayloadT]) -> SourceResult[PayloadT]:
    try:
        data = parser()
    except Exception as exc:
        return SourceResult(
            source=source,
            ok=False,
            fetched_at=datetime.now(timezone.utc),
            url=redact_url(url),
            error=str(exc),
        )
    return SourceResult(
        source=source,
        ok=True,
        fetched_at=datetime.now(timezone.utc),
        url=redact_url(url),
        data=data,
    )
