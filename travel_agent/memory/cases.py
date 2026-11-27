"""Past disruption case library + agentic RAG search tool (M4, Learn stage).

What this module does
---------------------
* Keeps a small library of past disruption cases: the seed file
  ``travel_agent/data/cases/disruption_cases.json`` plus cases *learned* at run
  time from human decisions (stored in ``<memory dir>/cases_learned.json``).
* Exposes ``search_disruption_cases(query, top_k)`` as an ADK tool. The agent
  decides what to search for, reads ``match_quality`` and may search again with
  different words. That loop is what makes the retrieval *agentic*.
* Provides ``few_shot_block(query)`` for static few-shot prompting.

Retrieval
---------
With a Gemini key: dense (meaning-based) retrieval with ``gemini-embedding-001``
over all cases, using the RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT task types, with
a similarity floor so off-topic queries return nothing. On the held-out test
half of the generated query set this beat keyword search clearly (see
``eval/results/m4_latest.md``).
Then Gemini (flash-lite by default; ``TDA_RERANK_MODEL=gemini-4-hydrogen`` for the
slower, stronger model) re-orders the 10 closest cases by kind of cause, scope and place (``_rerank``; switch off with
``TDA_CASE_RERANK=0``). The ADK tool also adds the live FAA status already fetched in
the session to the search text. Both were chosen on the round-2 dev set and scored
once on the blind test set (``eval/results/m4_latest.md``); the model only returns case ids, which are
checked against the candidates, so it cannot add text or new cases. If the
re-rank call fails, the embedding order is kept.
Without a key, or with ``TDA_CASE_EMBEDDINGS=0`` (offline tests), or on any
embedding error: BM25 keyword retrieval with a travel-jargon synonym map and a
boost for matching airport codes. ``retriever`` in each result says which ran.

Responsible AI
--------------
Every case is returned as evidence with ``data.untrusted=True``. A past case may
*guide* the agent, but the Verifier (``verification/checks.py``) refuses a claim
supported **only** by untrusted evidence, so a past case can never be used as
proof of today's conditions. Learned cases are PII-redacted and rejected if they
look like a prompt injection, because their text comes from users.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from travel_agent.memory.text_safety import looks_like_instruction
from travel_agent.guardrails.pii import redact_pii
from travel_agent.schemas import Evidence, ToolResult

SOURCE = "case_library"
SEED_PATH = Path(__file__).parents[1] / "data" / "cases" / "disruption_cases.json"
_LOCK = threading.Lock()

# BM25 parameters (standard defaults).
_K1 = 1.5
_B = 0.75

# Travel jargon -> extra search words. Keeps simple keyword search useful when
# the agent says "fog" and the case says "low ceilings".
_SYNONYMS: dict[str, list[str]] = {
    "fog": ["ceilings", "visibility", "low"],
    "ceiling": ["fog", "visibility"],
    "ceilings": ["fog", "visibility"],
    "gdp": ["ground", "delay", "program"],
    "edct": ["ground", "delay", "program"],
    "stop": ["ground", "stop"],
    "train": ["rail", "amtrak"],
    "rail": ["train", "amtrak"],
    "amtrak": ["rail", "train"],
    "car": ["drive", "driving"],
    "drive": ["driving", "road"],
    "storm": ["thunderstorm", "thunderstorms", "winter", "snow"],
    "thunderstorm": ["storm", "thunderstorms"],
    "snow": ["winter", "storm", "deicing"],
    "ice": ["winter", "deicing"],
    "redeye": ["red-eye", "overnight"],
    "overnight": ["red-eye", "redeye"],
    "late": ["delay", "delays", "delayed"],
    "delay": ["delays", "delayed"],
    "airport": ["airports", "alternate"],
    "alternate": ["airport", "neighbouring", "metro"],
    "smoke": ["visibility", "wildfire"],
    "hurricane": ["tropical", "closure", "closed"],
    "tropical": ["hurricane"],
    "deadline": ["meeting", "evening", "morning"],
    "outage": ["failure", "failed", "down"],
    "down": ["outage", "failure"],
    "computer": ["software", "system", "outage"],
    "software": ["computer", "system", "outage"],
    "glitch": ["outage", "software", "system"],
    "crash": ["outage", "failure"],
    "notam": ["faa", "system", "outage", "nationwide"],
    "nationwide": ["national"],
    "meltdown": ["airline", "cancellation", "crew"],
    "cancel": ["cancellation", "cancelled"],
    "cancelled": ["cancellation", "cancel"],
    "canceled": ["cancelled", "cancellation"],
    "radar": ["atc", "control", "outage"],
    "controller": ["atc", "staffing", "control"],
    "staffing": ["controller", "atc", "shortage"],
    "shutdown": ["government", "cuts", "staffing"],
    "cuts": ["reduction", "reduce", "capacity"],
    "hail": ["thunderstorm", "storm"],
    "wires": ["power", "rail", "outage"],
    "power": ["wires", "outage"],
    "connection": ["connecting", "hub"],
    "connecting": ["connection", "hub"],
    "hub": ["connecting", "connection"],
    "blizzard": ["winter", "snow", "storm"],
    "cold": ["winter", "storm"],
    # Common city / airport names -> IATA codes used in the cases.
    "newark": ["ewr"],
    "laguardia": ["lga"],
    "kennedy": ["jfk"],
    "oakland": ["oak"],
    "francisco": ["sfo"],
    "jose": ["sjc"],
    "chicago": ["ord", "mdw"],
    "boston": ["bos"],
    "denver": ["den"],
    "seattle": ["sea"],
    "portland": ["pdx"],
    "miami": ["mia"],
    "washington": ["dca", "iad", "bwi"],
    "dulles": ["iad"],
    "baltimore": ["bwi"],
    "atlanta": ["atl"],
    "dallas": ["dfw"],
    "angeles": ["lax"],
    "tampa": ["tpa"],
    "orlando": ["mco"],
    "sarasota": ["srq"],
    "florida": ["tpa", "mco"],
    "philadelphia": ["phl"],
    "charlotte": ["clt"],
    "delta": ["airline"],
    "southwest": ["airline"],
    "united": ["airline"],
    "american": ["airline"],
}

_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "by", "is",
    "are", "was", "were", "be", "with", "from", "it", "this", "that", "my", "i",
    "me", "we", "our", "you", "your", "as", "but", "if", "so", "than", "then",
    "there", "what", "which", "who", "how", "when", "need", "needs", "get",
    "about", "please", "can", "could", "would", "should", "will", "do", "does",
    "did", "any", "some", "just", "also", "like", "want", "write", "tell", "give",
    "into", "about", "have", "has", "had", "its", "their", "them", "they", "he",
    "she", "his", "her", "us", "am", "not", "no", "yes", "very", "more", "most",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)?")


def memory_dir() -> Path:
    """Directory for run-time memory files (git-ignored ``.memory/`` by default)."""
    configured = os.environ.get("TDA_MEMORY_DIR")
    path = Path(configured) if configured else Path(__file__).parents[2] / ".memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _learned_path() -> Path:
    return memory_dir() / "cases_learned.json"


def _stem(word: str) -> str:
    """Very light stemming: plural 's' only (delays -> delay, airports -> airport)."""
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _tokens(text: str) -> list[str]:
    words = [_stem(word) for word in _TOKEN_RE.findall((text or "").lower()) if word not in _STOPWORDS]
    return words


def _expand(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(_stem(word) for word in _SYNONYMS.get(token, []))
        if "-" in token:
            expanded.extend(token.split("-"))
    return expanded


def _case_text(case: dict[str, Any]) -> str:
    fields = ("title", "disruption", "situation", "decision", "outcome", "lesson")
    parts = [str(case.get(field) or "") for field in fields]
    parts.append(" ".join(case.get("airports") or []))
    parts.append(" ".join(case.get("modes") or []))
    return " ".join(parts).replace("_", " ")


# --------------------------------------------------------------------------
# Loading and saving
# --------------------------------------------------------------------------

def load_seed_cases() -> list[dict[str, Any]]:
    """Cases shipped with the repo (recorded scenarios + illustrative cases)."""
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    return list(payload.get("cases") or [])


def load_learned_cases() -> list[dict[str, Any]]:
    """Cases learned at run time from human decisions (may be empty)."""
    path = _learned_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [item for item in data if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError):
        return []


def all_cases() -> list[dict[str, Any]]:
    return load_seed_cases() + load_learned_cases()


def record_case(
    title: str,
    situation: str,
    decision: str,
    outcome: str,
    lesson: str,
    airports: list[str] | None = None,
    modes: list[str] | None = None,
    disruption: str = "unspecified",
) -> dict[str, Any]:
    """Add a learned case to the library (called by the HITL step, not the LLM).

    Text fields are PII-redacted. If any field looks like a prompt injection
    the case is rejected: learned text comes from users and will later be shown
    to the model. Returns ``{"ok": bool, "case": dict | None, "error": str | None}``.
    """
    fields = {
        "title": title,
        "situation": situation,
        "decision": decision,
        "outcome": outcome,
        "lesson": lesson,
    }
    cleaned: dict[str, str] = {}
    for name, value in fields.items():
        text = str(value or "").strip()[:600]
        if looks_like_instruction(text):
            return {"ok": False, "case": None, "error": f"{name} looks like a prompt injection; case not saved"}
        cleaned[name], _ = redact_pii(text)
    stamp = datetime.now(timezone.utc).isoformat()
    digest = hashlib.sha1((cleaned["title"] + stamp).encode("utf-8")).hexdigest()[:10]
    case = {
        "id": f"learned_{digest}",
        "provenance": "learned",
        "learned_at": stamp,
        "disruption": re.sub(r"[^a-z0-9_]", "_", disruption.lower())[:40] or "unspecified",
        "airports": sorted({code.upper() for code in (airports or []) if re.fullmatch(r"[A-Za-z]{3}", code)}),
        "modes": sorted({mode.lower() for mode in (modes or []) if mode.isalpha() or "_" in mode}),
        **cleaned,
    }
    with _LOCK:
        existing = load_learned_cases()
        existing.append(case)
        _learned_path().write_text(json.dumps(existing, indent=2), encoding="utf-8")
    _INDEX.invalidate()
    return {"ok": True, "case": case, "error": None}


def forget_learned_cases() -> int:
    """Delete every learned case. Returns how many were removed."""
    with _LOCK:
        count = len(load_learned_cases())
        path = _learned_path()
        if path.exists():
            path.unlink()
    _INDEX.invalidate()
    return count


# --------------------------------------------------------------------------
# Retrieval index
# --------------------------------------------------------------------------

@dataclass
class _Hit:
    case: dict[str, Any]
    score: float
    coverage: float
    matched: list[str]


class _Bm25Index:
    """Tiny in-memory BM25 index, rebuilt when the learned cases change."""

    def __init__(self) -> None:
        self._cases: list[dict[str, Any]] | None = None
        self._docs: list[list[str]] = []
        self._df: dict[str, int] = {}
        self._avgdl = 1.0

    def invalidate(self) -> None:
        self._cases = None

    def _build(self) -> None:
        self._cases = all_cases()
        self._docs = [_tokens(_case_text(case)) for case in self._cases]
        self._df = {}
        for doc in self._docs:
            for term in set(doc):
                self._df[term] = self._df.get(term, 0) + 1
        self._avgdl = (sum(len(doc) for doc in self._docs) / len(self._docs)) if self._docs else 1.0

    def search(self, query: str) -> list[_Hit]:
        if self._cases is None:
            self._build()
        assert self._cases is not None
        base_terms = _tokens(query)
        terms = _expand(base_terms)
        query_codes = {token.upper() for token in terms if len(token) == 3 and token.isalpha()}
        total = len(self._docs) or 1
        hits: list[_Hit] = []
        for case, doc in zip(self._cases, self._docs):
            if not doc:
                continue
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            score = 0.0
            matched: list[str] = []
            for term in set(terms):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total - self._df.get(term, 0) + 0.5) / (self._df.get(term, 0) + 0.5))
                score += idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * len(doc) / self._avgdl))
                matched.append(term)
            airport_overlap = query_codes.intersection(case.get("airports") or [])
            score += 1.5 * len(airport_overlap)
            # Coverage: share of the agent's own query words found in the case.
            # A word found only through a synonym counts half, so "hurricane"
            # does not strongly match a winter storm case via "storm".
            unique_terms = set(base_terms)
            covered = 0.0
            for term in unique_terms:
                if counts.get(term):
                    covered += 1.0
                elif any(counts.get(_stem(s)) for s in _SYNONYMS.get(term, [])):
                    covered += 0.5
            coverage = covered / len(unique_terms) if unique_terms else 0.0
            if score > 0:
                hits.append(_Hit(case=case, score=score, coverage=coverage, matched=sorted(set(matched))))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits


_INDEX = _Bm25Index()


# Dense retrieval thresholds (cosine similarity, gemini-embedding-001 with
# RETRIEVAL_QUERY / RETRIEVAL_DOCUMENT task types). Calibrated on the *dev*
# half of eval/data/m4_generated_queries.json: travel queries scored >= 0.69,
# off-topic queries <= 0.60, so below 0.64 nothing is returned.
_DENSE_MIN = 0.64
_DENSE_PARTIAL = 0.70
_DENSE_STRONG = 0.74
_VECTOR_CACHE: dict[str, list[float]] = {}
_QUERY_VECTORS: dict[str, list[float]] = {}  # in-process only (queries may hold user text)


def embeddings_enabled() -> bool:
    """Dense retrieval is used when a Gemini key is available.

    ``TDA_CASE_EMBEDDINGS=0`` forces keyword-only search (offline tests);
    ``=1`` forces dense search; unset means "use it if a key is present".
    """
    flag = os.environ.get("TDA_CASE_EMBEDDINGS")
    if flag == "0":
        return False
    if flag == "1":
        return True
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _dense_rank(query: str, cases: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float]] | None:
    """Rank ALL cases by embedding similarity, or ``None`` when unavailable.

    Case vectors are cached in memory and in ``<memory dir>/case_embeddings.json``,
    so each case is embedded once. Any error returns ``None`` and the caller
    falls back to BM25 keyword search.
    """
    if not embeddings_enabled() or not cases:
        return None
    try:
        from google import genai  # imported lazily: optional path only
        from google.genai import types

        model = os.environ.get("TDA_EMBEDDING_MODEL", "gemini-embedding-001")
        cache_path = memory_dir() / "case_embeddings.json"
        if not _VECTOR_CACHE and cache_path.exists():
            _VECTOR_CACHE.update(json.loads(cache_path.read_text(encoding="utf-8")))
        texts = [_case_text(case) for case in cases]
        keys = [hashlib.sha1((model + "|doc|" + text).encode("utf-8")).hexdigest() for text in texts]
        missing = [(key, text) for key, text in zip(keys, texts) if key not in _VECTOR_CACHE]
        client = genai.Client()
        for start in range(0, len(missing), 50):
            batch = missing[start:start + 50]
            response = client.models.embed_content(
                model=model, contents=[text for _, text in batch],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            for (key, _), item in zip(batch, response.embeddings):
                _VECTOR_CACHE[key] = list(item.values)
        if missing:
            cache_path.write_text(json.dumps(_VECTOR_CACHE), encoding="utf-8")
        query_key = model + "|query|" + query
        if query_key not in _QUERY_VECTORS:
            _QUERY_VECTORS[query_key] = list(client.models.embed_content(
                model=model, contents=[query],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")).embeddings[0].values)
        query_vector = _QUERY_VECTORS[query_key]
        ranked = [(case, _cosine(query_vector, _VECTOR_CACHE[key])) for case, key in zip(cases, keys)]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked
    except Exception:
        return None


_RERANK_PROMPT = """A traveller wrote: "{query}"
Below are past travel disruptions. Rank them by how similar each one is to the traveller's situation:
the same KIND of cause matters most (e.g. an airline's own IT failure vs an air traffic control failure vs
weather), then the scope (one airline / one airport / nationwide), then the place and mode.
{candidates}
Return JSON: {{"ranking": ["id", ...]}} with every id exactly once, most similar first."""
_RERANK_CACHE: dict[str, list[str]] = {}
RERANK_ERRORS = {"count": 0, "last": ""}  # read by the eval so a silent fallback cannot hide
RERANK_USED: dict[str, int] = {}          # which model actually answered (the eval reports it)
RERANK_POOL = 10
# Default: flash-lite (fast, < 1 s, available on every key). The stronger thinking model
# gemini-4-hydrogen was also tested (blind test with live status: hit@1 66%, family@1 94%;
# eval/results/m4_latest.md) but takes ~5 s a call and needs extra access; opt in with
# TDA_RERANK_MODEL=gemini-4-hydrogen.
RERANK_MODEL = "gemini-3.5-flash-lite"
RERANK_FALLBACK_MODEL = "gemini-3.5-flash-lite"


def rerank_enabled() -> bool:
    return os.environ.get("TDA_CASE_RERANK", "1") != "0"


def _rerank(query: str, ranked: list[tuple[dict[str, Any], float]]) -> list[tuple[dict[str, Any], float]]:
    """Let Gemini re-order the closest cases; returns ``ranked`` unchanged on any problem.

    Safety: the model sees only case ids, titles and cause types, and must answer
    with ids. Anything that is not one of the candidate ids is ignored, so the
    re-ranker can change the ORDER of real cases but never their content.
    """
    pool = ranked[:RERANK_POOL]
    if len(pool) < 2 or not rerank_enabled():
        return ranked
    ids = [str(case.get("id")) for case, _ in pool]
    wanted = os.environ.get("TDA_RERANK_MODEL", RERANK_MODEL)
    key = hashlib.sha1((wanted + "|" + query + "|" + "|".join(ids)).encode("utf-8")).hexdigest()
    try:
        if key not in _RERANK_CACHE:
            from google import genai
            from google.genai import types

            lines = chr(10).join(f"- {case.get('id')}: {case.get('title')} (cause type: {case.get('disruption')}; "
                               f"modes: {', '.join(case.get('modes') or [])})" for case, _ in pool)
            client = genai.Client()  # keep a reference: a temporary client is closed before the call
            prompt = _RERANK_PROMPT.format(query=query.replace('"', "'"), candidates=lines)
            config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
            response = None
            for model in dict.fromkeys([wanted, RERANK_FALLBACK_MODEL]):
                try:
                    response = client.models.generate_content(model=model, contents=prompt, config=config)
                    RERANK_USED[model] = RERANK_USED.get(model, 0) + 1
                    break
                except Exception as exc:  # e.g. quota on the stronger model: try the fallback
                    RERANK_ERRORS["last"] = f"{model}: {str(exc)[:160]}"
            if response is None:
                raise RuntimeError(RERANK_ERRORS["last"])
            answer = json.loads(response.text)
            if isinstance(answer, dict):
                answer = answer.get("ranking") or []
            _RERANK_CACHE[key] = [str(item) for item in answer if isinstance(item, str)]
        order = [cid for cid in dict.fromkeys(_RERANK_CACHE[key]) if cid in ids]
        order += [cid for cid in ids if cid not in order]
        by_id = {str(case.get("id")): (case, score) for case, score in pool}
        return [by_id[cid] for cid in order] + ranked[RERANK_POOL:]
    except Exception as exc:
        RERANK_ERRORS["count"] += 1
        RERANK_ERRORS["last"] = str(exc)[:200]
        return ranked


def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Rank cases for ``query``. Returns plain dicts (case + score + quality).

    Uses Gemini embeddings (meaning-based) when available, otherwise BM25
    keywords. ``retriever`` in each result says which one was used.
    """
    top_k = max(1, min(5, int(top_k or 3)))
    hits = _INDEX.search(query)
    dense = _dense_rank(query, all_cases())
    if dense is not None:
        # Off-topic gate first (similarity floor), THEN re-rank only real candidates.
        dense = _rerank(query, [(case, score) for case, score in dense if score >= _DENSE_MIN])
        coverage = {id(hit.case): hit for hit in hits}
        lexical = {str(hit.case.get("id")): hit for hit in hits}
        results = []
        for case, similarity in dense:
            if similarity < _DENSE_MIN or len(results) >= top_k:
                break
            quality = "strong" if similarity >= _DENSE_STRONG else "partial" if similarity >= _DENSE_PARTIAL else "weak"
            hit = lexical.get(str(case.get("id"))) or coverage.get(id(case))
            results.append({
                "case": case,
                "score": round(similarity, 3),
                "coverage": round(hit.coverage, 2) if hit else 0.0,
                "match_quality": quality,
                "matched_terms": hit.matched if hit else [],
                "retriever": "gemini_embeddings",
            })
        return results
    retriever = "bm25"
    results = []
    best_score = hits[0].score if hits else 0.0
    for hit in hits[:top_k]:
        # "strong" needs most query words AND a score close to the best hit.
        if hit.coverage >= 0.6 and hit.score >= 0.5 * best_score:
            quality = "strong"
        elif hit.coverage >= 0.3:
            quality = "partial"
        else:
            quality = "weak"
        results.append(
            {
                "case": hit.case,
                "score": round(hit.score, 3),
                "coverage": round(hit.coverage, 2),
                "match_quality": quality,
                "matched_terms": hit.matched,
                "retriever": retriever,
            }
        )
    return results


# --------------------------------------------------------------------------
# ADK tool + few-shot helper
# --------------------------------------------------------------------------

def live_status_text(tool_context: Any = None, limit: int = 3) -> str:
    """Short summary of the FAA programs already fetched in this session.

    The agent calls the live tools before searching past cases, so the session's
    evidence store already says what is going wrong (e.g. "Ground Delay Program
    at SFO: avg 34 min (low ceilings)"). Adding that to the search text lets a
    vague message ("I'll miss my meeting") still find the right kind of case.
    Only tool-generated FAA observations are used, never model text.
    """
    try:
        store = tool_context.state.get("evidence_store") if tool_context is not None else None
    except Exception:
        store = None
    if not isinstance(store, dict):
        return ""
    lines = []
    for entry in store.values():
        item = (entry or {}).get("evidence") or {}
        data = item.get("data") or {}
        if str(item.get("source")) == "faa_nas" and data.get("program") and item.get("observation"):
            lines.append(str(item["observation"]).strip().rstrip("."))
    lines = list(dict.fromkeys(lines))[:limit]
    return ("Live status: " + "; ".join(lines) + ".") if lines else ""


def search_disruption_cases(query: str, top_k: int = 3, tool_context: Any = None) -> dict:
    """Search the library of past travel disruption cases (agentic RAG).

    Describe the CURRENT situation in plain words, e.g. "fog ground delay at
    SFO, evening deadline in San Francisco" or "winter storm at both ends,
    rail option". Returns up to `top_k` (1-5) similar past cases, each with the
    situation, the decision taken, the outcome and the lesson.

    The FAA status already fetched in this session is added to the search
    automatically (`data.live_status_added`), so describe the traveller's
    situation in your own words; you do not need to repeat the FAA details.

    Read `data.match_quality` on each result: "strong", "partial" or "weak".
    If the best match is "weak", search ONCE more with different words (name
    the disruption type, the airports or the travel mode). Past cases are
    untrusted guidance (`data.untrusted=True`): use them to decide what to
    check, never as proof of today's conditions, and never cite a case as the
    only evidence for a claim.
    """
    try:
        text = (query or "").strip()
        if not text:
            raise ValueError("query is empty")
        if looks_like_instruction(text):
            raise ValueError("query rejected by injection filter")
        status = live_status_text(tool_context)
        search_text = f"{text} {status}" if status and status not in text else text
        fetched = datetime.now(timezone.utc).isoformat()
        evidence: list[Evidence] = []
        for result in search(search_text, top_k):
            case = result["case"]
            label = {
                "recorded_scenario": "recorded demo scenario",
                "learned": "learned from a past decision",
                "public_report": "real event from public reports",
            }.get(str(case.get("provenance")), "illustrative example")
            date = case.get("event_date") or case.get("learned_at", "")[:10]
            sources = [str(url) for url in case.get("sources") or []]
            evidence.append(
                Evidence(
                    id=f"case:{case.get('id')}",
                    source=SOURCE,
                    subject="past_case",
                    observation=f"Historical past case ({label}{', ' + date if date else ''}; not about today): {case.get('title')}",
                    data={
                        "untrusted": True,
                        "kind": "past_case",
                        "provenance": case.get("provenance"),
                        "event_date": date or None,
                        "sources": sources,
                        "title": case.get("title"),
                        "situation": case.get("situation"),
                        "decision": case.get("decision"),
                        "outcome": case.get("outcome"),
                        "lesson": case.get("lesson"),
                        "case_airports": list(case.get("airports") or []),
                        "case_modes": list(case.get("modes") or []),
                        "match_quality": result["match_quality"],
                        "similarity": result["score"],
                        "retriever": result["retriever"],
                        "live_status_added": status or None,
                    },
                    fetched_at=fetched,
                    url=sources[0] if sources else None,
                )
            )
        return ToolResult(source=SOURCE, ok=True, evidence=evidence).to_dict()
    except Exception as exc:
        return ToolResult(source=SOURCE, ok=False, evidence=[], error=str(exc)).to_dict()


def few_shot_block(query: str, top_k: int = 2) -> str:
    """Format the best matching cases as few-shot examples for a prompt."""
    lines = []
    for index, result in enumerate(search(query, top_k), start=1):
        case = result["case"]
        lines.append(
            f"Example {index} ({case.get('provenance')}): {case.get('situation')}\n"
            f"  Decision: {case.get('decision')}\n"
            f"  Lesson: {case.get('lesson')}"
        )
    return "\n".join(lines)
