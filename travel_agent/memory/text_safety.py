"""Extra check for user text that M4 stores or forwards to the model.

M1's ``detect_injection_rules`` is the main filter. Red-team tests found two
gaps for M4's paths (change requests, rejection reasons, memory, case search):

* full-width / compatibility characters ("ｉｇｎｏｒｅ previous instructions")
  are not normalised first;
* the same attack written in another common language is not recognised.

``looks_like_instruction`` closes both for M4's own inputs: it NFKC-normalises
the text, runs M1's filter on it, and adds a short multilingual phrase list.
The gap is also reported to M1 so the chat input guard can be updated.
"""

from __future__ import annotations

import re
import unicodedata

from travel_agent.guardrails.injection import detect_injection_rules

# "ignore / forget / disregard (the) previous|above instructions" and
# "approval not needed" in Spanish, French, German, Portuguese and Italian.
_MULTILINGUAL = re.compile(
    r"(?:ignora|olvida|ignorez|oubliez|ignoriere|vergiss|ignore|esque[cç]a|dimentica)\w*\s+"
    r"(?:\w+\s+){0,3}(?:instrucciones|instructions|anweisungen|instru[cç][oõ]es|istruzioni|reglas|r[eè]gles|regeln|regras|regole)"
    r"|sin\s+(?:pedir\s+)?(?:aprobaci[oó]n|permiso|preguntar)"
    r"|sans\s+(?:demander|approbation|autorisation)"
    r"|ohne\s+(?:zu\s+fragen|genehmigung|best[aä]tigung)"
    r"|sem\s+(?:perguntar|aprova[cç][aã]o)"
    r"|senza\s+(?:chiedere|approvazione)"
    r"|(?:approval|confirmation)\s+(?:is\s+)?(?:not\s+(?:required|needed)|unnecessary)",
    re.IGNORECASE,
)


def normalise(text: str) -> str:
    """NFKC-normalise (full-width -> ASCII etc.) and drop zero-width characters."""
    folded = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[​-‏⁠﻿]", "", folded)


def looks_like_instruction(text: str) -> bool:
    """True if ``text`` reads like an instruction to the AI rather than travel content."""
    if not text or not text.strip():
        return False
    clean = normalise(text)
    if detect_injection_rules(text)["flagged"] or detect_injection_rules(clean)["flagged"]:
        return True
    return bool(_MULTILINGUAL.search(clean))
