# src/guardrails.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re


# ✅ BM25 default threshold tuned to YOUR observed score range:
# - Your "good" normal cases often have best_score around 1.1–2.5+
# - Setting this too high (like 2.5) blocks legit normal questions.
DEFAULT_BM25_THRESHOLD = 1.0


@dataclass
class GuardrailDecision:
    can_answer: bool
    reason: str
    best_score: float
    threshold: float
    missing_terms: List[str]


STOPWORDS = {
    "a","an","the","and","or","but","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being",
    "i","me","my","you","your","we","our","they","their",
    "how","what","why","when","where","can","could","should","would","do","does","did",
    "please","help",

    # extra conversational filler (these were showing up as “missing_terms”)
    "this","that","these","those","just","anything","everything","into","about",
    "tell","give","exact","exactly","need","asap","month","right","left","side",
}

# Generic support words that match lots of docs and cause false positives
GENERIC_TERMS = {
    "reset","issue","problem","error","fail","failed","fix","troubleshoot","troubleshooting",
    "account","login","access","setup","configure","request",
}

_REFUSE_PATTERNS = [
    r"\bbypass\b",
    r"\bcircumvent\b",
    r"\bdisable\b.*\bmfa\b",
    r"\bturn\s*off\b.*\bmfa\b",
    r"\bwithout verification\b",
    r"\badmin password\b",
    r"\bvpn server password\b",
    r"\binternal vpn\b.*\b(ip|address)\b",
    r"\bserver ip\b",
    r"\bnetwork architecture\b",
    r"\barchitecture\b.*\bdiagram\b",
    r"\bcredentials?\b",
    r"\bmaster password\b",
    r"\bbackdoor\b",
]


def should_refuse(question: str) -> Tuple[bool, str]:
    q = (question or "").lower().strip()
    if not q:
        return False, ""

    for pat in _REFUSE_PATTERNS:
        if re.search(pat, q):
            return True, "Sensitive/out-of-scope request (security/credential/bypass)."

    return False, ""


def _extract_key_terms(question: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", (question or "").lower())
    terms: List[str] = []
    for w in words:
        if len(w) < 4:
            continue
        if w in STOPWORDS:
            continue
        if w in GENERIC_TERMS:
            continue
        terms.append(w)

    # de-dupe while preserving order
    seen = set()
    unique: List[str] = []
    for t in terms:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def _term_variants(t: str) -> List[str]:
    """
    Small normalization to reduce false negatives:
    - handle simple plural forms: sites -> site
    """
    variants = [t]
    if t.endswith("s") and len(t) > 4:
        variants.append(t[:-1])
    return list(dict.fromkeys(variants))


def decide_if_can_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    score_threshold: float = DEFAULT_BM25_THRESHOLD,
    min_chunks: int = 1,
    check_top_n_chunks_text: int = 3,
) -> GuardrailDecision:
    # 1) Need enough retrieved chunks
    if not retrieved_chunks or len(retrieved_chunks) < min_chunks:
        return GuardrailDecision(
            can_answer=False,
            reason="No relevant documentation chunks were retrieved.",
            best_score=0.0,
            threshold=float(score_threshold),
            missing_terms=[],
        )

    # 2) Need best score >= threshold
    best_score = float(retrieved_chunks[0].get("score") or 0.0)
    if best_score < float(score_threshold):
        return GuardrailDecision(
            can_answer=False,
            reason="Top retrieved chunk score is below threshold.",
            best_score=best_score,
            threshold=float(score_threshold),
            missing_terms=[],
        )

    # 3) Keyword coverage check
    key_terms = _extract_key_terms(question)

    combined_text = " ".join(
        (retrieved_chunks[i].get("text") or "").lower()
        for i in range(min(check_top_n_chunks_text, len(retrieved_chunks)))
    )

    present: List[str] = []
    missing: List[str] = []

    for t in key_terms:
        variants = _term_variants(t)
        if any(v in combined_text for v in variants):
            present.append(t)
        else:
            missing.append(t)

    # If question has specific key terms and NONE are present, block.
    if key_terms and len(present) == 0:
        return GuardrailDecision(
            can_answer=False,
            reason="Question-specific keywords were not found in the retrieved docs.",
            best_score=best_score,
            threshold=float(score_threshold),
            missing_terms=missing,
        )

    # If multiple key terms and most missing, block (conservative)
    if len(key_terms) >= 3:
        missing_ratio = len(missing) / max(len(key_terms), 1)
        if len(missing) >= 2 and missing_ratio >= 0.80:
            return GuardrailDecision(
                can_answer=False,
                reason="Most question-specific keywords were not found in the retrieved docs.",
                best_score=best_score,
                threshold=float(score_threshold),
                missing_terms=missing,
            )

    return GuardrailDecision(
        can_answer=True,
        reason="Sufficient doc support detected (score + keyword coverage).",
        best_score=best_score,
        threshold=float(score_threshold),
        missing_terms=missing,
    )


def idk_response(user_question: str) -> str:
    return (
        "I don't know based on the documentation I have loaded.\n\n"
        "**Next steps (so we can resolve this):**\n"
        "- If this is a product question, tell me the product/feature name and the exact screen or error text.\n"
        "- Share: environment (prod/dev), what you expected, what happened, and any steps to reproduce.\n"
        "- If this is urgent/blocking, escalate to support with:\n"
        "  - exact error message or screenshot\n"
        "  - steps to reproduce\n"
        "  - timestamps + timezone\n"
        "  - user/account/org (if relevant)\n\n"
        "**Quick question:** what product/app is this about (and what exact error text do you see)?"
    )
