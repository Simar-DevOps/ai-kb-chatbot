# src/guardrails.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re


@dataclass
class GuardrailDecision:
    can_answer: bool
    reason: str
    best_score: float
    threshold: float
    missing_terms: List[str]


# Keep this small + practical
STOPWORDS = {
    "a","an","the","and","or","but","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being",
    "i","me","my","you","your","we","our","they","their",
    "how","what","why","when","where","can","could","should","would","do","does","did",
    "please","help",
}

# These are “generic support words” that match lots of docs and cause false positives
GENERIC_TERMS = {
    "reset","issue","problem","error","fail","failed","fix","troubleshoot","troubleshooting",
    "account","login","password","access","setup","configure","request",
}

# Minimal "should refuse" gate for sensitive/out-of-scope prompts (matches your I0x bucket intent)
# Keep this list tight—only add what your eval set expects to be refused.
SENSITIVE_PATTERNS = [
    r"\bphish(ing)?\b",
    r"\bmalware\b",
    r"\bkeylogger\b",
    r"\bransomware\b",
    r"\bddos\b",
    r"\bexploit\b",
    r"\bbypass\b",
    r"\bhack(ing)?\b",
    r"\bsteal\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\bcounterfeit\b",
    r"\bbomb\b",
    r"\bweapon\b",
    r"\bssn\b",
    r"\bsocial security\b",
    r"\bcredit card\b",
    r"\bbank account\b",
]


def should_refuse(question: str) -> bool:
    """
    Returns True if the user's request is sensitive/out-of-scope and should be refused
    (based on your eval set's "I0x should refuse" cases).
    """
    q = (question or "").lower()
    return any(re.search(pat, q) for pat in SENSITIVE_PATTERNS)


def _extract_key_terms(question: str) -> List[str]:
    """
    Pull “meaningful” terms from the question:
    - letters/numbers only
    - length >= 4 (filters out tiny words)
    - remove stopwords + generic terms
    """
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
    unique = []
    for t in terms:
        if t not in seen:
            unique.append(t)
            seen.add(t)
    return unique


def decide_if_can_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    score_threshold: float = 0.20,
    min_chunks: int = 1,
    check_top_n_chunks_text: int = 3,
    high_score_override: float = 0.35,
) -> GuardrailDecision:
    """
    Guardrail policy (Day 20 tuned):
    0) Refuse if sensitive/out-of-scope (I0x cases)
    1) Need enough retrieved chunks
    2) Need best score >= threshold
    3) Keyword coverage check (but less aggressive to avoid false blocks)
       - If top score is very high, allow even if keyword coverage is imperfect.
       - Only block on keyword coverage when coverage is clearly absent AND score isn't strong.
    """

    # 0) Sensitive/out-of-scope refusal
    if should_refuse(question):
        return GuardrailDecision(
            can_answer=False,
            reason="Sensitive/out-of-scope request (refuse).",
            best_score=0.0,
            threshold=score_threshold,
            missing_terms=[],
        )

    # 1) Need enough retrieved chunks
    if not retrieved_chunks or len(retrieved_chunks) < min_chunks:
        return GuardrailDecision(
            can_answer=False,
            reason="No relevant documentation chunks were retrieved.",
            best_score=0.0,
            threshold=score_threshold,
            missing_terms=[],
        )

    # 2) Need best score >= threshold
    best_score = float(retrieved_chunks[0].get("score") or 0.0)
    if best_score < score_threshold:
        return GuardrailDecision(
            can_answer=False,
            reason="Top retrieved chunk score is below threshold.",
            best_score=best_score,
            threshold=score_threshold,
            missing_terms=[],
        )

    # If retriever is very confident, do not over-block on keyword heuristics.
    if best_score >= high_score_override:
        return GuardrailDecision(
            can_answer=True,
            reason="High retriever confidence (score override).",
            best_score=best_score,
            threshold=score_threshold,
            missing_terms=[],
        )

    # 3) Keyword coverage check (less aggressive than before)
    key_terms = _extract_key_terms(question)

    combined_text = " ".join(
        (retrieved_chunks[i].get("text") or "").lower()
        for i in range(min(check_top_n_chunks_text, len(retrieved_chunks)))
    )

    missing = [t for t in key_terms if t not in combined_text]

    # If there are NO key terms (question is generic), allow (avoid false blocks).
    if not key_terms:
        return GuardrailDecision(
            can_answer=True,
            reason="No specific key terms found; allowing answer with retrieved docs.",
            best_score=best_score,
            threshold=score_threshold,
            missing_terms=[],
        )

    # If at least one key term appears, allow.
    if len(missing) < len(key_terms):
        return GuardrailDecision(
            can_answer=True,
            reason="Some question-specific keywords found in retrieved docs.",
            best_score=best_score,
            threshold=score_threshold,
            missing_terms=missing,
        )

    # If none appear AND score is only moderate, block (true mismatch).
    return GuardrailDecision(
        can_answer=False,
        reason="Question-specific keywords were not found in the retrieved docs.",
        best_score=best_score,
        threshold=score_threshold,
        missing_terms=missing,
    )


def idk_response(user_question: str) -> str:
    # Streamlit Markdown renders bullet lists cleanly; keep it action-oriented.
    return (
        "I don’t know based on the documentation I have loaded.\n\n"
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
