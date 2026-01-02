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


def _extract_key_terms(question: str) -> List[str]:
    """
    Pull “meaningful” terms from the question:
    - letters/numbers only
    - length >= 4 (filters out tiny words)
    - remove stopwords + generic terms
    """
    words = re.findall(r"[a-z0-9]+", question.lower())
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
) -> GuardrailDecision:
    """
    Hard guardrails:
    1) Need enough retrieved chunks
    2) Need best score >= threshold
    3) Need the question’s key terms to actually appear in the retrieved text
       (prevents “reset my iphone” → password reset chunks)
    """

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

    # 3) Keyword coverage check
    key_terms = _extract_key_terms(question)

    combined_text = " ".join(
        (retrieved_chunks[i].get("text") or "").lower()
        for i in range(min(check_top_n_chunks_text, len(retrieved_chunks)))
    )

    missing = [t for t in key_terms if t not in combined_text]

    # If question has specific key terms and NONE are present, refuse.
    if key_terms and len(missing) == len(key_terms):
        return GuardrailDecision(
            can_answer=False,
            reason="Question-specific keywords were not found in the retrieved docs.",
            best_score=best_score,
            threshold=score_threshold,
            missing_terms=missing,
        )

    # Conservative extra safety:
    # If the question has multiple key terms and MOST are missing, also refuse.
    # This catches cases like: "reset iphone icloud" where docs match only "reset"
    if len(key_terms) >= 2:
        missing_ratio = len(missing) / max(len(key_terms), 1)
        if len(missing) >= 2 and missing_ratio >= 0.80:
            return GuardrailDecision(
                can_answer=False,
                reason="Most question-specific keywords were not found in the retrieved docs.",
                best_score=best_score,
                threshold=score_threshold,
                missing_terms=missing,
            )

    return GuardrailDecision(
        can_answer=True,
        reason="Sufficient doc support detected (score + keyword coverage).",
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
