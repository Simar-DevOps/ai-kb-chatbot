from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Tuple

# Optional: only used if OPENAI_API_KEY is set
try:
    from openai import OpenAI, RateLimitError, APIStatusError
except Exception:
    OpenAI = None
    RateLimitError = Exception
    APIStatusError = Exception


SYSTEM_PROMPT = (
    "You are a support knowledge base assistant.\n"
    "You must answer using ONLY the provided SOURCES.\n\n"
    "Critical rules:\n"
    '- If the answer is not explicitly supported by the SOURCES, say exactly: "I don\'t know based on the provided docs."\n'
    "- Do NOT invent policies, SLAs, internal IPs, passwords, or secret details.\n"
    "- Do NOT paste long doc excerpts. Synthesize into short steps or a filled template.\n"
    "- Cite sources inline like [S1], [S2] only when the claim comes from that source.\n"
    "- If the user asks for a template/message, output the actual template/message (not instructions).\n"
)

# Cost / safety guard: limit how much source text we send to the LLM
MAX_SOURCE_CHARS_PER_CHUNK = 1800
MAX_TOTAL_SOURCE_CHARS = 8000


# --- Simple keyword logic so extractive fallback doesn't "answer from docs" when the question is vague ---
_STOPWORDS = {
    "a","an","the","and","or","but","to","of","in","on","for","with","at","by","from",
    "is","are","was","were","be","been","being",
    "i","me","my","you","your","we","our","they","their",
    "how","what","why","when","where","can","could","should","would","do","does","did",
    "please","help",
    "this","that","these","those","just","anything","everything","into","about",
    "tell","give","exact","exactly","need","asap","month","right","left","side",
}

_GENERIC_TERMS = {
    "reset","issue","problem","error","fail","failed","fix","troubleshoot","troubleshooting",
    "account","login","access","setup","configure","request",
}

def _extract_key_terms(question: str) -> List[str]:
    words = re.findall(r"[a-z0-9]+", (question or "").lower())
    terms: List[str] = []
    for w in words:
        if len(w) < 4:
            continue
        if w in _STOPWORDS:
            continue
        if w in _GENERIC_TERMS:
            continue
        terms.append(w)

    # de-dupe, keep order
    seen = set()
    out: List[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _extractive_fallback(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Docs-only fallback: builds a helpful answer by extracting lines from sources.
    BUT: if question is too vague or keywords don't appear, return an IDK response instead.
    """

    if not sources:
        return "I don't know based on the provided docs."

    key_terms = _extract_key_terms(question)

    combined_text = " ".join(((s.get("text") or "").lower()) for s in sources[:3])

    # If question has no meaningful keywords, it's too vague to answer safely
    if not key_terms:
        return "I don't know based on the provided docs."

    # If none of the key terms show up in the top sources, don't pretend to answer
    if all(t not in combined_text for t in key_terms):
        return "I don't know based on the provided docs."

    def best_lines(text: str, max_lines: int = 5) -> List[str]:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith(("-", "*", "•"))]
        chosen = bullets[:max_lines]

        if len(chosen) < max_lines:
            for ln in lines:
                if ln not in chosen:
                    chosen.append(ln)
                if len(chosen) == max_lines:
                    break
        return chosen[:max_lines]

    parts: List[str] = ["**Answer (from docs):**"]
    for i, s in enumerate(sources, start=1):
        lines = best_lines(s.get("text", ""), max_lines=3)
        if not lines:
            continue
        parts.append(f"\n**From [S{i}] {s.get('source')} (chunk {s.get('chunk_id')}):**")
        for ln in lines:
            parts.append(f"- {ln}")

    if len(parts) == 1:
        return "I don't know based on the provided docs."

    parts.append("\nIf this doesn’t fully answer your question, ask a more specific question and I’ll pull more relevant sections.")
    return "\n".join(parts)


def _build_sources_block(sources: List[Dict[str, Any]]) -> str:
    """
    Build a bounded SOURCES block for the model.
    """
    lines: List[str] = []
    total = 0

    for i, s in enumerate(sources, start=1):
        header = f"[S{i}] file={s.get('source')} chunk={s.get('chunk_id')} score={float(s.get('score', 0.0)):.4f}"
        body = ((s.get("text", "") or "").strip())[:MAX_SOURCE_CHARS_PER_CHUNK]

        block = header + "\n" + body + "\n\n"
        if total + len(block) > MAX_TOTAL_SOURCE_CHARS:
            break

        lines.append(header)
        lines.append(body)
        lines.append("")  # spacer
        total += len(block)

    return "\n".join(lines).strip()


def _task_hint(question: str) -> str:
    q = (question or "").lower()
    if any(k in q for k in ["incident update", "incident comm", "stakeholder", "sev", "status:", "impact:", "next update"]):
        return "INCIDENT_UPDATE"
    if any(k in q for k in ["template", "message template", "draft", "write a short", "email template", "respond and guide"]):
        return "TEMPLATE_OR_MESSAGE"
    if any(k in q for k in ["summarize", "summary", "bullet", "5 bullet", "steps"]):
        return "SUMMARY"
    return "QA"


def answer_with_sources(
    question: str,
    sources: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 350,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns: (answer_text, sources)
    """
    if not sources:
        return "I don't know based on the provided docs.", sources

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    can_call_llm = bool(api_key) and (OpenAI is not None)

    if not can_call_llm:
        return _extractive_fallback(question, sources), sources

    sources_block = _build_sources_block(sources)
    task = _task_hint(question)

    user_prompt = (
        f"TASK_TYPE: {task}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "SOURCES:\n"
        f"{sources_block}\n\n"
        "OUTPUT REQUIREMENTS:\n"
        "- If TASK_TYPE is INCIDENT_UPDATE: output a stakeholder-ready update message using the doc template fields.\n"
        "  Use the given Status/Impact/Next Update values from the question. Keep it short.\n"
        "- If TASK_TYPE is TEMPLATE_OR_MESSAGE: output a usable template/message with placeholders or filled fields.\n"
        "- If TASK_TYPE is QA or SUMMARY: output actionable steps/bullets, synthesized (not pasted excerpts).\n"
        "- Always include citations inline like [S1].\n"
        "- Do not include a 'Sources:' section. Do not list [S1] by itself.\n"
        "- Include an 'Evidence:' section with 1–3 short quoted snippets (<= 15 words each) that support key claims, each with citation.\n"
        "- If SOURCES do not explicitly support the answer, reply exactly: \"I don't know based on the provided docs.\"\n"
    )

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = resp.choices[0].message.content.strip()
        return answer, sources

    except RateLimitError:
        msg = "LLM call blocked (quota/rate limit). Using docs-only extractive answer instead.\n\n" + _extractive_fallback(question, sources)
        return msg, sources

    except APIStatusError:
        msg = "LLM call failed with an API status error. Using docs-only extractive answer instead.\n\n" + _extractive_fallback(question, sources)
        return msg, sources

    except Exception as e:
        msg = f"LLM call failed ({type(e).__name__}). Using docs-only extractive answer instead.\n\n" + _extractive_fallback(question, sources)
        return msg, sources
