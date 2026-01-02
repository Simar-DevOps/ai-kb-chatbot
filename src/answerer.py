from __future__ import annotations

import os
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
    "Answer the user's question using ONLY the provided SOURCES.\n\n"
    "Rules:\n"
    '- If the answer is not explicitly supported by the SOURCES, say: "I don\'t know based on the provided docs."\n'
    "- When you use a source, cite it inline like [S1], [S2], etc.\n"
    "- Be concise and practical (steps, bullets).\n"
)

# Cost / safety guard: limit how much source text we send to the LLM
MAX_SOURCE_CHARS_PER_CHUNK = 1800
MAX_TOTAL_SOURCE_CHARS = 8000


def _extractive_fallback(sources: List[Dict[str, Any]]) -> str:
    """
    Docs-only fallback: builds a helpful answer by extracting lines from sources.
    This keeps Day 4 working even without API access.
    """

    def best_lines(text: str, max_lines: int = 4) -> List[str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = [ln for ln in lines if ln.startswith(("-", "*", "•"))]
        chosen = bullets[:max_lines]

        if len(chosen) < max_lines:
            for ln in lines:
                if ln not in chosen:
                    chosen.append(ln)
                if len(chosen) == max_lines:
                    break
        return chosen[:max_lines]

    parts: List[str] = []
    parts.append("**Answer (from docs):**")

    for i, s in enumerate(sources, start=1):
        lines = best_lines(s.get("text", ""), max_lines=3)
        if not lines:
            continue

        parts.append(f"\n**From [S{i}] {s.get('source')} (chunk {s.get('chunk_id')}):**")
        for ln in lines:
            parts.append(f"- {ln}")

    if len(parts) == 1:
        return "I don't know based on the provided docs."

    parts.append(
        "\nIf this doesn’t fully answer your question, ask a more specific question and I’ll pull more relevant sections."
    )
    return "\n".join(parts)


def _build_sources_block(sources: List[Dict[str, Any]]) -> str:
    """
    Build a bounded SOURCES block for the model.
    Prevents huge prompts and keeps costs predictable.
    """
    lines: List[str] = []
    total = 0

    for i, s in enumerate(sources, start=1):
        header = f"[S{i}] file={s.get('source')} chunk={s.get('chunk_id')} score={float(s.get('score', 0.0)):.4f}"
        body = ((s.get("text", "") or "").strip())[:MAX_SOURCE_CHARS_PER_CHUNK]

        block = header + "\n" + body + "\n"
        if total + len(block) > MAX_TOTAL_SOURCE_CHARS:
            break

        lines.append(header)
        lines.append(body)
        lines.append("")  # spacer
        total += len(block)

    return "\n".join(lines).strip()


def answer_with_sources(
    question: str,
    sources: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 250,  # cost control
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns:
      (answer_text, sources)

    sources are returned unchanged so the UI can display them.
    """
    if not sources:
        return "I don't know based on the provided docs.", sources

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    can_call_llm = bool(api_key) and (OpenAI is not None)

    # No key (or no SDK) → docs-only extractive answer
    if not can_call_llm:
        return _extractive_fallback(sources), sources

    sources_block = _build_sources_block(sources)

    user_prompt = (
        "QUESTION:\n"
        f"{question}\n\n"
        "SOURCES:\n"
        f"{sources_block}\n\n"
        "Now write the answer using ONLY the SOURCES. Remember to cite like [S1], [S2].\n"
        'If the SOURCES do not explicitly support the answer, say: "I don\'t know based on the provided docs."'
    )

    try:
        # OpenAI client will read OPENAI_API_KEY from environment by default.
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
        msg = (
            "LLM call blocked (quota/rate limit). Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(sources)
        )
        return msg, sources

    except APIStatusError:
        msg = (
            "LLM call failed with an API status error. Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(sources)
        )
        return msg, sources

    except Exception as e:
        msg = (
            f"LLM call failed ({type(e).__name__}). Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(sources)
        )
        return msg, sources
