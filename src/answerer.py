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
    "Answer the user's question using ONLY the provided SOURCES.\n\n"
    "You MUST follow this exact output format:\n"
    "Answer:\n"
    "1) ...\n"
    "2) ...\n"
    "3) ...\n\n"
    "Message Template:\n"
    "- If the user asks for an email/message/template, provide a ready-to-send template.\n"
    "- Otherwise write: N/A\n\n"
    "Evidence:\n"
    "- \"short quote (<= 20 words)\" — [S#]\n"
    "- \"short quote (<= 20 words)\" — [S#]\n\n"
    "Sources:\n"
    "- [S#]\n"
    "- [S#]\n\n"
    "Rules:\n"
    "- Do NOT paste large doc excerpts. You must synthesize into steps.\n"
    "- Every answer must include at least 2 citations in Evidence and list them again in Sources.\n"
    "- If the SOURCES do not explicitly support the answer, say exactly: \"I don't know based on the provided docs.\"\n"
    "  Then give 2–3 bullet next steps for escalation.\n"
)

# Cost / safety guard: limit how much source text we send to the LLM
MAX_SOURCE_CHARS_PER_CHUNK = 1800
MAX_TOTAL_SOURCE_CHARS = 8000


def _needs_rewrite_to_format(answer: str) -> bool:
    """
    Lightweight validator to reduce 'format' bucket failures.
    We don't need perfection; we just need rubric-friendly structure.
    """
    if not answer or not answer.strip():
        return True

    required_headers = ["Answer:", "Message Template:", "Evidence:", "Sources:"]
    if not all(h in answer for h in required_headers):
        return True

    # Require at least 2 citations like [S1]
    if len(re.findall(r"\[S\d+\]", answer)) < 2:
        return True

    # Discourage giant pasted excerpts: if multiple very long lines exist, likely excerpt dump
    long_lines = [ln for ln in answer.splitlines() if len(ln.strip()) > 220]
    if len(long_lines) >= 2:
        return True

    return False


def _rewrite_to_required_format(client: Any, model: str, temperature: float, max_tokens: int, question: str, sources_block: str, draft: str) -> str:
    """
    One retry: ask the model to reformat + synthesize without changing facts.
    """
    rewrite_prompt = (
        "You produced a draft answer that does not follow the required format.\n\n"
        "Fix it by rewriting into the exact required format.\n"
        "- Keep the content strictly grounded in SOURCES.\n"
        "- Do NOT paste large excerpts; synthesize into steps.\n"
        "- Include Evidence quotes (<= 20 words) with [S#] citations.\n"
        "- Include a Sources section listing at least two [S#].\n"
        "- If unsupported, say: \"I don't know based on the provided docs.\" and add escalation steps.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "SOURCES:\n"
        f"{sources_block}\n\n"
        "DRAFT (to fix):\n"
        f"{draft}\n"
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=max(0.0, min(0.2, temperature)),  # keep rewrite deterministic-ish
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": rewrite_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def _extractive_fallback(question: str, sources: List[Dict[str, Any]]) -> str:
    """
    Docs-only fallback: builds a helpful answer by extracting lines from sources.
    This keeps Day 4 working even without API access.

    Updated for Week 3/Day 20: tries to be more "synthesized steps" instead of raw excerpts.
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

    # If we have sources but cannot call LLM, still try to produce a structured answer.
    parts: List[str] = []
    parts.append("Answer:")
    parts.append("1) Review the relevant KB guidance below and follow the steps.")
    parts.append("2) If you get stuck, share the exact error text/screen and what step you are on.")
    parts.append("3) If this is urgent, escalate with the requested details.\n")

    # Template heuristic: if question seems to ask for a message/email/template
    q = (question or "").lower()
    wants_template = any(k in q for k in ["template", "email", "message", "write me", "draft", "reply"])
    parts.append("Message Template:")
    if wants_template:
        parts.append("Subject: [Add subject here]")
        parts.append("Hi [Name],")
        parts.append("")
        parts.append("I’m reaching out regarding [issue]. Here’s what I’m seeing:")
        parts.append("- [What happened]")
        parts.append("- [Exact error text]")
        parts.append("- [Steps to reproduce]")
        parts.append("")
        parts.append("Could you please advise on the next steps or confirm the correct process per the KB?")
        parts.append("")
        parts.append("Thanks,")
        parts.append("[Your Name]\n")
    else:
        parts.append("N/A\n")

    parts.append("Evidence:")
    citations: List[str] = []

    for i, s in enumerate(sources, start=1):
        lines = best_lines(s.get("text", ""), max_lines=2)
        if not lines:
            continue
        # Use short "quotes" as evidence lines
        quote = lines[0]
        if len(quote) > 100:
            quote = quote[:97] + "..."
        parts.append(f"- \"{quote}\" — [S{i}]")
        citations.append(f"[S{i}]")

        if len(citations) >= 2:
            break

    if not citations:
        return "I don't know based on the provided docs.\n\n- Please provide more context (product/feature and exact error text).\n- If urgent, escalate to support with screenshots and steps to reproduce."

    parts.append("\nSources:")
    for c in citations:
        parts.append(f"- {c}")

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
    max_tokens: int = 450,  # increased so sections don't get truncated
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
        return _extractive_fallback(question, sources), sources

    sources_block = _build_sources_block(sources)

    user_prompt = (
        "QUESTION:\n"
        f"{question}\n\n"
        "SOURCES:\n"
        f"{sources_block}\n\n"
        "TASK:\n"
        "- Use ONLY the SOURCES.\n"
        "- Write a SYNTHESIZED answer (do not paste large excerpts).\n"
        "- If the user is asking for a message/email/template, you MUST provide it in the Message Template section.\n"
        "- Include Evidence quotes (<= 20 words each) with citations like [S1].\n"
        "- Include a Sources section listing at least two [S#].\n"
        "- If not supported by SOURCES, say: \"I don't know based on the provided docs.\" and provide escalation steps.\n"
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

        # One rewrite pass if format is not compliant (reduces 'format' bucket failures)
        if _needs_rewrite_to_format(answer):
            answer = _rewrite_to_required_format(
                client=client,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                question=question,
                sources_block=sources_block,
                draft=answer,
            )

        # If still broken, fail safe to IDK (better than bad format for rubric)
        if _needs_rewrite_to_format(answer):
            answer = (
                "I don't know based on the provided docs.\n\n"
                "**Next steps:**\n"
                "- Tell me the exact product/feature name and the exact error text.\n"
                "- Share what you tried and where you got stuck.\n"
                "- If urgent, escalate to support with screenshots + steps to reproduce.\n"
            )

        return answer, sources

    except RateLimitError:
        msg = (
            "LLM call blocked (quota/rate limit). Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(question, sources)
        )
        return msg, sources

    except APIStatusError:
        msg = (
            "LLM call failed with an API status error. Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(question, sources)
        )
        return msg, sources

    except Exception as e:
        msg = (
            f"LLM call failed ({type(e).__name__}). Using docs-only extractive answer instead.\n\n"
            + _extractive_fallback(question, sources)
        )
        return msg, sources
