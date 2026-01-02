from __future__ import annotations

import sys
import os
import json
import uuid
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import streamlit as st
import pandas as pd  # Day 10: analytics

# Load environment variables from .env (local only; should be gitignored)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, env vars can still come from the OS.
    pass

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()


# =========================
# Fix imports when app is inside /app
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import build_kb_from_data_dir, BM25Retriever
from src.answerer import answer_with_sources
from src.guardrails import decide_if_can_answer, idk_response
from src.feedback_store import append_feedback_csv, append_feedback_jsonl

# Day 9: persisted admin settings
from src.admin_settings import load_settings, save_settings, reset_settings, AdminSettings


# =========================
# Config knobs
# =========================
APP_TITLE = "Support KB Chatbot"

# Day 10: bump version
APP_VERSION = "week2-day10"

# Day 8: feedback logging
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"
FEEDBACK_CSV_PATH = FEEDBACK_DIR / "feedback.csv"
FEEDBACK_JSONL_PATH = FEEDBACK_DIR / "feedback.jsonl"

# Day 10: questions logging (you already created /logs)
LOGS_DIR = PROJECT_ROOT / "logs"
QUESTIONS_CSV_PATH = LOGS_DIR / "questions.csv"

# IMPORTANT: your docs live under data/raw (per your directory listing)
DOCS_DIR = PROJECT_ROOT / "data" / "raw"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Day 6 guardrail defaults (you can tune later)
IDK_SCORE_THRESHOLD_DEFAULT = 0.20  # if best chunk score < this => IDK

EXAMPLE_QUESTIONS = [
    "How do I reset my password?",
    "My account is locked — what should I do?",
    "Why am I seeing a 500 error?",
    "How do I request a refund?",
    "Where can I find my invoices?",
]


# =========================
# Page setup
# =========================
st.set_page_config(page_title=APP_TITLE, page_icon="💬", layout="wide")
st.title(f"💬 {APP_TITLE}")
st.caption("Ask questions about the support docs. Answers include sources (chunks).")


# =========================
# Build / cache retriever
# =========================
@st.cache_resource
def get_retriever(data_dir: Path, chunk_size: int, overlap: int) -> BM25Retriever:
    """
    Why: Streamlit reruns on every interaction; caching prevents rebuild every time.
    """
    return build_kb_from_data_dir(
        data_dir=data_dir,
        k_chunk_size=chunk_size,
        k_overlap=overlap,
    )


# =========================
# Day 10: Analytics helpers
# =========================
def ensure_dir(p: Path) -> None:
    # Why: prevent "folder not found" crashes when writing logs
    p.mkdir(parents=True, exist_ok=True)


def append_question_csv(path: Path, row: Dict[str, Any]) -> None:
    """
    Write one question row to logs/questions.csv (create file + header if missing).
    Why (kid version): This is our "diary" of what users asked.
    """
    ensure_dir(path.parent)

    fieldnames = [
        "ts_utc",
        "session_id",
        "question_id",
        "user_question",
        "model",
        "use_llm",
        "top_k",
        "temperature",
        "max_tokens",
        "idk_threshold",
        "can_answer",
        "best_score",
        "retrieved_chunks_count",
        "sources_shown_count",
        "app_version",
    ]

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        cleaned = {k: row.get(k, "") for k in fieldnames}
        w.writerow(cleaned)


def normalize_query(text: str) -> str:
    """
    Make similar questions count as the same:
    'Reset password?' and 'reset password' -> same bucket.
    """
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)  # remove punctuation
    return t


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # If file is mid-write or weird encoding, don’t crash the app.
        return pd.DataFrame()


# =========================
# Session state
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "pending_user_message" not in st.session_state:
    st.session_state.pending_user_message = None

# Day 9: persisted admin settings (disk -> session_state once)
if "admin_settings" not in st.session_state:
    st.session_state.admin_settings = load_settings()


# =========================
# Sidebar UX
# =========================
with st.sidebar:
    st.subheader("Quick actions")

    if st.button("🧹 Reset chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_user_message = None
        st.session_state.session_id = str(uuid.uuid4())  # new session for new chat
        st.rerun()

    st.divider()

    st.subheader("Example questions")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending_user_message = q
            st.rerun()

    st.divider()

    # =========================
    # Day 9: Admin Controls (persisted)
    # =========================
    st.subheader("Admin controls (Day 9)")

    s: AdminSettings = st.session_state.admin_settings

    MODEL_OPTIONS = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
    ]

    # Widgets default to saved values
    new_model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(s.model) if s.model in MODEL_OPTIONS else 0,
    )
    new_top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=int(s.top_k), step=1)
    new_temperature = st.slider("Answer temperature", min_value=0.0, max_value=1.0, value=float(s.temperature), step=0.05)
    new_max_tokens = st.slider("Max answer tokens", min_value=100, max_value=2000, value=int(s.max_tokens), step=50)
    new_use_llm = st.toggle("LLM Answering (ON/OFF)", value=bool(s.use_llm))

    c1, c2 = st.columns(2)
    with c1:
        save_now = st.button("💾 Save settings", use_container_width=True)
    with c2:
        do_reset = st.button("↩️ Reset settings", use_container_width=True)

    if do_reset:
        st.session_state.admin_settings = reset_settings()
        st.toast("Admin settings reset to defaults", icon="✅")
        st.rerun()

    changed = (
        new_model != s.model
        or int(new_top_k) != int(s.top_k)
        or float(new_temperature) != float(s.temperature)
        or int(new_max_tokens) != int(s.max_tokens)
        or bool(new_use_llm) != bool(s.use_llm)
    )

    if save_now or changed:
        st.session_state.admin_settings = AdminSettings(
            model=new_model,
            top_k=int(new_top_k),
            temperature=float(new_temperature),
            max_tokens=int(new_max_tokens),
            use_llm=bool(new_use_llm),
        )
        save_settings(st.session_state.admin_settings)

    # Make these available to the rest of the app
    s = st.session_state.admin_settings
    top_k = int(s.top_k)
    temperature = float(s.temperature)
    max_tokens = int(s.max_tokens)
    model = str(s.model)

    # IMPORTANT: If the OpenAI key isn't set, force retrieval-only mode safely.
    requested_use_llm = bool(s.use_llm)
    if requested_use_llm and not OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set. LLM mode is disabled. Add it to a local .env file.", icon="🔒")
    use_llm = bool(requested_use_llm and OPENAI_API_KEY)

    st.divider()

    # =========================
    # Guardrails controls (Day 6)
    # =========================
    st.subheader("Guardrails (Day 6)")

    idk_threshold = st.slider(
        "IDK threshold (min top score to answer)",
        min_value=0.0,
        max_value=1.0,
        value=IDK_SCORE_THRESHOLD_DEFAULT,
        step=0.05,
    )
    show_guardrail_debug = st.checkbox("Show guardrail debug", value=False)

    st.divider()
    st.subheader("Docs status")

    if not DOCS_DIR.exists():
        st.error(f"Docs folder not found: {DOCS_DIR}")
    else:
        md_count = len(list(DOCS_DIR.glob("*.md")))
        txt_count = len(list(DOCS_DIR.glob("*.txt")))
        if md_count == 0 and txt_count == 0:
            st.error(f"No .md/.txt docs found in: {DOCS_DIR}")
        else:
            st.success(f"Using docs folder: {DOCS_DIR}")
            st.caption(f".md: {md_count} | .txt: {txt_count}")

    st.caption(f"LLM: {'ON' if use_llm else 'OFF'} | model: {model}")
    if not OPENAI_API_KEY:
        st.caption("To enable LLM: create a local .env with OPENAI_API_KEY=... (do not commit it).")

    st.divider()

    # =========================
    # Day 10: Analytics toggle
    # =========================
    show_analytics = st.toggle("📊 Show Analytics (Day 10)", value=False)


# =========================
# Helpers
# =========================
def render_sources(sources: List[Dict[str, Any]]) -> None:
    if not sources:
        return

    sources_to_show = sources[:3]

    with st.expander("Sources", expanded=False):
        for i, s in enumerate(sources_to_show, start=1):
            file_name = s.get("source", "unknown_file")
            chunk_id = s.get("chunk_id", "unknown_chunk")
            score = s.get("score", None)

            header = f"**{i}. {file_name} — chunk {chunk_id}**"
            if score is not None:
                header += f" (score: {float(score):.4f})"
            st.markdown(header)

            text = (s.get("text") or "").strip()
            if text:
                st.code(text, language="markdown")


def render_guardrail_debug(guardrail: Optional[Dict[str, Any]]) -> None:
    if not guardrail:
        return
    with st.expander("Guardrail debug", expanded=False):
        st.write(f"can_answer: {guardrail.get('can_answer')}")
        st.write(f"reason: {guardrail.get('reason')}")
        st.write(f"best_score: {guardrail.get('best_score')}")
        st.write(f"threshold: {guardrail.get('threshold')}")
        if "missing_terms" in guardrail:
            st.write(f"missing_terms: {guardrail.get('missing_terms')}")


# ---- Day 8: feedback save + buttons ----
def save_feedback(rating: str, msg: Dict[str, Any]) -> None:
    """
    rating: "up" or "down"
    msg: assistant message dict from st.session_state.messages
    """
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": st.session_state.session_id,
        "message_id": msg.get("message_id", ""),
        "rating": rating,
        "user_question": msg.get("question", ""),
        "assistant_answer": msg.get("content", "")[:2000],
        "sources": [
            {"source": s.get("source"), "chunk_id": s.get("chunk_id"), "score": s.get("score")}
            for s in (msg.get("sources", []) or [])
        ],
        "guardrail": msg.get("guardrail", {}),
        # Day 9: include admin settings for analytics later
        "admin": {
            "model": msg.get("model", ""),
            "use_llm": msg.get("use_llm", ""),
            "top_k": msg.get("top_k", ""),
            "temperature": msg.get("temperature", ""),
            "max_tokens": msg.get("max_tokens", ""),
        },
        "idk_threshold": msg.get("idk_threshold", ""),
        "app_version": msg.get("app_version", APP_VERSION),
    }

    row = {
        "timestamp_utc": payload["timestamp_utc"],
        "session_id": payload["session_id"],
        "message_id": payload["message_id"],
        "rating": payload["rating"],
        "user_question": payload["user_question"],
        "assistant_answer": payload["assistant_answer"],
        "sources_json": json.dumps(payload["sources"], ensure_ascii=False),
        "top_k": payload["admin"]["top_k"],
        "temperature": payload["admin"]["temperature"],
        "max_tokens": payload["admin"]["max_tokens"],
        "use_llm": payload["admin"]["use_llm"],
        "model": payload["admin"]["model"],
        "app_version": payload["app_version"],
    }

    append_feedback_csv(FEEDBACK_CSV_PATH, row)
    append_feedback_jsonl(FEEDBACK_JSONL_PATH, payload)


def render_feedback_buttons(msg: Dict[str, Any]) -> None:
    mid = msg.get("message_id")
    if not mid:
        return

    c1, c2 = st.columns(2)
    if c1.button("👍 Helpful", key=f"fb_up_{mid}"):
        save_feedback("up", msg)
        st.toast("Saved 👍 feedback", icon="✅")

    if c2.button("👎 Not helpful", key=f"fb_down_{mid}"):
        save_feedback("down", msg)
        st.toast("Saved 👎 feedback", icon="📝")


def run_pipeline(question: str) -> Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """
    Returns:
      answer_text
      sources_to_show (what the UI will show)
      guardrail info (for debug)
      retrieved_chunks_count (for analytics even if we show no sources)
    """
    # Guard so the app doesn't crash if docs are missing
    if (not DOCS_DIR.exists()) or (
        len(list(DOCS_DIR.glob("*.md"))) == 0 and len(list(DOCS_DIR.glob("*.txt"))) == 0
    ):
        msg = (
            "I don’t know based on the documentation I have loaded.\n\n"
            f"**Reason:** No .md/.txt files found in **{DOCS_DIR}**.\n\n"
            "Put your support docs in `data/raw/` and try again."
        )
        return msg, [], {
            "can_answer": False,
            "reason": "Docs folder missing or empty.",
            "best_score": 0.0,
            "threshold": float(idk_threshold),
        }, 0

    retriever = get_retriever(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)
    retrieved = retriever.search(question, k=top_k)
    retrieved_count = len(retrieved)

    # =========================
    # Day 6: HARD guardrail gate
    # If docs don't support the question => do NOT call the LLM
    # =========================
    decision = decide_if_can_answer(
        question=question,
        retrieved_chunks=retrieved,
        score_threshold=float(idk_threshold),
        min_chunks=1,
        check_top_n_chunks_text=3,
    )

    guardrail_info = {
        "can_answer": bool(decision.can_answer),
        "reason": str(decision.reason),
        "best_score": float(decision.best_score),
        "threshold": float(decision.threshold),
    }
    if hasattr(decision, "missing_terms"):
        guardrail_info["missing_terms"] = list(getattr(decision, "missing_terms"))

    if not decision.can_answer:
        # Keep UI clean: show no sources for IDK
        return idk_response(question), [], guardrail_info, retrieved_count

    # =========================
    # Day 9: LLM Toggle
    # If LLM is OFF => retrieval-only response (no generation)
    # =========================
    if not use_llm:
        answer_lines = [
            "**LLM is OFF (retrieval-only mode).**",
            "Here are the most relevant excerpts from the KB:",
            "",
        ]
        for i, c in enumerate(retrieved[:3], start=1):
            src = c.get("source", "unknown_file")
            cid = c.get("chunk_id", "unknown_chunk")
            snippet = (c.get("text") or "").strip().replace("\n", " ")
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            answer_lines.append(f"{i}. **{src} — chunk {cid}**: {snippet}")
        return "\n".join(answer_lines), retrieved, guardrail_info, retrieved_count

    # LLM answering (existing)
    # NOTE: src.answerer should read OPENAI_API_KEY from environment.
    answer_text, sources = answer_with_sources(
        question=question,
        sources=retrieved,
        temperature=temperature,
        max_tokens=max_tokens,
        # model is persisted in settings for Day 9,
        # but only pass it if your answerer supports it.
    )
    return answer_text, sources, guardrail_info, retrieved_count


# =========================
# Day 10: Analytics UI (shows above chat)
# =========================
if show_analytics:
    st.subheader("📊 Analytics v1 (Day 10)")

    qdf = safe_read_csv(QUESTIONS_CSV_PATH)
    fdf = safe_read_csv(FEEDBACK_CSV_PATH)

    total_questions = int(len(qdf)) if not qdf.empty else 0
    unique_sessions = int(qdf["session_id"].nunique()) if (not qdf.empty and "session_id" in qdf.columns) else 0

    # =========================
    # Day 10 FIX:
    # Only count feedback that matches question_ids in logs/questions.csv.
    # Also compute "feedback rate" as: questions with feedback / total questions
    # so it never goes above 100% from double-clicks.
    # =========================
    fdf_matched = pd.DataFrame()
    unmatched_feedback = 0

    if (not qdf.empty) and (not fdf.empty) and ("question_id" in qdf.columns) and ("message_id" in fdf.columns):
        q_ids = set(qdf["question_id"].astype(str).tolist())
        fdf_matched = fdf[fdf["message_id"].astype(str).isin(q_ids)].copy()
        unmatched_feedback = int(len(fdf) - len(fdf_matched))
    else:
        fdf_matched = pd.DataFrame()
        unmatched_feedback = int(len(fdf)) if not fdf.empty else 0

    matched_feedback_clicks = int(len(fdf_matched)) if not fdf_matched.empty else 0
    feedback_questions = int(fdf_matched["message_id"].nunique()) if (not fdf_matched.empty and "message_id" in fdf_matched.columns) else 0
    feedback_rate = (feedback_questions / total_questions) if total_questions else 0.0

    thumbs_up = int((fdf_matched["rating"] == "up").sum()) if (not fdf_matched.empty and "rating" in fdf_matched.columns) else 0
    thumbs_down = int((fdf_matched["rating"] == "down").sum()) if (not fdf_matched.empty and "rating" in fdf_matched.columns) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Questions", total_questions)
    c2.metric("Unique Sessions", unique_sessions)
    c3.metric("Matched Feedback Clicks", matched_feedback_clicks)
    c4.metric("Feedback Rate", f"{feedback_rate:.0%}")

    st.caption("Feedback rate = (questions that received feedback) ÷ total questions (only counting matched feedback)")

    if unmatched_feedback > 0:
        st.caption(f"Ignoring {unmatched_feedback} older feedback rows that don't match current questions log.")

    if not qdf.empty and "user_question" in qdf.columns:
        qdf["norm"] = qdf["user_question"].fillna("").apply(normalize_query)
        top = (
            qdf.groupby("norm")
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        st.markdown("### Top Queries (Top 10)")
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("### Most Recent Questions")
        cols = [c for c in ["ts_utc", "user_question", "can_answer", "best_score", "retrieved_chunks_count"] if c in qdf.columns]
        st.dataframe(qdf[cols].tail(10), use_container_width=True, hide_index=True)

    st.markdown("### Feedback Breakdown")
    st.write({"thumbs_up": thumbs_up, "thumbs_down": thumbs_down})

    if not fdf_matched.empty:
        st.markdown("### Most Recent Feedback (matched)")
        st.dataframe(fdf_matched.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("No matched feedback yet for the questions in logs/questions.csv.")

    st.divider()


# =========================
# Render chat history
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_feedback_buttons(msg)
            if show_guardrail_debug:
                render_guardrail_debug(msg.get("guardrail"))


# =========================
# Chat input
# =========================
user_input = st.chat_input("Type your question...")

if st.session_state.pending_user_message and not user_input:
    user_input = st.session_state.pending_user_message
    st.session_state.pending_user_message = None


# =========================
# Handle new message
# =========================
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    message_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer_text, sources, guardrail_info, retrieved_chunks_count = run_pipeline(user_input)

        st.markdown(answer_text)
        render_sources(sources)
        if show_guardrail_debug:
            render_guardrail_debug(guardrail_info)

        assistant_msg = {
            "role": "assistant",
            "message_id": message_id,
            "timestamp_utc": timestamp_utc,
            "question": user_input,
            "content": answer_text,
            "sources": sources,
            "guardrail": guardrail_info,
            # Day 9: capture admin settings on each answer
            "top_k": int(top_k),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "use_llm": bool(use_llm),
            "model": str(model),
            "idk_threshold": float(idk_threshold),
            "app_version": APP_VERSION,
        }

        # =========================
        # Day 10: Log the question (this is the key Day 10 feature)
        # Why (kid version): every question goes into a diary file so we can count later.
        # =========================
        append_question_csv(
            QUESTIONS_CSV_PATH,
            {
                "ts_utc": timestamp_utc,
                "session_id": st.session_state.session_id,
                "question_id": message_id,  # 1 question -> 1 assistant message
                "user_question": user_input,
                "model": str(model),
                "use_llm": str(bool(use_llm)),
                "top_k": int(top_k),
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "idk_threshold": float(idk_threshold),
                "can_answer": bool(guardrail_info.get("can_answer", False)),
                "best_score": guardrail_info.get("best_score", ""),
                "retrieved_chunks_count": int(retrieved_chunks_count),
                "sources_shown_count": int(len(sources)),
                "app_version": APP_VERSION,
            },
        )

        render_feedback_buttons(assistant_msg)

    st.session_state.messages.append(assistant_msg)
