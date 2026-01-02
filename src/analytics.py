# src/analytics.py
from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

LOG_DIR = Path("logs")
QUESTIONS_CSV = LOG_DIR / "questions.csv"
FEEDBACK_CSV = LOG_DIR / "feedback.csv"


def utc_now_iso() -> str:
    # Why: consistent timestamps regardless of your computer timezone
    return datetime.now(timezone.utc).isoformat()


def ensure_log_dir() -> None:
    # Why: prevent "folder not found" crashes
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _append_row_csv(path: Path, fieldnames: List[str], row: Dict) -> None:
    """
    Append one row to a CSV. If file doesn't exist, create it with headers.
    Why: CSV is simple, human-readable, and easy to analyze later.
    """
    ensure_log_dir()
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Only write keys we expect (avoid breaking CSV with surprise keys)
        cleaned = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(cleaned)


def log_question(
    *,
    question_id: str,
    session_id: str,
    user_text: str,
    model_name: str,
    temperature: float,
    top_k: int,
    llm_enabled: bool,
    max_tokens: int,
    num_sources: int,
) -> None:
    """
    Log each user question.
    Why: Without this, you can't compute "top queries" or usage volume.
    """
    fields = [
        "ts_utc",
        "question_id",
        "session_id",
        "user_text",
        "model_name",
        "temperature",
        "top_k",
        "llm_enabled",
        "max_tokens",
        "num_sources",
    ]
    _append_row_csv(
        QUESTIONS_CSV,
        fields,
        {
            "ts_utc": utc_now_iso(),
            "question_id": question_id,
            "session_id": session_id,
            "user_text": user_text,
            "model_name": model_name,
            "temperature": temperature,
            "top_k": top_k,
            "llm_enabled": str(llm_enabled),
            "max_tokens": max_tokens,
            "num_sources": num_sources,
        },
    )


def log_feedback(
    *,
    question_id: str,
    session_id: str,
    user_text: str,
    rating: str,  # "up" or "down"
    note: str = "",
) -> None:
    """
    Log thumbs up/down.
    Why: Later you can compute feedback rate and satisfaction.
    """
    fields = ["ts_utc", "question_id", "session_id", "user_text", "rating", "note"]
    _append_row_csv(
        FEEDBACK_CSV,
        fields,
        {
            "ts_utc": utc_now_iso(),
            "question_id": question_id,
            "session_id": session_id,
            "user_text": user_text,
            "rating": rating,
            "note": note,
        },
    )


def normalize_query(text: str) -> str:
    """
    Light normalization for top-queries counting.
    Why: "Reset password?" and "reset password" should count as the same.
    """
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text


@dataclass
class AnalyticsSummary:
    total_questions: int
    unique_sessions: int
    total_feedback: int
    feedback_rate: float  # feedback / questions
    thumbs_up: int
    thumbs_down: int


def compute_summary(questions_df, feedback_df) -> AnalyticsSummary:
    total_questions = int(len(questions_df))
    unique_sessions = int(questions_df["session_id"].nunique()) if total_questions else 0
    total_feedback = int(len(feedback_df))

    feedback_rate = (total_feedback / total_questions) if total_questions else 0.0

    thumbs_up = int((feedback_df["rating"] == "up").sum()) if total_feedback else 0
    thumbs_down = int((feedback_df["rating"] == "down").sum()) if total_feedback else 0

    return AnalyticsSummary(
        total_questions=total_questions,
        unique_sessions=unique_sessions,
        total_feedback=total_feedback,
        feedback_rate=feedback_rate,
        thumbs_up=thumbs_up,
        thumbs_down=thumbs_down,
    )
