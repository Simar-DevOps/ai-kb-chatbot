# src/feedback_store.py
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any

CSV_COLUMNS = [
    "timestamp_utc",
    "session_id",
    "message_id",
    "rating",              # "up" or "down"
    "user_question",
    "assistant_answer",
    "sources_json",        # JSON string of list
    "top_k",
    "temperature",
    "max_tokens",
    "use_llm",
    "model",
    "app_version",
]


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_feedback_csv(csv_path: Path, row: Dict[str, Any]) -> None:
    _ensure_parent_dir(csv_path)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

        if not file_exists:
            writer.writeheader()

        clean_row = {col: row.get(col, "") for col in CSV_COLUMNS}
        writer.writerow(clean_row)


def append_feedback_jsonl(jsonl_path: Path, payload: Dict[str, Any]) -> None:
    _ensure_parent_dir(jsonl_path)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
