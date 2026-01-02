# src/admin_settings.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

SETTINGS_PATH = Path("data") / "admin_settings.json"


@dataclass
class AdminSettings:
    # Core admin controls
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    top_k: int = 5
    use_llm: bool = True
    max_tokens: int = 500


def load_settings() -> AdminSettings:
    """Load settings from disk; fall back to defaults if missing/bad."""
    try:
        if SETTINGS_PATH.exists():
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return AdminSettings(**data)
    except Exception:
        # If file is corrupted or schema changed, ignore and use defaults.
        pass
    return AdminSettings()


def save_settings(settings: AdminSettings) -> None:
    """Persist settings to disk."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")


def reset_settings() -> AdminSettings:
    """Delete the settings file and return defaults."""
    try:
        if SETTINGS_PATH.exists():
            SETTINGS_PATH.unlink()
    except Exception:
        pass
    return AdminSettings()
