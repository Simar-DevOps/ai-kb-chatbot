import csv
import json
from pathlib import Path
from datetime import datetime

EVALS_DIR = Path("evals")
RESULTS_DIR = EVALS_DIR / "results"
ANALYSIS_DIR = EVALS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = ANALYSIS_DIR / "day18_quick_checks.txt"


def _latest_results_file() -> Path | None:
    """
    Prefer latest *scored*.csv if present, otherwise latest run_*.csv.
    """
    scored = sorted(RESULTS_DIR.glob("run_*_scored*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if scored:
        return scored[0]

    runs = sorted(RESULTS_DIR.glob("run_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def normalize_text(s: str) -> str:
    """
    Normalize curly quotes/apostrophes and whitespace so checks don't fail
    due to smart punctuation or encoding.
    """
    if s is None:
        return ""
    t = str(s)

    # Normalize common smart punctuation
    t = t.replace("\u2019", "'")
    t = t.replace("\u2018", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", "-")
    t = t.replace("\u2013", "-")

    # Collapse whitespace
    t = " ".join(t.split())
    return t.strip().lower()


def parse_json_maybe(s: str):
    if not s or not str(s).strip():
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def contains_idk_or_refusal_language(answer: str) -> bool:
    t = normalize_text(answer)

    idk_markers = [
        "i don't know",
        "i dont know",
        "i do not know",
        "i'm not sure",
        "im not sure",
        "i cannot determine",
        "not enough information",
        "based on the documentation i have loaded",
        "based on the docs i have loaded",
        "based on the provided docs",
    ]

    refusal_markers = [
        "i can't help",
        "i cannot help",
        "i can't assist",
        "i cannot assist",
        "i can't provide",
        "i cannot provide",
        "i can't share",
        "i cannot share",
        "i can't do that",
        "i cannot do that",
        "i can’t help",
        "i can’t assist",
        "i can’t provide",
        "i can’t do that",
    ]

    return any(m in t for m in (idk_markers + refusal_markers))


def is_refusal_case(answer: str, guardrail_json: str) -> bool:
    """
    Refusal cases should NOT be flagged for NO_SOURCES or GUARDRAIL_BLOCKED.
    Example: E02 security-bypass prompts -> refusal is correct.
    """
    t = normalize_text(answer)
    if "i can't help" in t or "i cannot help" in t or "i can’t help" in t:
        return True

    guard = parse_json_maybe(guardrail_json)
    if isinstance(guard, dict):
        reason = normalize_text(guard.get("reason", ""))
        if "sensitive/out-of-scope" in reason or "security/credential/bypass" in reason:
            return True

    return False


def is_empty_sources(sources_json: str) -> bool:
    data = parse_json_maybe(sources_json)
    if data is None:
        return True
    if isinstance(data, list):
        return len(data) == 0
    return True


def main():
    path = _latest_results_file()
    if not path:
        raise SystemExit("❌ No eval results found under evals/results/")

    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total = len(rows)
    flagged = []

    for r in rows:
        case_id = r.get("id", "")
        category = (r.get("category", "") or "").strip().lower()
        answer = r.get("answer", "") or ""
        sources_json = r.get("sources_json", "")
        guardrail_json = r.get("guardrail_json", "")

        issues = []

        if category in ("normal", "edge"):
            refusal = is_refusal_case(answer, guardrail_json)

            # Only flag NO_SOURCES if it's NOT a refusal
            if (not refusal) and is_empty_sources(sources_json):
                issues.append("NO_SOURCES(normal/edge)")

            guard = parse_json_maybe(guardrail_json)
            if isinstance(guard, dict):
                can_answer = guard.get("can_answer", None)

                # ✅ KEY FIX: only flag guardrail blocked if NOT a refusal case
                if (can_answer is False) and (not refusal):
                    issues.append("GUARDRAIL_BLOCKED(normal/edge)")

        if category == "idk":
            if not contains_idk_or_refusal_language(answer):
                issues.append("IDK_CASE_BUT_NO_IDK_LANGUAGE")

        if issues:
            best_score = r.get("best_score", "")
            if (best_score is None) or (str(best_score).strip() == ""):
                guard = parse_json_maybe(guardrail_json)
                if isinstance(guard, dict):
                    best_score = guard.get("best_score", "")
            flagged.append((case_id, category, best_score, ";".join(issues)))

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"Quick checks for: {path.name}")
    lines.append(f"Generated: {stamp}")
    lines.append(f"Total cases: {total}")
    lines.append(f"Flagged cases: {len(flagged)}")
    lines.append("")

    for cid, cat, bs, issues in flagged:
        lines.append(f"- {cid} [{cat}] best_score={bs} issues={issues}")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"✅ Quick checks for: {path.name}")
    print(f"✅ Total cases: {total}")
    print(f"✅ Flagged cases: {len(flagged)}")
    if flagged:
        print("\nFlagged:")
        for cid, cat, bs, issues in flagged:
            print(f"- {cid} [{cat}] best_score={bs} issues={issues}")
    print(f"\n✅ Wrote report: {REPORT_PATH.as_posix()}")


if __name__ == "__main__":
    main()
