# evals/run_all.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALS_DIR = REPO_ROOT / "evals"
ANALYSIS_REPORT = EVALS_DIR / "analysis" / "day18_quick_checks.txt"


def run(cmd: list[str]) -> None:
    print(f"\n▶ {' '.join(cmd)}")
    p = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def parse_flagged_count(report_path: Path) -> int:
    if not report_path.exists():
        return -1
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if line.lower().startswith("flagged cases:"):
            # e.g. "Flagged cases: 0"
            try:
                return int(line.split(":")[1].strip())
            except Exception:
                return -1
    return -1


def main() -> None:
    # 1) Run eval
    run([sys.executable, "evals/run_eval.py"])

    # 2) Make scoring sheet
    run([sys.executable, "evals/make_scoring_sheet.py"])

    # 3) Quick checks
    run([sys.executable, "evals/quick_checks.py"])

    flagged = parse_flagged_count(ANALYSIS_REPORT)
    if flagged == 0:
        print("\n✅✅✅ PASS: Quick checks flagged 0 cases.")
        raise SystemExit(0)

    if flagged > 0:
        print(f"\n❌ FAIL: Quick checks flagged {flagged} cases.")
        print(f"Open: {ANALYSIS_REPORT.as_posix()}")
        raise SystemExit(1)

    print("\n⚠️ Could not read flagged count from report (but scripts ran).")
    print(f"Check: {ANALYSIS_REPORT.as_posix()}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
