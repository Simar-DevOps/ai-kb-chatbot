from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


RESULTS_DIR = Path("evals") / "results"


SCORE_COLUMNS = [
    "score_correctness",
    "score_groundedness",
    "score_retrieval_relevance",
    "score_helpfulness",
    "score_guardrails",
    "critical_fail_flags",
    "overall_score",
    "pass_fail",
    "notes",
    "fix_bucket",
]


def _find_latest_raw_run(results_dir: Path) -> Path:
    """
    Pick the newest raw eval run file.
    We exclude any file that contains 'scored' to avoid selecting scoring sheets.
    """
    candidates = [
        p for p in results_dir.glob("run_*.csv")
        if "scored" not in p.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No raw run_*.csv found in {results_dir}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_in_arg(argv: list[str]) -> Path | None:
    """
    Support: --in <path>  OR  --input <path>
    """
    for flag in ("--in", "--input"):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 >= len(argv):
                raise ValueError(f"Missing value after {flag}")
            return Path(argv[idx + 1])
    return None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    in_path = _parse_in_arg(sys.argv)
    if in_path is None:
        in_path = _find_latest_raw_run(RESULTS_DIR)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)

    # Ensure scoring columns exist (blank for manual scoring)
    for c in SCORE_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    out_path = in_path.with_name(in_path.stem + "_scored.csv")
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"✅ Created scoring sheet: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
