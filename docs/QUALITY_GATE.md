# Week 3 Day 21 — Quality Gate + Regression Process

## Evidence used

Baseline evidence run (used for scoring + failure buckets):
- Raw: `evals/results/run_2026-01-04_200800.csv`
- Scored: `evals/results/run_2026-01-04_200800_scored.csv`

Supporting analysis outputs:
- Day 18 Quick Checks: `evals/analysis/day18_quick_checks.txt`
- Day 19 Failure Summary: `evals/analysis/day19_out_fixbucket/top_failures.md`

Most recent captured run files (stored for reference):
- Raw: `evals/results/run_2026-01-05_002114.csv`
- Scored: `evals/results/run_2026-01-05_002114_scored.csv`

## Quality gate decision (PASS/FAIL)

**Decision: FAIL**

Reason (based on baseline evidence run `run_2026-01-04_200800_*`):
- Pass rate (bucket PASS or score >= 4.0): **3.3% (1/30)**
- Failure buckets:
  - format: **60.0%**
  - guardrail: **36.7%**

## Regression process (when you change X → rerun Y)

Any change to:
- retriever / retrieval → rerun eval + scoring + checks
- guardrails → rerun eval + scoring + checks
- answerer prompt → rerun eval + scoring + checks
- docs (`data/raw`) → rerun eval + scoring + checks

## PR / merge gate routine (every time)

1) Generate run CSV → `evals/results/run_*.csv`
2) Score/prefill → `evals/results/*_scored.csv`
3) Quick checks (Day 18)
4) Failure buckets (Day 19)
5) Update this file with the exact evidence run filenames
6) Ship only if PASS criteria met

Note: The run generation/scoring may be executed via local scripts or manual steps. The source of truth is the evidence run filenames listed above.

## Release criteria (PASS)

PASS only if ALL are true:
- PASS rate meets target: **≥ 80%**
- No new hallucination / wrong-doc confident answers (especially on “should say IDK” cases)
- “Should say IDK” behaves correctly (guardrails work, not over-blocking answerable questions)
- format bucket reduced significantly (citations + answer formatting consistent)
