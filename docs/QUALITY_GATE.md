# Week 3 Day 21 — Quality Gate + Regression Process

## Evidence used
- Day 18 Quick Checks: `evals/analysis/day18_quick_checks.txt`
- Latest scored run: `evals/results/run_2026-01-04_200800_scored.csv`
- Day 19 Failure Summary: `evals/analysis/day19_out_fixbucket/top_failures.md`
- Evidence run: evals/results/run_2026-01-05_002114.csv + evals/results/run_  2026-01-05_002114_scored.csv

## Quality gate decision (PASS/FAIL)
**Decision: FAIL**

Reason:
- Pass rate (bucket PASS or score>=4.0): **3.3% (1/30)**
- Failure buckets:
  - format: **60%**
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
5) Ship only if PASS criteria met

## Release criteria (PASS)
PASS only if:
- PASS rate meets target (set by rubric acceptance criteria)
- No new hallucination / wrong-doc confident answers
- “Should say IDK” behaves correctly
- format bucket reduced significantly (citations + answer formatting consistent)
