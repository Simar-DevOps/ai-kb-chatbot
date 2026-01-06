# Quality Gate + Regression Process

## Purpose
This project uses an evaluation suite and a release gate to prevent regressions and reduce hallucination risk in a RAG-style support chatbot.

## Evidence (local)
Evaluation evidence is produced locally in `evals/results/` (raw runs + scored sheets). These files are intentionally not committed.

## Regression process (when you change X → rerun Y)
Any change to:
- retriever / retrieval
- guardrails (“I don’t know” behavior)
- answerer prompt/template
- docs (`data/raw`)

…requires rerunning:
1) Generate run CSV (`evals/results/run_*.csv`)
2) Create scoring sheet (`evals/results/run_*_scored.csv`)
3) Quick checks + failure buckets
4) Update “latest pointers” locally

## Merge gate routine
Ship only if:
- Format is consistent (citations + structure)
- Guardrails behave correctly on “should say IDK”
- No wrong-doc confident answers
- PASS rate meets target threshold (tracked locally)

## Release criteria
- Meet quality threshold (tracked locally)
- No hallucination / wrong-doc confident answers
- “Should say IDK” works correctly
- Answer format/citations consistent
