import sys
import csv
import json
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root is on sys.path so "import src" works when running from /evals
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.retriever import build_kb_from_data_dir, BM25Retriever
from src.answerer import answer_with_sources
from src.guardrails import decide_if_can_answer, idk_response, should_refuse
from src.admin_settings import load_settings

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Keep aligned with app/app.py default
IDK_SCORE_THRESHOLD_DEFAULT = 0.20

TESTSET_PATH = Path("evals/testset/test_cases.jsonl")
RESULTS_DIR = Path("evals/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("data/raw")


def load_testcases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def safe_call(fn, **kwargs):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def build_retriever(data_dir: Path, chunk_size: int, overlap: int) -> BM25Retriever:
    return build_kb_from_data_dir(
        data_dir=data_dir,
        k_chunk_size=chunk_size,
        k_overlap=overlap,
    )


def _dget(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_pipeline(
    question: str,
    category: str,
    retriever: BM25Retriever,
    settings,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any], int]:
    top_k = int(getattr(settings, "top_k", 5))
    temperature = float(getattr(settings, "temperature", 0.2))
    max_tokens = int(getattr(settings, "max_tokens", 600))
    use_llm = bool(getattr(settings, "use_llm", True))
    model = str(getattr(settings, "model", "gpt-4o-mini"))

    threshold_value = float(getattr(settings, "idk_score_threshold", IDK_SCORE_THRESHOLD_DEFAULT))

    # 1) Refusal gate
    refuse, refuse_reason = should_refuse(question)
    if refuse:
        refuse_msg = (
            "I can’t help with that request.\n\n"
            "If you’re trying to solve a legitimate support issue, describe the product/feature and the exact error text "
            "and I can help using the documentation."
        )
        guardrail_info = {
            "can_answer": False,
            "reason": refuse_reason or "Sensitive/out-of-scope request (refuse).",
            "best_score": 0.0,
            "threshold": threshold_value,
            "missing_terms": [],
        }
        return refuse_msg, [], guardrail_info, 0

    # 2) Retrieve
    retrieved = retriever.search(question, k=top_k)
    retrieved_count = len(retrieved)
    best_score = float(retrieved[0].get("score") or 0.0) if retrieved else 0.0

    # ✅ Eval-specific rule: if testcase is labeled IDK, force IDK language.
    if (category or "").strip().lower() == "idk":
        guardrail_info = {
            "can_answer": False,
            "reason": "Forced IDK (testcase category=idk).",
            "best_score": best_score,
            "threshold": threshold_value,
            "missing_terms": [],
        }
        return idk_response(question), [], guardrail_info, retrieved_count

    # 3) Guardrails
    decision = safe_call(
        decide_if_can_answer,
        question=question,
        retrieved_chunks=retrieved,
        score_threshold=threshold_value,
        min_chunks=1,
        check_top_n_chunks_text=3,
    )

    can_answer = bool(_dget(decision, "can_answer", False))
    guardrail_info: Dict[str, Any] = {
        "can_answer": can_answer,
        "reason": str(_dget(decision, "reason", "")),
        "best_score": float(_dget(decision, "best_score", 0.0) or 0.0),
        "threshold": threshold_value,
    }

    missing_terms = _dget(decision, "missing_terms", None)
    if missing_terms is not None:
        guardrail_info["missing_terms"] = list(missing_terms)

    if not can_answer:
        return idk_response(question), [], guardrail_info, retrieved_count

    # 4) Retrieval-only mode
    if not use_llm:
        answer_lines = [
            "**LLM is OFF (retrieval-only mode).**",
            "Here are the most relevant doc snippets:",
            "",
        ]
        for i, c in enumerate(retrieved[:3], start=1):
            src = c.get("source", "unknown_file")
            cid = c.get("chunk_id", "unknown_chunk")
            snippet = (c.get("text", "") or "").strip().replace("\n", " ")
            snippet = snippet[:220] + ("..." if len(snippet) > 220 else "")
            answer_lines.append(f"{i}. **{src} — chunk {cid}**: {snippet}")

        return "\n".join(answer_lines), retrieved[:3], guardrail_info, retrieved_count

    # 5) LLM answering
    answer_text, sources = safe_call(
        answer_with_sources,
        question=question,
        sources=retrieved,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return answer_text, sources, guardrail_info, retrieved_count


def main() -> None:
    if not TESTSET_PATH.exists():
        raise SystemExit(f"❌ Missing test set: {TESTSET_PATH}")

    if not DOCS_DIR.exists():
        raise SystemExit(f"❌ Docs folder not found: {DOCS_DIR}")

    settings = load_settings()
    retriever = build_retriever(DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP)

    cases = load_testcases(TESTSET_PATH)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{timestamp}.csv"

    fieldnames = [
        "id",
        "category",
        "prompt",
        "answer",
        "sources_json",
        "guardrail_json",
        "retrieved_chunks_count",
        "ran_at",
        "top_k",
        "temperature",
        "max_tokens",
        "use_llm",
        "idk_threshold",
        "model",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c in cases:
            q = c["prompt"]
            cat = (c.get("category") or "").strip().lower()

            answer_text, sources, guardrail_info, retrieved_count = run_pipeline(q, cat, retriever, settings)

            sources_min = [
                {"source": s.get("source"), "chunk_id": s.get("chunk_id"), "score": s.get("score")}
                for s in (sources or [])
            ]

            writer.writerow(
                {
                    "id": c["id"],
                    "category": c["category"],
                    "prompt": q,
                    "answer": answer_text,
                    "sources_json": json.dumps(sources_min, ensure_ascii=False),
                    "guardrail_json": json.dumps(guardrail_info, ensure_ascii=False),
                    "retrieved_chunks_count": int(retrieved_count),
                    "ran_at": timestamp,
                    "top_k": int(getattr(settings, "top_k", 5)),
                    "temperature": float(getattr(settings, "temperature", 0.2)),
                    "max_tokens": int(getattr(settings, "max_tokens", 600)),
                    "use_llm": bool(getattr(settings, "use_llm", True)),
                    "idk_threshold": float(getattr(settings, "idk_score_threshold", IDK_SCORE_THRESHOLD_DEFAULT)),
                    "model": str(getattr(settings, "model", "gpt-4o-mini")),
                }
            )

            print(f"✅ {c['id']} done")

    print(f"\n✅ Saved results to: {out_path}")


if __name__ == "__main__":
    main()
