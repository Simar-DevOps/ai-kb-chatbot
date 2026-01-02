from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import re

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise ImportError(
        "Missing dependency: rank-bm25. Install it with:\n"
        "pip install rank-bm25"
    ) from e

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def simple_tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Splits text into overlapping chunks by character length.
    chunk_size/overlap are easy knobs for later.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


@dataclass
class KBChunk:
    source: str   # filename (or path)
    chunk_id: int # chunk number within file
    text: str     # chunk content


class BM25Retriever:
    def __init__(self, chunks: List[KBChunk]):
        self.chunks = chunks
        self.tokenized = [simple_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        q_tokens = simple_tokenize(query)
        if not q_tokens:
            return []

        scores = self.bm25.get_scores(q_tokens)

        # If BM25 can't find matches (all zeros), fall back to keyword counting
        max_score = float(max(scores)) if len(scores) else 0.0
        if max_score <= 0.0:
            fallback_scores: List[float] = []
            for c in self.chunks:
                text_l = c.text.lower()
                src_l = c.source.lower()

                # Count token occurrences in text + extra weight if token appears in filename
                s = 0.0
                for t in q_tokens:
                    s += text_l.count(t)
                    if t in src_l:
                        s += 5.0

                fallback_scores.append(s)
            scores = fallback_scores

        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        top = ranked[:k]

        results: List[Dict[str, Any]] = []
        for i in top:
            c = self.chunks[i]
            results.append({
                "source": c.source,
                "chunk_id": c.chunk_id,
                "text": c.text,
                "score": float(scores[i]),
            })
        return results


def build_kb_from_data_dir(
    data_dir: str | Path,
    k_chunk_size: int = 800,
    k_overlap: int = 150
) -> BM25Retriever:
    """
    Loads .txt/.md files from a folder, chunks them, and builds BM25 index.

    IMPORTANT for Day 4:
    Each chunk stores metadata (source + chunk_id) so we can show citations.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    chunks: List[KBChunk] = []
    for fp in sorted(list(data_dir.glob("*.txt")) + list(data_dir.glob("*.md"))):
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        file_chunks = chunk_text(raw, chunk_size=k_chunk_size, overlap=k_overlap)
        for idx, ch in enumerate(file_chunks, start=1):
            chunks.append(KBChunk(source=fp.name, chunk_id=idx, text=ch))

    if not chunks:
        raise ValueError(f"No .txt or .md files found in {data_dir}")

    return BM25Retriever(chunks)
