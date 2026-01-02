from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Config + Data Structures
# -----------------------------

@dataclass
class Chunk:
    doc_id: str          # filename (or any id)
    chunk_id: int        # chunk number within doc
    text: str            # chunk content
    start_char: int      # where in the doc it came from (nice for debugging)
    end_char: int


@dataclass
class RetrievalIndex:
    vectorizer: TfidfVectorizer
    matrix
    chunks: List[Chunk]


# -----------------------------
# Loading docs
# -----------------------------

def load_docs_from_folder(folder_path: str) -> Dict[str, str]:
    """
    Loads .txt and .md files from a folder into a dict: {doc_id: full_text}
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Docs folder not found: {folder.resolve()}")

    docs: Dict[str, str] = {}
    for p in sorted(folder.glob("**/*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}:
            docs[p.name] = p.read_text(encoding="utf-8", errors="ignore")

    if not docs:
        raise ValueError(f"No .txt/.md docs found in: {folder.resolve()}")

    return docs


# -----------------------------
# Chunking
# -----------------------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    doc_id: str,
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Simple character-based chunking.
    chunk_size/overlap are tunable in the Streamlit sidebar.
    """
    text = normalize_whitespace(text)
    chunks: List[Chunk] = []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    start = 0
    chunk_id = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk_str,
                    start_char=start,
                    end_char=end,
                )
            )
            chunk_id += 1

        # move window forward with overlap
        start = end - overlap

    return chunks


# -----------------------------
# Indexing
# -----------------------------

def build_tfidf_index(chunks: List[Chunk]) -> RetrievalIndex:
    """
    Builds a TF-IDF matrix for all chunks.
    """
    corpus = [c.text for c in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),   # helps match short phrases like "reset password"
        min_df=1,
    )
    matrix = vectorizer.fit_transform(corpus)
    return RetrievalIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def build_index_from_folder(
    docs_folder: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> RetrievalIndex:
    docs = load_docs_from_folder(docs_folder)

    all_chunks: List[Chunk] = []
    for doc_id, text in docs.items():
        all_chunks.extend(chunk_text(doc_id, text, chunk_size=chunk_size, overlap=overlap))

    return build_tfidf_index(all_chunks)


# -----------------------------
# Retrieval
# -----------------------------

def retrieve_top_k(
    index: RetrievalIndex,
    query: str,
    top_k: int = 5,
) -> List[Tuple[Chunk, float]]:
    """
    Returns [(chunk, score)] sorted by score desc.
    Score is cosine similarity between query vector and chunk vector.
    """
    query = (query or "").strip()
    if not query:
        return []

    q_vec = index.vectorizer.transform([query])
    sims = cosine_similarity(q_vec, index.matrix).flatten()

    # get top_k indices
    top_k = max(1, int(top_k))
    best_idx = sims.argsort()[::-1][:top_k]

    results: List[Tuple[Chunk, float]] = []
    for i in best_idx:
        results.append((index.chunks[i], float(sims[i])))

    return results
