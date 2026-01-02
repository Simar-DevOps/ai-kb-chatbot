import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/chunks.json")

def chunk_text(text: str):
    # Simple chunking: split by blank lines and keep meaningful paragraphs
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    # Filter out tiny chunks
    return [p for p in parts if len(p) >= 40]

def main():
    chunks = []
    for fp in sorted(RAW_DIR.glob("*.md")):
        text = fp.read_text(encoding="utf-8")
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "doc": fp.name,
                "chunk_id": f"{fp.stem}-{i}",
                "text": chunk
            })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(chunks)} chunks to {OUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
