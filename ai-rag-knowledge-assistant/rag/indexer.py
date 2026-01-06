from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from .chunking import chunk_text, Chunk
from .embeddings import embed_texts
from .config import Settings

META_FILE = "meta.json"
INDEX_FILE = "faiss.index"


def build_index_from_dir(data_dir: str, index_dir: str, settings: Settings) -> None:
    data_path = Path(data_dir)
    out_path = Path(index_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    chunks: List[Chunk] = []
    for fp in sorted(data_path.glob("*")):
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in [".txt", ".md"]:
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        chunks.extend(chunk_text(text, source=fp.name))

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, settings)
    if embeddings.shape[0] == 0:
        raise ValueError("No chunks were produced from documents.")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    meta = [{"text": c.text, "source": c.source} for c in chunks]
    (out_path / META_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    faiss.write_index(index, str(out_path / INDEX_FILE))


def load_index(index_dir: str) -> Dict:
    p = Path(index_dir)
    index = faiss.read_index(str(p / INDEX_FILE))
    meta = json.loads((p / META_FILE).read_text(encoding="utf-8"))
    return {"index": index, "meta": meta}
