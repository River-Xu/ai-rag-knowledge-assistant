from __future__ import annotations

from typing import Dict, List

import faiss

from .embeddings import embed_texts
from .config import Settings


def retrieve(question: str, index: Dict, top_k: int, settings: Settings) -> List[dict]:
    top_k = max(1, int(top_k))
    q_emb = embed_texts([question], settings)
    if q_emb.shape[0] == 0:
        return []
    faiss.normalize_L2(q_emb)
    D, I = index["index"].search(q_emb, top_k)

    hits: List[dict] = []
    for idx in I[0]:
        if idx < 0:
            continue
        m = index["meta"][idx]
        hits.append({"text": m["text"], "source": m.get("source", "unknown")})
    return hits
