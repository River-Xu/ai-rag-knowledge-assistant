from __future__ import annotations

import hashlib
from typing import List

import numpy as np
from openai import OpenAI

from .config import Settings


def _hash_embedding(text: str, dim: int = 384) -> List[float]:
    # Deterministic local embedding fallback: hash → vector
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big", signed=False))
    vec = rng.normal(0, 1, size=(dim,))
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec.astype(np.float32).tolist()


def embed_texts(texts: List[str], settings: Settings) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    # No API key: local fallback embeddings (works offline)
    if not settings.openai_api_key:
        return np.array([_hash_embedding(t) for t in texts], dtype=np.float32)

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    resp = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype=np.float32)
