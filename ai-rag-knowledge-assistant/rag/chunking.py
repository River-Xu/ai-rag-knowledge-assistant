from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    text: str
    source: str


def chunk_text(text: str, *, source: str, chunk_size: int = 800, overlap: int = 120) -> List[Chunk]:
    text = (text or "").strip()
    if not text:
        return []

    # normalize whitespace
    text = " ".join(text.split())

    chunks: List[Chunk] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        part = text[start:end].strip()
        if part:
            chunks.append(Chunk(text=part, source=source))
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks
