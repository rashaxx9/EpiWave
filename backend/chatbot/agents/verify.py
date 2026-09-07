from __future__ import annotations

from typing import Any

from config import VERIFY_THRESHOLD


RetrievedChunk = tuple[str, dict[str, Any], float]


def verify_chunks(retrieved_chunks: list[RetrievedChunk], threshold: float = VERIFY_THRESHOLD) -> list[RetrievedChunk]:
    verified_chunks = [chunk for chunk in retrieved_chunks if chunk[2] >= threshold]
    verified_chunks.sort(key=lambda chunk: chunk[2], reverse=True)
    return verified_chunks
