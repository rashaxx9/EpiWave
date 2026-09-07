from __future__ import annotations

from typing import Any

from config import TOP_K
from rag.vectorize import _get_chroma_collection, embed_query


RetrievedChunk = tuple[str, dict[str, Any], float]


def retrieve_chunks(user_query: str) -> list[RetrievedChunk]:
    collection = _get_chroma_collection()
    if collection.count() == 0:
        return []

    query_embedding = embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_chunks: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # Chroma returns cosine distance, so we invert it into a more intuitive
        # similarity-style score where larger is better.
        score = max(0.0, 1.0 - float(distance))
        retrieved_chunks.append((document, metadata, score))

    return retrieved_chunks
