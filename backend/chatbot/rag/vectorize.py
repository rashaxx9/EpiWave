from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache
from typing import Any

import chromadb
from langchain_text_splitters import TokenTextSplitter
from openai import OpenAI

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
)


@lru_cache(maxsize=1)
def _get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            # Storing the backend alongside the model helps us catch accidental
            # index/query mismatches when users switch providers later.
            "embedding_backend": _embedding_backend_name(),
            "hnsw:space": "cosine",
        },
    )
    _ensure_embedding_compatibility(collection)
    return collection


def _embedding_backend_name() -> str:
    return "openai" if LLM_PROVIDER == "openai" else "sentence-transformers"


def _ensure_embedding_compatibility(collection) -> None:
    metadata = collection.metadata or {}
    stored_model = metadata.get("embedding_model")
    stored_backend = metadata.get("embedding_backend")

    if stored_model and stored_model != EMBEDDING_MODEL:
        raise ValueError(
            "The existing ChromaDB index was built with a different EMBEDDING_MODEL. "
            f"Expected '{stored_model}' but config is '{EMBEDDING_MODEL}'."
        )
    if stored_backend and stored_backend != _embedding_backend_name():
        raise ValueError(
            "The existing ChromaDB index was built with a different embedding backend. "
            f"Expected '{stored_backend}' but config implies '{_embedding_backend_name()}'."
        )


@lru_cache(maxsize=1)
def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer
    # Force CPU to avoid GPU memory contention with Ollama's Llama3 model.
    # On 8GB M2 Macs, both models can't fit in GPU memory simultaneously.
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    if LLM_PROVIDER == "openai":
        response = _get_openai_client().embeddings.create(
            model=EMBEDDING_MODEL,
            input=list(texts),
        )
        return [list(map(float, item.embedding)) for item in response.data]

    model = _get_sentence_transformer()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    return [list(map(float, vector)) for vector in vectors]


def embed_query(query: str) -> list[float]:
    return embed_texts([query])[0]


def _batched(items: Sequence[dict[str, Any]], batch_size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _chunk_documents(documents: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        encoding_name="cl100k_base",
    )

    chunked_documents: list[dict[str, Any]] = []
    for document in documents:
        text = document["text"].strip()
        if not text:
            continue

        metadata = document["metadata"]
        chunks = splitter.split_text(text)
        for chunk_index, chunk_text in enumerate(chunks):
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            # Build chunk metadata — include all fields from the source metadata
            # plus the chunk index. Works for both EEG and any future doc types.
            chunk_meta: dict[str, Any] = {
                "filename": metadata["filename"],
                "chunk_index": chunk_index,
            }
            # Carry over optional EEG-specific metadata fields
            for key in ("scan_date", "duration_seconds", "num_channels", "document_type"):
                if key in metadata:
                    chunk_meta[key] = metadata[key]

            # Use scan_date in the ID if available, otherwise fall back to a
            # simple index. This keeps IDs unique across re-analyses.
            date_tag = metadata.get("scan_date", "doc")
            chunked_documents.append(
                {
                    "id": f"{metadata['filename']}::{date_tag}::chunk:{chunk_index}",
                    "text": cleaned_chunk,
                    "metadata": chunk_meta,
                }
            )

    return chunked_documents


def clear_vector_store() -> None:
    collection = _get_chroma_collection()
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)


def vectorize_documents(documents: Sequence[dict[str, Any]]) -> int:
    collection = _get_chroma_collection()
    chunked_documents = _chunk_documents(documents)

    if not chunked_documents:
        return 0

    # Delete by filename before re-upserting so re-analysed scans do not leave
    # behind stale chunks from a previous analysis of the same file.
    filenames = {document["metadata"]["filename"] for document in chunked_documents}
    for filename in filenames:
        collection.delete(where={"filename": filename})

    for batch in _batched(chunked_documents, batch_size=64):
        texts = [item["text"] for item in batch]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=texts,
            metadatas=[item["metadata"] for item in batch],
            embeddings=embeddings,
        )

    return len(chunked_documents)
