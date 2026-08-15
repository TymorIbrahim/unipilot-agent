"""Query embeddings, and the cosine the reranker blends.

Ported from UniPilot, trimmed to what a READER needs: this deployment never
builds the index, so `embed_documents` and the upsert path are gone.

**Pinned to LLMod regardless of which provider chat is using.** The Pinecone
index was built with `MB5R2CF-azure/text-embedding-3-small`; embedding a query
with any other model puts it in a different vector space, and the search still
returns 50 results -- just meaningless ones, ranked confidently. That is why
`Settings` configures embeddings separately from chat rather than deriving one
from the other.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def _embeddings_client_for(api_key: str, base_url: str, model: str) -> Any | None:
    """One client per (credential, model). Cached because building it opens a
    connection pool, and a cold serverless invocation may embed more than once."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        logger.warning("embeddings_unavailable: langchain_openai is not installed")
        return None
    return OpenAIEmbeddings(api_key=api_key, base_url=base_url or None, model=model)


@lru_cache(maxsize=256)
def embed_query_cached(
    query: str, api_key: str, base_url: str, model: str
) -> tuple[float, ...] | None:
    """Cached query embedding, keyed by credentials AND model.

    Credentials stay explicit parameters so they are part of the cache key: a
    provider swap must not silently reuse vectors embedded by the old one.

    Returns None -- never raises -- so an embedding outage degrades the ranking
    to keyword-only instead of failing the request.
    """
    if not api_key:
        return None
    client = _embeddings_client_for(api_key, base_url, model)
    if client is None:
        return None
    try:
        vector = client.embed_query(query or "")
    except Exception:  # noqa: BLE001 -- an outage is a missing signal, not an error
        logger.exception("embedding_query_failed")
        return None
    return tuple(float(value) for value in vector) if vector else None


def embed_query(text: str, *, settings: Any | None = None) -> list[float] | None:
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    if not settings.embeddings_available():
        return None
    cached = embed_query_cached(
        text or "",
        settings.llm_embedding_api_key.strip(),
        settings.llm_embedding_base_url.strip(),
        settings.llm_embedding_model.strip(),
    )
    return list(cached) if cached else None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def reset_embeddings_client_cache() -> None:
    _embeddings_client_for.cache_clear()
    embed_query_cached.cache_clear()


__all__ = [
    "cosine_similarity",
    "embed_query",
    "embed_query_cached",
    "reset_embeddings_client_cache",
]
