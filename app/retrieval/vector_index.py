"""The Pinecone half of retrieval: content-addressed ids, top-K, and fetch.

Ported from UniPilot's `wiki_vector_index` + `vector_store`, collapsed into one
module and trimmed to the READ path. This deployment never builds the index --
`upsert`, `delete`, `ensure_index` and `list_ids` are gone, because the vectors
already exist and a serverless function that could rewrite them is a hazard, not
a feature.

**The stored vectors are still addressable, and that is not luck.**
`chunk_vector_id` hashes `source_file|section_title|content`, and the seed's
chunking reproduces UniPilot's `load_wiki_chunks` byte for byte -- verified
4,895/4,895 ids, and 8/8 sampled ids fetched live from the index. So the
semantic half works with no reindexing and no re-embedding of the corpus: the
only embedding call at runtime is the QUERY.

Everything here returns empty rather than raising. An unreachable index is a
missing SIGNAL -- the ranking degrades to keyword-only, which
`reranker.semantic_similarity_score` handles by scoring every chunk 0.0 so the
order stays monotone in the keyword score. An exception would instead fail a
request that could still have answered.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from app.retrieval.wiki_chunk import WikiChunk

logger = logging.getLogger(__name__)


def chunk_vector_id(chunk: WikiChunk) -> str:
    """Stable, content-addressed Pinecone id for a chunk.

    Pure ASCII on purpose: Pinecone rejects non-ASCII record ids and this corpus
    is largely Hebrew, so the readable `source_file::section_title` form is not a
    legal id. Content-addressing means an edited chunk gets a NEW id, which is
    what lets a reindex detect and prune what went stale.
    """
    digest = hashlib.sha256(
        f"{chunk.source_file}|{chunk.section_title}|{chunk.content}".encode("utf-8")
    ).hexdigest()
    return digest


@lru_cache(maxsize=1)
def _index(api_key: str, index_name: str) -> Any | None:
    try:
        from pinecone import Pinecone
    except ImportError:
        logger.warning("pinecone package is not installed; semantic search is off")
        return None
    try:
        return Pinecone(api_key=api_key).Index(index_name)
    except Exception:  # noqa: BLE001
        logger.exception("pinecone_index_open_failed")
        return None


def _open(settings: Any) -> Any | None:
    if not settings.vector_index_enabled():
        return None
    return _index(settings.pinecone_api_key.strip(), settings.pinecone_index_name.strip())


def query_semantic_candidates(
    *, query: str, chunks_by_id: dict[str, WikiChunk], limit: int, settings: Any | None = None
) -> list[tuple[WikiChunk, float]]:
    """Top-`limit` chunks by cosine similarity, hydrated from the artifact.

    `chunks_by_id` is the corpus keyed by `chunk_vector_id`. A Pinecone hit with
    no matching chunk means the index holds a vector for content that has since
    changed; dropping it is correct, and it is logged because a large gap means
    the artifact and the index have drifted apart.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    if limit <= 0:
        return []
    index = _open(settings)
    if index is None:
        return []

    from app.retrieval.embedding_service import embed_query

    vector = embed_query(query, settings=settings)
    if not vector:
        return []

    try:
        response = index.query(vector=vector, top_k=limit, include_values=False)
        matches = getattr(response, "matches", None) or response.get("matches", [])
    except Exception:  # noqa: BLE001
        logger.exception("wiki_semantic_query_failed")
        return []

    hits: list[tuple[WikiChunk, float]] = []
    for match in matches:
        vector_id = match.get("id") if isinstance(match, dict) else match.id
        score = match.get("score") if isinstance(match, dict) else match.score
        chunk = chunks_by_id.get(str(vector_id))
        if chunk is not None:
            hits.append((chunk, float(score or 0.0)))
    if len(hits) < len(matches):
        logger.warning(
            "wiki_semantic_stale_vectors_skipped matched=%d hydrated=%d",
            len(matches),
            len(hits),
        )
    return hits


def fetch_chunk_vectors(
    chunks: Sequence[WikiChunk], *, settings: Any | None = None
) -> dict[str, tuple[float, ...]]:
    """Vectors for an arbitrary candidate set, keyed by `chunk_vector_id`.

    ONE batched round trip for the whole set, never a call per chunk. The
    reranker needs this because its candidates include keyword-only hits that
    never appeared in a corpus-wide top-K, and those would otherwise be scored
    on the keyword axis alone while their neighbours carry a cosine.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()
    if not chunks:
        return {}
    index = _open(settings)
    if index is None:
        return {}
    ids = [chunk_vector_id(chunk) for chunk in chunks]
    try:
        response = index.fetch(ids=ids)
        vectors = getattr(response, "vectors", None) or response.get("vectors", {})
    except Exception:  # noqa: BLE001
        logger.exception("wiki_vector_fetch_failed")
        return {}

    found: dict[str, tuple[float, ...]] = {}
    for vector_id, record in vectors.items():
        values = record.get("values") if isinstance(record, dict) else record.values
        if values:
            found[str(vector_id)] = tuple(float(v) for v in values)
    return found


def reset_vector_index_cache() -> None:
    _index.cache_clear()


__all__ = [
    "chunk_vector_id",
    "fetch_chunk_vectors",
    "query_semantic_candidates",
    "reset_vector_index_cache",
]
