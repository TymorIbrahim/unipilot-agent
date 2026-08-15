"""Wiki retrieval -- UniPilot's hybrid search, served from the corpus artifact.

This is `AcademicGraphEngine.search_wiki` ported whole. The only thing that
changed is where the chunks come from: an artifact instead of 2,753 markdown
files and a networkx graph, because a serverless invocation has no warm process
to hold them and re-reading the corpus per cold start would spend the request's
latency budget before the model was called once.

Everything else is the original pipeline, and the shape matters:

  1. TWO independent candidate pools, each capped, unioned before reranking.
     Reranking the whole corpus on every call would be a real latency
     regression, so both filters only narrow.
       - a cheap literal keyword-match count -- catches exact course codes,
         acronyms and rare terms that a semantic match can miss or under-rank
       - a full-corpus Pinecone query -- one bounded round trip, and the thing
         that catches verbose natural-language questions whose wording does not
         literally overlap the right chunk
  2. `rerank_chunks` over the union, blending normalized BM25 with real cosine
     at the profile's 40/60 split, plus the tuned metadata boosts.
  3. one hop of wikilink expansion, because a page's outbound links are a
     curated relevance signal no scorer reconstructs.

Cosine scores from step 1's Pinecone query are CARRIED INTO step 2 rather than
recomputed: the reranker only fetches vectors for the keyword-only remainder.

Dropping the semantic half is not a neutral simplification. It is 60% of the
ranking weight, and without it "graduation requirements for computer science"
ranks a philosophy course first -- the exact failure the vector pass exists to
prevent.
"""

from __future__ import annotations

import gzip
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.retrieval.corpus_index import ChunkStats, CorpusIndex
from app.retrieval.vector_index import chunk_vector_id
from app.retrieval.wiki_chunk import WikiChunk, chunk_from_payload

logger = logging.getLogger(__name__)

ARTIFACT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "wiki_corpus.json.gz"

# Bounds how many chunks reach the full reranker. Sized well above the profile's
# `finalTopN` so real candidates are not dropped, while keeping the rerank pass
# fast regardless of how many chunks a common query term loosely matches.
CANDIDATE_POOL_CAP = 60

_MAX_EXCERPT_CHARS = 2000
_MAX_PAGE_CHARS = 24000


def _match_score(stats: ChunkStats, tokens: list[str]) -> float:
    """Cheap weighted match count for the candidate pre-filter.

    Deliberately not a full BM25 pass -- just enough to rank and bound how many
    chunks reach the real reranker. Whole-token matching, because `token in
    haystack` meant "cs" matched "physics"; title and course-number hits outweigh
    body hits, which spreads a distribution that otherwise left ~108 chunks tied
    at the cutoff.
    """
    score = 0.0
    for token in tokens:
        if token in stats.course_numbers:
            score += 5.0
        elif token in stats.title_tokens:
            score += 3.0
        elif token in stats.body_tokens:
            score += 1.0
    return score


class WikiCorpus:
    """The artifact: chunks, their precomputed stats, and the corpus index."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.chunks: list[WikiChunk] = []
        self.by_vector_id: dict[str, WikiChunk] = {}
        self._by_slug: dict[str, list[int]] = {}
        stats_by_vector_id: dict[str, ChunkStats] = {}

        for index, raw in enumerate(payload["chunks"]):
            chunk = chunk_from_payload(raw)
            self.chunks.append(chunk)
            vector_id = str(raw.get("id") or chunk_vector_id(chunk))
            self.by_vector_id[vector_id] = chunk
            self._by_slug.setdefault(chunk.slug, []).append(index)
            stats_by_vector_id[vector_id] = ChunkStats(
                title_tokens=frozenset(raw.get("titleTokens") or ()),
                body_tokens=frozenset(raw.get("bodyTokens") or ()),
                course_numbers=frozenset(raw.get("courseNumbers") or ()),
                term_frequencies=raw.get("termFrequencies") or {},
                length=int(raw.get("length") or 0),
            )

        self.index = CorpusIndex(
            stats_by_vector_id=stats_by_vector_id,
            document_frequency=payload["documentFrequencies"],
            document_count=len(self.chunks),
            average_length=float(payload.get("averageLength") or 0.0),
        )
        self._stats = stats_by_vector_id
        # Inverted index over the precomputed term frequencies, so the cheap
        # pre-filter touches only chunks that contain a query term instead of
        # scoring all 4,895 on every call.
        self._postings: dict[str, list[str]] = {}
        for vector_id, stats in stats_by_vector_id.items():
            for term in stats.term_frequencies:
                self._postings.setdefault(term, []).append(vector_id)

    def __len__(self) -> int:
        return len(self.chunks)

    def keyword_candidates(self, tokens: list[str], limit: int) -> list[WikiChunk]:
        scores: dict[str, float] = {}
        for term in set(tokens):
            for vector_id in self._postings.get(term, ()):  # noqa: B007
                if vector_id not in scores:
                    scores[vector_id] = _match_score(self._stats[vector_id], tokens)
        ranked = sorted(
            ((vid, s) for vid, s in scores.items() if s > 0),
            key=lambda item: (-item[1], item[0]),
        )
        return [self.by_vector_id[vid] for vid, _ in ranked[:limit]]

    def page(self, slug: str) -> str | None:
        """Every chunk of one page, in document order, reassembled.

        `interpret` and `extract_list` need the page a slug NAMES, not the
        fragments that happened to reach top-k. The wiki is heading-segmented, so
        one page returns as several chunks and a live plan once extracted 2 of
        ~40 required course codes because the rest never made the cut.
        """
        indexes = self._by_slug.get(slug)
        if not indexes:
            return None
        parts: list[str] = []
        for index in indexes:
            chunk = self.chunks[index]
            parts.append(f"## {chunk.section_title}\n{chunk.content}" if chunk.section_title else chunk.content)
        return ("\n\n".join(parts).strip())[:_MAX_PAGE_CHARS] or None


@lru_cache(maxsize=1)
def get_corpus(path: str | None = None) -> WikiCorpus | None:
    """The corpus, loaded once per process.

    Returns None rather than raising when the artifact is absent: a missing
    corpus is a MISSING CAPABILITY, and the catalog's job is to stop advertising
    the three tools that need it. An exception here would take down every
    request, including the many that never touch prose.
    """
    artifact = Path(path) if path else ARTIFACT_PATH
    if not artifact.is_file():
        logger.warning(
            "wiki corpus artifact not found at %s -- search_corpus, interpret and "
            "extract_list will not be advertised. Run scripts/seed.py to build it.",
            artifact,
        )
        return None
    try:
        payload = json.loads(gzip.decompress(artifact.read_bytes()))
        corpus = WikiCorpus(payload)
    except Exception:  # noqa: BLE001 -- a corrupt artifact is a missing capability
        logger.exception("wiki corpus artifact could not be loaded from %s", artifact)
        return None
    logger.info("wiki corpus loaded: %d chunks from %s", len(corpus), artifact.name)
    return corpus


class CorpusRetriever:
    """The `Retriever` the fact layer expects: `search` and `page`."""

    def __init__(self, corpus: WikiCorpus, settings: Any | None = None) -> None:
        self._corpus = corpus
        self._settings = settings

    def _resolved_settings(self) -> Any:
        if self._settings is None:
            from app.config import get_settings

            self._settings = get_settings()
        return self._settings

    async def search(self, query: str, limit: int) -> list[Any]:
        from app.agent_core.facts.prose import Passage

        ranked = self.search_wiki(query, limit=limit)
        return [
            Passage(
                slug=chunk.slug,
                title=chunk.page_title or chunk.slug,
                excerpt=chunk.content[:_MAX_EXCERPT_CHARS],
                score=float(score),
            )
            for chunk, score in ranked
        ]

    def search_wiki(self, query: str, *, limit: int = 5) -> list[tuple[WikiChunk, float]]:
        """The hybrid pipeline. Never raises: an empty list is a legitimate
        answer the loop can act on, an exception ends the turn with nothing
        learned."""
        from app.retrieval.profiles import get_profile
        from app.retrieval.reranker import expand_linked_chunks, rerank_chunks, tokenize
        from app.retrieval.vector_index import query_semantic_candidates

        tokens = tokenize(query)
        if not tokens:
            return []

        settings = self._resolved_settings()
        profile = get_profile("fallback_academic_search")

        candidates: dict[str, WikiChunk] = {}
        for chunk in self._corpus.keyword_candidates(tokens, CANDIDATE_POOL_CAP):
            candidates[chunk_vector_id(chunk)] = chunk

        # Keep the cosine scores Pinecone already computed -- discarding them
        # made the reranker re-fetch these same vectors to recalculate an
        # identical number.
        semantic_scores: dict[str, float] = {}
        for chunk, score in query_semantic_candidates(
            query=query,
            chunks_by_id=self._corpus.by_vector_id,
            limit=CANDIDATE_POOL_CAP,
            settings=settings,
        ):
            vector_id = chunk_vector_id(chunk)
            candidates.setdefault(vector_id, chunk)
            semantic_scores[vector_id] = score

        if not candidates:
            return []

        ranked = rerank_chunks(
            list(candidates.values()),
            query=query,
            corpus=self._corpus.index,
            limit=max(limit, profile.wikiChunksFinal),
            profile=profile,
            semantic_scores=semantic_scores,
            settings=settings,
        )
        if profile.linkExpansionDepth > 0 and ranked:
            ranked = expand_linked_chunks(
                ranked,
                all_chunks=self._corpus.chunks,
                depth=profile.linkExpansionDepth,
                max_linked=profile.maxLinkedChunks,
                query=query,
                profile=profile,
                corpus=self._corpus.index,
            )
        return ranked[:limit]

    def page(self, slug: str) -> str | None:
        return self._corpus.page(slug)


def build_retriever(path: str | None = None, settings: Any | None = None) -> CorpusRetriever | None:
    corpus = get_corpus(path)
    return CorpusRetriever(corpus, settings) if corpus is not None else None


__all__ = [
    "ARTIFACT_PATH",
    "CANDIDATE_POOL_CAP",
    "CorpusRetriever",
    "WikiCorpus",
    "build_retriever",
    "get_corpus",
]
