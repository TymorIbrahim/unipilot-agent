"""Precomputed per-chunk tokens and corpus-wide statistics.

Ported from UniPilot, with one change: the corpus is loaded from
`data/wiki_corpus.json.gz` rather than by walking 2,753 markdown files. The
statistics themselves are computed by the SAME `build_chunk_stats` at seed time,
so there is one implementation of the tokenization and the runtime just reads
its output.

Why it exists at all, unchanged from the original reasoning:

* the candidate pre-filter tokenized every chunk on every call (227ms) to answer
  a question whose inputs never change between queries.
* real BM25 needs document frequencies and a mean document length, which are
  properties of the CORPUS and cannot be derived from a candidate subset -- IDF
  computed over 114 candidates would rate a corpus-common term as rare.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.retrieval.wiki_chunk import WikiChunk

# Standard BM25 constants. k1 controls how fast term frequency saturates; b
# controls how strongly document length is normalized (0 = not at all).
_BM25_K1 = 1.2
_BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """Defined here and imported by the reranker, so there is exactly one.

    A second tokenizer would produce document frequencies describing a corpus
    that no query is ever tokenized into: plausible BM25 numbers, wrong ranking.
    """
    from app.retrieval.reranker import tokenize as _tokenize

    return _tokenize(text)


@dataclass(frozen=True)
class ChunkStats:
    """Everything scoring needs about one chunk, tokenized once."""

    title_tokens: frozenset[str] = field(default_factory=frozenset)
    body_tokens: frozenset[str] = field(default_factory=frozenset)
    course_numbers: frozenset[str] = field(default_factory=frozenset)
    term_frequencies: Mapping[str, int] = field(default_factory=dict)
    length: int = 0


_EMPTY_STATS = ChunkStats()


def build_chunk_stats(chunk: WikiChunk) -> ChunkStats:
    # Aliases join the TITLE tokens rather than the body: they name the thing,
    # so a query using one should score like a title hit. This is what lets
    # "discrete math" or "מתמטיקה דיסקרטית" reach the keyword path at all --
    # previously only the slug registry ever saw them.
    title_tokens = tokenize(
        " ".join([chunk.page_title, chunk.section_title, *(chunk.aliases or ())])
    )
    body_tokens = tokenize(" ".join([chunk.content, *(chunk.tags or ())]))
    course_numbers = {str(number) for number in (chunk.course_numbers_mentioned or ())}
    all_tokens = [*title_tokens, *body_tokens, *course_numbers]
    return ChunkStats(
        title_tokens=frozenset(title_tokens),
        body_tokens=frozenset(body_tokens),
        course_numbers=frozenset(course_numbers),
        term_frequencies=Counter(all_tokens),
        length=len(all_tokens),
    )


@dataclass(frozen=True)
class CorpusIndex:
    """Corpus-wide term statistics plus per-chunk tokens, keyed by vector id."""

    stats_by_vector_id: Mapping[str, ChunkStats] = field(default_factory=dict)
    document_frequency: Mapping[str, int] = field(default_factory=dict)
    document_count: int = 0
    average_length: float = 0.0

    def stats_for(self, vector_id: str, chunk: WikiChunk | None = None) -> ChunkStats:
        """Precomputed stats, tokenizing on demand for chunks outside the corpus."""
        stats = self.stats_by_vector_id.get(vector_id)
        if stats is not None:
            return stats
        if chunk is not None:
            return build_chunk_stats(chunk)
        return _EMPTY_STATS

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones IDF, floored at 0 so a term present in nearly
        every document cannot contribute negatively.

        Returns a neutral 1.0 when there are no corpus statistics, so an empty
        index degrades to plain term-frequency scoring rather than zeroing every
        score.
        """
        if self.document_count <= 0:
            return 1.0
        df = self.document_frequency.get(term, 0)
        return max(0.0, math.log(1.0 + (self.document_count - df + 0.5) / (df + 0.5)))

    def bm25(self, stats: ChunkStats, query_tokens: Sequence[str]) -> float:
        """Textbook BM25 over one chunk."""
        if not query_tokens or stats.length <= 0:
            return 0.0
        avg_length = self.average_length or float(stats.length)
        score = 0.0
        for term in query_tokens:
            tf = stats.term_frequencies.get(term, 0)
            if not tf:
                continue
            denominator = tf + _BM25_K1 * (
                1.0 - _BM25_B + _BM25_B * (stats.length / avg_length)
            )
            score += self.idf(term) * (tf * (_BM25_K1 + 1.0)) / denominator
        return score


_EMPTY_CORPUS = CorpusIndex()


def build_corpus_index(chunks: Sequence[WikiChunk]) -> CorpusIndex:
    """Corpus statistics computed from chunks. Used by the SEED.

    The runtime does not call this -- it reads the result out of the artifact --
    but it is the definition of what those numbers mean, so it lives beside the
    scorer that consumes them rather than in the seeding script.
    """
    from app.retrieval.vector_index import chunk_vector_id

    stats_by_vector_id: dict[str, ChunkStats] = {}
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for chunk in chunks:
        stats = build_chunk_stats(chunk)
        stats_by_vector_id[chunk_vector_id(chunk)] = stats
        document_frequency.update(set(stats.term_frequencies))
        total_length += stats.length
    count = len(stats_by_vector_id)
    return CorpusIndex(
        stats_by_vector_id=stats_by_vector_id,
        document_frequency=dict(document_frequency),
        document_count=count,
        average_length=(total_length / count) if count else 0.0,
    )


__all__ = [
    "ChunkStats",
    "CorpusIndex",
    "build_chunk_stats",
    "build_corpus_index",
    "tokenize",
]
