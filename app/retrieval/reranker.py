"""Profile-aware hybrid reranking for wiki chunks -- ported from UniPilot.

The scoring is carried across unchanged, including the two corrections that
were expensive to find:

**BM25 is normalized before it is blended.** Raw BM25 is unbounded (commonly
15-30 for a chunk with real keyword overlap, growing with length and repeated
terms) while cosine is bounded [0, 1]. Blending them directly meant a long,
keyword-dense but semantically wrong chunk (raw 28.13, cosine 0.541) outranked
the best semantic match in the corpus (raw 16.36, cosine 0.761) under a profile
weighted 60% semantic -- the EFFECTIVE weighting was nearer 4/96.

**A missing vector scores 0.0, never a lexical stand-in.** Falling back to token
overlap mixed two incomparable scales in one ranking: measured over 114
candidates, real cosine ran 0.257-0.747 while the stand-in ran 0.200-0.900 and
did not correlate. Worse, because the stand-in is itself keyword-derived, a
Pinecone outage silently double-counted keywords under a profile that believed
it was weighting 60/40. Scoring 0.0 keeps every chunk on one axis, and when no
candidate has a vector the ranking collapses to pure keyword order -- monotone
in the keyword score, so genuinely BM25-only rather than subtly wrong.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Iterable

from app.retrieval.profiles import RerankBoosts, RetrievalProfile, get_rerank_boosts
from app.retrieval.wiki_chunk import WikiChunk

if TYPE_CHECKING:
    from app.retrieval.corpus_index import CorpusIndex

_TOKEN = re.compile(r"[\w֐-׿]+", re.UNICODE)
_WIKI_LINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]]+)?\]\]")

# The raw `bm25_score` that maps to 0.5 after `_normalize_bm25`'s saturating
# transform -- chosen from the observed range for genuinely-matching chunks in
# this corpus (commonly 15-30), not derived analytically.
_BM25_SATURATION_K = 10.0


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text or "") if len(token) > 1]


def embedding_text(chunk: WikiChunk) -> str:
    """Text handed to the embedding model for one chunk.

    Kept because it DEFINES what the stored vectors represent: the index was
    built from exactly this rendering, so it documents why `aliases` and `tags`
    belong in the artifact even though nothing reads them at query time.
    """
    lines = [
        f"Page: {chunk.page_title}",
        f"Heading path: {' > '.join(chunk.heading_path)}",
    ]
    if chunk.aliases:
        lines.append(f"Also known as: {', '.join(chunk.aliases)}")
    if chunk.tags:
        lines.append(f"Tags: {', '.join(chunk.tags)}")
    if chunk.track:
        lines.append(f"Track: {chunk.track}")
    if chunk.faculty:
        lines.append(f"Faculty: {chunk.faculty}")
    if chunk.credits:
        lines.append(f"Credits: {chunk.credits}")
    if chunk.level:
        lines.append(f"Level: {chunk.level}")
    lines += ["", "Content:", chunk.content]
    return "\n".join(lines)


def bm25_score(
    chunk: WikiChunk, query_tokens: list[str], *, corpus: "CorpusIndex | None" = None
) -> float:
    """BM25 over the chunk, plus this corpus's own domain boosts.

    The section-title and course-number boosts stay ADDITIVE on top: they are
    domain signals about *where* a term matched, which BM25 has no notion of.
    Course-number matching is exact -- it was once `token in number`, so a query
    token "0044" scored against every course in the 0044xxxx faculty.
    """
    if not query_tokens:
        return 0.0

    from app.retrieval.corpus_index import build_chunk_stats
    from app.retrieval.vector_index import chunk_vector_id

    active_corpus = corpus if corpus is not None else _empty_corpus()
    stats = (
        active_corpus.stats_for(chunk_vector_id(chunk), chunk)
        if corpus is not None
        else build_chunk_stats(chunk)
    )
    score = active_corpus.bm25(stats, query_tokens)

    section_tokens = set(tokenize(chunk.section_title))
    for token in query_tokens:
        if token in section_tokens:
            score += 2.0
        if token in stats.course_numbers:
            score += 3.0
    return score


@lru_cache(maxsize=1)
def _empty_corpus() -> "CorpusIndex":
    from app.retrieval.corpus_index import CorpusIndex

    return CorpusIndex()


def _normalize_bm25(raw_score: float) -> float:
    """Saturating transform into [0, 1) -- see the module docstring."""
    if raw_score <= 0.0:
        return 0.0
    return raw_score / (raw_score + _BM25_SATURATION_K)


def semantic_similarity_score(*, semantic_override: float | None = None) -> float:
    """The embedding half: real cosine, or nothing. Never a stand-in."""
    if semantic_override is None:
        return 0.0
    return max(0.0, semantic_override)


def apply_metadata_boosts(
    chunk: WikiChunk,
    *,
    query_tokens: list[str],
    entities: dict[str, Any],
    user_context: dict[str, Any],
    boosts: RerankBoosts,
) -> float:
    adjustment = 0.0
    profile = user_context.get("profile") or {}
    course_number = str(entities.get("courseNumber") or "")
    if course_number and (
        chunk.primary_course_number == course_number
        or course_number in chunk.course_numbers_mentioned
    ):
        adjustment += boosts.exactCourseNumberBoost

    # Track relevance reads two real signals. `chunk.track` is set for track
    # pages (from their path), and `tags` carry membership for everything else --
    # a course tagged `required-dne` is required by the dne track. An earlier
    # version keyed on frontmatter present in ZERO files, so trackMatchBoost,
    # wrongTrackPenalty and the catalog-year boosts were all unreachable.
    track = str(profile.get("track") or "").lower()
    if track:
        chunk_tags = {tag.lower() for tag in (chunk.tags or ())}
        tagged_for_track = any(tag.endswith(f"-{track}") or tag == track for tag in chunk_tags)
        if chunk.track and track in chunk.track.lower():
            adjustment += boosts.trackMatchBoost
        elif tagged_for_track:
            adjustment += boosts.trackMatchBoost
        elif chunk.track:
            adjustment += boosts.wrongTrackPenalty

    target_semester = str(entities.get("targetSemesterCode") or "")
    if target_semester and target_semester not in chunk.content and "semester" in query_tokens:
        adjustment += boosts.wrongSemesterPenalty * 0.25

    topic = str(entities.get("topic") or "").strip()
    normalized_path = chunk.source_file.replace("\\", "/").lower()
    if topic and "courses/009-dds/" in normalized_path:
        adjustment += boosts.sourcePriorityBoost * 2.0
    if topic and (
        normalized_path.endswith("/log.md")
        or normalized_path == "log.md"
        or normalized_path.endswith("/index.md")
    ):
        adjustment -= boosts.sourcePriorityBoost * 4.0

    profile_name = str(user_context.get("profileName") or "")
    if profile_name in {"catalog_requirement_lookup", "requirement_explanation"}:
        if track and f"entities/tracks/{track}.md" in normalized_path:
            adjustment += boosts.sourcePriorityBoost * 6.0
        elif track and "entities/tracks/" in normalized_path:
            adjustment += boosts.wrongTrackPenalty
        faculty_slug = str(entities.get("faculty") or "").strip().lower()
        if faculty_slug and f"entities/faculties/{faculty_slug}.md" in normalized_path:
            adjustment += boosts.sourcePriorityBoost * 6.0
        if "faculty" in query_tokens and "entities/faculties/" in normalized_path:
            adjustment += boosts.sourcePriorityBoost * 4.0
        if "courses/" in normalized_path:
            adjustment -= boosts.sourcePriorityBoost * 3.0
        if "entities/programs/" in normalized_path:
            adjustment -= boosts.sourcePriorityBoost * 2.0

    return adjustment


def hybrid_score_chunk(
    chunk: WikiChunk,
    *,
    query: str,
    profile: RetrievalProfile,
    entities: dict[str, Any] | None = None,
    user_context: dict[str, Any] | None = None,
    boosts: RerankBoosts | None = None,
    semantic_override: float | None = None,
    corpus: "CorpusIndex | None" = None,
) -> float:
    query_tokens = tokenize(query)
    keyword = _normalize_bm25(bm25_score(chunk, query_tokens, corpus=corpus))
    semantic = semantic_similarity_score(semantic_override=semantic_override)
    score = (
        profile.hybrid_keyword_weight_normalized * keyword
        + profile.hybrid_vector_weight_normalized * semantic
    )
    score += apply_metadata_boosts(
        chunk,
        query_tokens=query_tokens,
        entities=entities or {},
        user_context=user_context or {},
        boosts=boosts or get_rerank_boosts(),
    )
    return score


def rerank_chunks(
    chunks: Iterable[WikiChunk],
    *,
    query: str,
    corpus: "CorpusIndex",
    limit: int = 5,
    profile: RetrievalProfile | None = None,
    entities: dict[str, Any] | None = None,
    user_context: dict[str, Any] | None = None,
    semantic_scores: dict[str, float] | None = None,
    settings: Any | None = None,
) -> list[tuple[WikiChunk, float]]:
    """Rank `chunks` against `query` on BM25 plus embedding similarity.

    `semantic_scores` are cosine scores the caller already holds, keyed by
    `chunk_vector_id` -- whatever `query_semantic_candidates` returned. Only the
    candidates ABSENT from it are fetched, which roughly halves a payload that
    was the single biggest cost in the pipeline (2105ms for 114 ids). Verified
    safe to mix: Pinecone's cosine and a locally recomputed one agree to within
    4e-4, which is float32 storage precision.

    `corpus` is required rather than looked up, because IDF must come from the
    WHOLE corpus: over ~100 candidates a corpus-common term would look rare and
    get weighted up.
    """
    from app.retrieval.embedding_service import cosine_similarity, embed_query
    from app.retrieval.profiles import get_profile
    from app.retrieval.vector_index import chunk_vector_id, fetch_chunk_vectors

    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    active_profile = profile or get_profile("fallback_academic_search")
    chunk_list = list(chunks)
    semantic_by_chunk_id: dict[int, float] = {}

    # At most ONE Pinecone round trip, never a call per chunk. There is no
    # "embed every candidate" fallback: if Pinecone is unreachable nothing is
    # scored, every chunk gets semantic 0.0, and the ranking collapses to pure
    # keyword order. Embedding ~60 candidate documents per call instead would
    # turn an outage into a latency and cost cliff.
    if settings.vector_index_enabled() and chunk_list:
        known = dict(semantic_scores or {})
        missing = [c for c in chunk_list if chunk_vector_id(c) not in known]
        if missing:
            query_vector = embed_query(query, settings=settings)
            if query_vector:
                vectors = fetch_chunk_vectors(missing, settings=settings)
                for chunk in missing:
                    stored = vectors.get(chunk_vector_id(chunk))
                    if stored:
                        known[chunk_vector_id(chunk)] = cosine_similarity(
                            query_vector, list(stored)
                        )
        for chunk in chunk_list:
            score = known.get(chunk_vector_id(chunk))
            if score is not None:
                semantic_by_chunk_id[id(chunk)] = score

    scored = [
        (
            chunk,
            hybrid_score_chunk(
                chunk,
                query=query,
                profile=active_profile,
                entities=entities,
                user_context=user_context,
                semantic_override=semantic_by_chunk_id.get(id(chunk)),
                corpus=corpus,
            ),
        )
        for chunk in chunk_list
    ]
    scored = [(chunk, score) for chunk, score in scored if score > 0]
    # Sorted on (score, then the content-addressed id). The id tie-break is
    # arbitrary but -- unlike corpus order, which is alphabetical file path --
    # it is not correlated with the filename, so it cannot systematically favour
    # `track-aerospace-*` over `track-zoology-*` at the cutoff.
    scored.sort(key=lambda item: (-item[1], chunk_vector_id(item[0])))
    return scored[: max(1, limit)]


def extract_wiki_links(chunk: WikiChunk) -> list[str]:
    links: list[str] = []
    for match in _WIKI_LINK.finditer(chunk.content):
        label = match.group(1).strip()
        if label:
            links.append(label)
    return links


def expand_linked_chunks(
    ranked: list[tuple[WikiChunk, float]],
    *,
    all_chunks: list[WikiChunk],
    depth: int,
    max_linked: int,
    query: str,
    profile: RetrievalProfile,
    corpus: "CorpusIndex",
    entities: dict[str, Any] | None = None,
    user_context: dict[str, Any] | None = None,
) -> list[tuple[WikiChunk, float]]:
    """Pull in pages the top hits LINK TO -- the corpus is a wiki, and a page's
    outbound links are a curated relevance signal no scorer can reconstruct."""
    if depth <= 0 or max_linked <= 0 or not ranked:
        return ranked

    selected_ids = {id(chunk) for chunk, _ in ranked}
    by_title: dict[str, list[WikiChunk]] = {}
    for chunk in all_chunks:
        by_title.setdefault(chunk.page_title.lower(), []).append(chunk)

    linked: list[tuple[WikiChunk, float]] = []
    boosts = get_rerank_boosts()
    for chunk, _score in ranked:
        for link_label in extract_wiki_links(chunk):
            for candidate in by_title.get(link_label.lower(), []):
                if id(candidate) in selected_ids:
                    continue
                link_score = (
                    hybrid_score_chunk(
                        candidate,
                        query=query,
                        profile=profile,
                        entities=entities,
                        user_context=user_context,
                        boosts=boosts,
                        corpus=corpus,
                    )
                    + boosts.linkRelevanceBoost
                )
                if link_score <= 0:
                    continue
                linked.append((candidate, link_score))
                selected_ids.add(id(candidate))
                if len(linked) >= max_linked:
                    break
            if len(linked) >= max_linked:
                break

    merged = list(ranked)
    merged.extend(sorted(linked, key=lambda pair: pair[1], reverse=True))
    merged.sort(key=lambda pair: pair[1], reverse=True)
    return merged[: profile.finalTopN]


__all__ = [
    "apply_metadata_boosts",
    "bm25_score",
    "embedding_text",
    "expand_linked_chunks",
    "extract_wiki_links",
    "hybrid_score_chunk",
    "rerank_chunks",
    "semantic_similarity_score",
    "tokenize",
]
