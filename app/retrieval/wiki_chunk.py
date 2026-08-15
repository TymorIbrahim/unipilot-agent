"""One heading-segmented section of a wiki page.

Ported from UniPilot's `obsidian_wiki_indexer.WikiChunk`, minus the chunker
itself. Nothing here SPLITS a page: that happens once at seed time, against the
real markdown, and the result is serialised into `data/wiki_corpus.json.gz`. The
deployed agent only ever REHYDRATES chunks, so it needs the shape and not the
parser.

The field list is deliberately complete rather than trimmed to what today's
scorer reads. `reranker.apply_metadata_boosts` keys on `tags`, `track`,
`faculty`, `primary_course_number` and `course_numbers_mentioned`, and
`embedding_text` also reads `aliases`, `credits`, `level` and `heading_path` --
a chunk missing any of them scores differently from the one the Pinecone index
was built against, silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WikiChunk:
    source_file: str
    page_title: str
    section_title: str
    heading_path: tuple[str, ...]
    content: str
    catalog_year: str | None = None
    faculty: str | None = None
    degree_program: str | None = None
    track: str | None = None
    course_numbers_mentioned: tuple[str, ...] = ()
    primary_course_number: str | None = None
    language: str = "he"
    # Frontmatter present on ~2,750 pages. `aliases` matters most: they are the
    # Hebrew/English name variants students actually type ("discrete math",
    # "מתמטיקה דיסקרטית"), and they join the TITLE tokens when scoring, so a
    # query using one scores like a title hit rather than not matching at all.
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    credits: str | None = None
    level: str | None = None

    @property
    def slug(self) -> str:
        """The page's slug -- its filename stem.

        `Passage.slug` is what the model names when it asks to `interpret` a
        page, so this has to agree with how `search_corpus` reported the hit.
        """
        name = self.source_file.replace("\\", "/").rsplit("/", 1)[-1]
        return name[:-3] if name.endswith(".md") else name


def chunk_from_payload(payload: Mapping[str, Any]) -> WikiChunk:
    """Rebuild a chunk from its serialised form in the corpus artifact."""
    return WikiChunk(
        source_file=str(payload.get("sourceFile") or ""),
        page_title=str(payload.get("pageTitle") or ""),
        section_title=str(payload.get("sectionTitle") or ""),
        heading_path=tuple(payload.get("headingPath") or ()),
        content=str(payload.get("content") or ""),
        catalog_year=payload.get("catalogYear"),
        faculty=payload.get("faculty"),
        degree_program=payload.get("degreeProgram"),
        track=payload.get("track"),
        course_numbers_mentioned=tuple(payload.get("courseNumbers") or ()),
        primary_course_number=payload.get("primaryCourseNumber"),
        language=str(payload.get("language") or "he"),
        aliases=tuple(payload.get("aliases") or ()),
        tags=tuple(payload.get("tags") or ()),
        credits=payload.get("credits"),
        level=payload.get("level"),
    )


def chunk_to_payload(chunk: Any) -> dict[str, Any]:
    """Serialise a chunk for the artifact. Mirrors `chunk_from_payload` exactly.

    Takes `Any` because the seed builds these with UniPilot's own chunker class,
    loaded by file path -- a different class object with the same fields.
    """
    return {
        "sourceFile": chunk.source_file,
        "pageTitle": chunk.page_title,
        "sectionTitle": chunk.section_title,
        "headingPath": list(chunk.heading_path or ()),
        "content": chunk.content,
        "catalogYear": chunk.catalog_year,
        "faculty": chunk.faculty,
        "degreeProgram": chunk.degree_program,
        "track": chunk.track,
        "courseNumbers": [str(n) for n in (chunk.course_numbers_mentioned or ())],
        "primaryCourseNumber": chunk.primary_course_number,
        "language": chunk.language,
        "aliases": list(chunk.aliases or ()),
        "tags": list(chunk.tags or ()),
        "credits": chunk.credits,
        "level": chunk.level,
    }


__all__ = ["WikiChunk", "chunk_from_payload", "chunk_to_payload"]
