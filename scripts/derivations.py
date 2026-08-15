"""Seed-time derivations -- the work that used to happen on every cold start.

Three things the agent once computed at runtime by holding a networkx graph and
2,753 markdown files in memory. All three are pure functions of data that never
changes between requests, so they are computed ONCE here and materialised: two
into Supabase tables, one into a committed artifact.

That is what lets `networkx` (16MB) and the wiki corpus (12MB) leave the
deployment entirely. Nothing in `app/` imports this module -- it runs on a
developer's machine, against a UniPilot checkout, and only its OUTPUT ships.

**Ported, not reinvented.** The prerequisite grammar and the chunker are
live-validated code whose failure modes were expensive to find, so they are
lifted with their behaviour intact rather than rewritten from the docs. The
chunker is loaded straight out of the UniPilot checkout BY FILE PATH: it is
stdlib-only and has no intra-package imports, and loading it by path sidesteps
the fact that both repos have a top-level package called `app`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Prerequisites: Hebrew catalog prose -> an AST -> edges with alternative groups
# ---------------------------------------------------------------------------
# Ported verbatim from UniPilot's `academic_graph_engine.parse_prerequisites_string`.
# The grammar is OR > AND > COURSE, spelled in Hebrew: `או` is or, `ו-` is and.

COURSE_CODE_RE = re.compile(r"\d{8}")

_NO_PREREQUISITES = {"none", "none listed", "אין"}


def parse_prerequisites_string(prereq_string: str) -> dict[str, Any]:
    """Parse a Hebrew prerequisite string into an AST (OR > AND > COURSE)."""
    text = (prereq_string or "").strip()
    if not text or text.lower() in _NO_PREREQUISITES:
        return {"type": "AND", "operands": []}

    tokens = _tokenize_prerequisites(text)
    if not tokens:
        return {"type": "AND", "operands": []}

    ast, position = _parse_or(tokens, 0)
    if position != len(tokens):
        raise ValueError(f"Unexpected tokens after position {position}: {tokens[position:]}")
    return ast


def _tokenize_prerequisites(text: str) -> list[str | tuple[str, str]]:
    tokens: list[str | tuple[str, str]] = []
    index = 0
    length = len(text)

    while index < length:
        if text[index].isspace():
            index += 1
            continue
        if text[index] in "()":
            tokens.append(text[index])
            index += 1
            continue
        if text.startswith("או", index) and (index + 2 >= length or not text[index + 2].isalnum()):
            tokens.append("OR")
            index += 2
            continue
        if text.startswith("ו-", index):
            tokens.append("AND")
            index += 2
            continue

        match = COURSE_CODE_RE.match(text, index)
        if match:
            tokens.append(("COURSE", match.group(0)))
            index = match.end()
            continue

        # Anything else is catalog noise -- prose, punctuation, a stray word.
        # Skipped rather than rejected: refusing the whole string would lose the
        # codes that ARE parseable alongside it.
        index += 1

    return _strip_empty_parens(tokens)


def _strip_empty_parens(tokens: list[str | tuple[str, str]]) -> list[str | tuple[str, str]]:
    """Drop `()` groups left behind by malformed catalog strings."""
    cleaned: list[str | tuple[str, str]] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == "(" and index + 1 < len(tokens) and tokens[index + 1] == ")":
            index += 2
            continue
        cleaned.append(tokens[index])
        index += 1
    return cleaned


def _parse_or(tokens: Sequence[Any], position: int) -> tuple[dict[str, Any], int]:
    left, position = _parse_and(tokens, position)
    operands = [left]
    while position < len(tokens) and tokens[position] == "OR":
        position += 1
        right, position = _parse_and(tokens, position)
        operands.append(right)
    if len(operands) == 1:
        return operands[0], position
    return {"type": "OR", "operands": operands}, position


def _parse_and(tokens: Sequence[Any], position: int) -> tuple[dict[str, Any], int]:
    left, position = _parse_primary(tokens, position)
    operands = [left]
    while position < len(tokens) and tokens[position] == "AND":
        position += 1
        right, position = _parse_primary(tokens, position)
        operands.append(right)
    if len(operands) == 1:
        return operands[0], position
    return {"type": "AND", "operands": operands}, position


def _parse_primary(tokens: Sequence[Any], position: int) -> tuple[dict[str, Any], int]:
    if position >= len(tokens):
        raise ValueError("Unexpected end of prerequisite expression")

    token = tokens[position]
    if token == "(":
        expression, position = _parse_or(tokens, position + 1)
        if position >= len(tokens) or tokens[position] != ")":
            raise ValueError("Missing closing parenthesis")
        return expression, position + 1
    if isinstance(token, tuple) and token[0] == "COURSE":
        return {"type": "COURSE", "id": token[1]}, position + 1
    raise ValueError(f"Unexpected token at {position}: {token!r}")


def prerequisite_edge_rows(
    catalog: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    """`prerequisite_edges` rows for a catalog, plus the codes that would not parse.

    Delegates the AND/OR walk to `wiring._walk_prerequisites` -- the SAME
    function the runtime `DerivedSchema` used -- so the materialised table and
    the code it replaces cannot disagree about what a group means. Edges sharing
    a group are alternatives; edges in different groups are each mandatory.
    """
    from app.agent_core.facts.wiring import _walk_prerequisites

    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    unparseable: list[str] = []

    for course in catalog:
        code = str(course.get("courseNumber") or "").strip()
        raw = str(course.get("prerequisitesText") or "").strip()
        if not code or not raw:
            continue
        try:
            ast = parse_prerequisites_string(raw)
        except ValueError:
            # `build_graph` swallowed these into an empty AST. Collected instead,
            # so a catalog that starts failing to parse is visible rather than
            # silently producing a course with no prerequisites.
            unparseable.append(code)
            continue

        for group, target in _walk_prerequisites(ast, path=code):
            edge = f"{code}->{target}"
            if (edge, group) in seen:
                # `A או A` in one group. The primary key is (edge, group), so a
                # duplicate would abort the whole load.
                continue
            seen.add((edge, group))
            rows.append({"edge": edge, "course": code, "requires": target, "group": group})

    return rows, unparseable


# ---------------------------------------------------------------------------
# Curriculum membership: which courses a track's wiki page links to
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
FRONTMATTER_COURSE_CODE_RE = re.compile(r'^course_code:\s*"?(\d{8})"?', re.MULTILINE)


@dataclass(frozen=True)
class WikiPage:
    slug: str
    relative_path: str
    content: str
    kind: str
    course_code: str | None


def classify_page(relative_path: str) -> str:
    if relative_path.startswith("courses/"):
        return "course"
    if relative_path.startswith("entities/tracks/"):
        return "track"
    if relative_path.startswith("entities/faculty"):
        return "faculty"
    return "wiki"


def extract_course_code(content: str, slug: str) -> str | None:
    match = FRONTMATTER_COURSE_CODE_RE.search(content)
    if match:
        return match.group(1)
    slug_match = COURSE_CODE_RE.match(slug)
    return slug_match.group(0) if slug_match else None


def load_wiki_pages(wiki_root: Path) -> dict[str, WikiPage]:
    """Every markdown page in the corpus, keyed by slug (the filename stem)."""
    pages: dict[str, WikiPage] = {}
    for markdown in sorted(wiki_root.rglob("*.md")):
        content = markdown.read_text(encoding="utf-8")
        relative = markdown.relative_to(wiki_root).as_posix()
        slug = markdown.stem
        pages[slug] = WikiPage(
            slug=slug,
            relative_path=relative,
            content=content,
            kind=classify_page(relative),
            course_code=extract_course_code(content, slug),
        )
    return pages


def track_course_rows(pages: Mapping[str, WikiPage]) -> list[dict[str, str]]:
    """`track_courses` rows -- the `contains` edges of the knowledge graph.

    A track page linking to a course page IS the membership record: that is what
    `build_graph` turned into `relation="contains"`, and 'which courses belong to
    my degree' has always been this edge set filtered by `programSlug`.

    Membership ONLY. The required/elective split lives in the credit-breakdown
    table on the track's page and is reached with search_corpus + interpret --
    the edge records the link, not the section it sat under.
    """
    slug_to_course = {slug: page.course_code for slug, page in pages.items() if page.course_code}

    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for slug, page in pages.items():
        if page.kind != "track":
            continue
        for target_slug, _display in WIKILINK_RE.findall(page.content):
            target = pages.get(target_slug.strip())
            if target is None:
                continue
            course = target.course_code or slug_to_course.get(target_slug.strip())
            if not course:
                continue
            edge = f"{slug}->{course}"
            # The source was a DiGraph, which collapses a repeated edge. A track
            # page that links the same course twice must not become two rows.
            if edge in seen:
                continue
            seen.add(edge)
            rows.append({"edge": edge, "track": slug, "course": course})
    return rows


# ---------------------------------------------------------------------------
# The wiki corpus artifact: chunks + corpus-wide BM25 statistics
# ---------------------------------------------------------------------------
# `search_corpus` needs two things the 2,753 markdown files used to supply on
# every cold start: the heading-segmented chunks themselves, and the corpus-wide
# document frequencies BM25 needs. Neither changes between requests, and IDF
# computed over a candidate subset would rate a corpus-common term as rare, so
# the statistics have to be corpus-wide or they are wrong.

_TOKEN = re.compile(r"[\w֐-׿]+", re.UNICODE)

BM25_K1 = 1.2
BM25_B = 0.75


def tokenize(text: str) -> list[str]:
    """UniPilot's `reranker.tokenize`, verbatim: tokens of 2+ characters, lowered.

    Must stay identical to the runtime tokenizer. A different tokenizer here
    would produce document frequencies describing a corpus that no query is ever
    tokenized against -- BM25 scores that look plausible and rank wrongly.
    """
    return [token.lower() for token in _TOKEN.findall(text or "") if len(token) > 1]


def load_chunker(unipilot_root: Path):
    """UniPilot's wiki chunker, imported BY PATH out of the checkout.

    Both repos have a top-level package named `app`, so a normal import would
    resolve to this one. `obsidian_wiki_indexer` is stdlib-only and imports
    nothing from its own package, which makes loading it by file path safe --
    and keeps the chunk boundaries identical to the ones the Pinecone index was
    built against. Re-chunking differently would change every
    `chunk_vector_id`, and the stored vectors would address chunks that no
    longer exist.
    """
    path = unipilot_root / "services" / "ai" / "app" / "retrieval" / "obsidian_wiki_indexer.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"UniPilot's wiki chunker is not at {path}. Pass --unipilot pointing at a checkout."
        )
    spec = importlib.util.spec_from_file_location("unipilot_wiki_indexer", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- defensive
        raise ImportError(f"could not load a module spec from {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution, not after. `WikiChunk` is a dataclass in a
    # module using `from __future__ import annotations`, and resolving its
    # annotations means looking the module up in `sys.modules` -- which fails
    # with a bare AttributeError if it is not there yet.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def chunk_vector_id(source_file: str, section_title: str, content: str) -> str:
    """The Pinecone id for a chunk -- content-addressed, exactly as UniPilot builds it.

    sha256 over `source_file|section_title|content`. It has to match to the byte,
    or a semantic hit cannot be mapped back to the chunk it names.
    """
    digest = hashlib.sha256(f"{source_file}|{section_title}|{content}".encode("utf-8"))
    return digest.hexdigest()


def build_corpus_artifact(unipilot_root: Path, wiki_root: Path) -> dict[str, Any]:
    """The whole retrieval corpus as one serialisable payload.

    Holds, per chunk: its Pinecone id, its slug, the titles, the text, and the
    precomputed term frequencies. Plus corpus-wide document frequencies and the
    mean document length, which BM25 cannot derive from a candidate subset.
    """
    from app.retrieval.corpus_index import build_chunk_stats
    from app.retrieval.wiki_chunk import chunk_to_payload

    indexer = load_chunker(unipilot_root)

    chunks: list[dict[str, Any]] = []
    document_frequencies: Counter[str] = Counter()
    total_length = 0

    # Deliberately mirrors `load_wiki_chunks` step for step -- sorted order,
    # the dotfile skip, and `str(...)` rather than `.as_posix()` for the relative
    # path. That path is hashed into `chunk_vector_id`, so a cosmetic difference
    # here would silently repoint every chunk away from its stored vector.
    for markdown in sorted(wiki_root.rglob("*.md")):
        if markdown.name.startswith("."):
            continue
        try:
            text = markdown.read_text(encoding="utf-8")
        except OSError:
            continue
        relative = str(markdown.relative_to(wiki_root))
        for chunk in indexer.chunk_wiki_page(relative_path=relative, text=text):
            # Statistics come from the RUNTIME's own `build_chunk_stats`, not a
            # second copy written here. One implementation of the tokenization
            # means the document frequencies always describe the corpus the
            # scorer actually tokenizes queries into.
            stats = build_chunk_stats(chunk)
            document_frequencies.update(set(stats.term_frequencies))
            total_length += stats.length

            chunks.append(
                {
                    # The full chunk, not a readable subset: the reranker's
                    # boosts key on `tags`, `track`, `faculty` and
                    # `primaryCourseNumber`, and a chunk missing any of them
                    # scores differently from the one the vectors were built
                    # against -- silently.
                    **chunk_to_payload(chunk),
                    "id": chunk_vector_id(chunk.source_file, chunk.section_title, chunk.content),
                    "slug": Path(chunk.source_file).stem,
                    "titleTokens": sorted(stats.title_tokens),
                    "bodyTokens": sorted(stats.body_tokens),
                    "termFrequencies": dict(stats.term_frequencies),
                    "length": stats.length,
                }
            )

    if not chunks:
        raise RuntimeError(f"no chunks produced from {wiki_root} -- is it the right directory?")

    return {
        "version": 1,
        "bm25": {"k1": BM25_K1, "b": BM25_B},
        "chunkCount": len(chunks),
        "averageLength": total_length / len(chunks),
        "documentFrequencies": dict(document_frequencies),
        "chunks": chunks,
    }


def idf(document_frequency: int, chunk_count: int) -> float:
    """BM25 inverse document frequency, matching UniPilot's `corpus_index`."""
    return math.log(1 + (chunk_count - document_frequency + 0.5) / (document_frequency + 0.5))


__all__ = [
    "BM25_B",
    "BM25_K1",
    "WikiPage",
    "build_corpus_artifact",
    "chunk_vector_id",
    "classify_page",
    "extract_course_code",
    "idf",
    "load_chunker",
    "load_wiki_pages",
    "parse_prerequisites_string",
    "prerequisite_edge_rows",
    "tokenize",
    "track_course_rows",
]
