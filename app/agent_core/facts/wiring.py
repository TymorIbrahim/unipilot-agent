"""Wiring -- connecting the tool layer to the systems that feed it.

Phase 11 built the primitives and the registry. This connects the ones whose
inputs live somewhere other than a Mongo collection, and it exists because of a
gap found by probing rather than by reading: the catalog advertised eight tools
and only three could actually be used. Two crashed on a `None` dependency, three
had no obtainable inputs, and nothing in the test suite noticed -- every one of
them was "correct" in isolation.

A capability is not available because it is implemented. It is available when
something can feed it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from app.agent_core.facts.find import DerivedSchema
from app.agent_core.facts.prose import Passage
from app.agent_core.facts.types import Basis, ScalarKind

logger = logging.getLogger(__name__)


class WikiRetriever:
    """Adapts the academic graph engine's hybrid wiki search to `search_corpus`.

    Returns `[]` rather than raising when the corpus is unavailable -- an empty
    hit list is a legitimate answer that the loop can act on, where an exception
    ends the turn with nothing learned.
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    async def search(self, query: str, limit: int) -> list[Passage]:
        try:
            hits = self._engine.search_wiki(query, limit=limit)
        except Exception:  # noqa: BLE001 -- an unavailable corpus is "no hits"
            return []
        return [
            Passage(
                slug=str(hit.get("slug", "")),
                title=str(hit.get("title") or hit.get("slug", "")),
                excerpt=str(hit.get("content", ""))[:2000],
                score=float(hit.get("score", 0.0) or 0.0),
            )
            for hit in hits
            if hit.get("slug")
        ]

    def page(self, slug: str) -> str | None:
        """The WHOLE wiki page for a slug, or None if it is not held.

        A prose tool (`extract_list`, `interpret`) needs the page a slug NAMES,
        not the fragments retrieval happened to surface. The wiki is
        heading-segmented, so one page returns as several small chunks and top-k
        search brings back only some -- a live plan extracted 2 of ~40 required
        codes because the rest never made the cut. The engine holds every page in
        full by slug, so hand that over and let the extractor read the complete
        section."""
        pages = getattr(self._engine, "wiki_pages", None)
        if not isinstance(pages, dict):
            return None
        page = pages.get(slug)
        content = page.get("content") if isinstance(page, dict) else None
        return content if isinstance(content, str) and content else None


_EXTRACT_PROMPT = """Read the passage and answer the question with ONE value.

Reply with JSON only: {"value": <the value>, "quote": "<the exact sentence it came from>"}

The value must APPEAR in the passage. Do not calculate, infer, or combine --
extraction only; arithmetic happens elsewhere where it can be audited. If the
passage does not contain such a value, reply {"value": null, "quote": ""}.

Expected kind: %s

PASSAGE (%s):
%s

QUESTION: %s"""


_EXTRACT_ALL_PROMPT = """Read the passage and list EVERY value that answers the question.

Reply with JSON only: {"items": [{"value": <a value>, "quote": "<the exact text it came from>"}, ...]}

Each value must APPEAR in the passage. Do not calculate, infer, or combine --
extraction only; list what the passage literally states, once per distinct value.
If the passage lists none, reply {"items": []}.

Expected kind of each value: %s

PASSAGE (%s):
%s

QUESTION: %s"""


def _stamped(chat: Any, module: Any) -> Any:
    """The same client, labelled with the module that is about to use it.

    `extract` and `extract_all` are two of the agent's three model call sites and
    they are DIFFERENT modules -- Interpreter and ListInterpreter -- so a single
    label on the client would file both under one name and misreport the trace.
    Stamping happens here, at the call site, because that is the only place that
    knows which of the two is running.

    A plain client has no `for_module`, and then this is a no-op: nothing about
    extraction depends on being traced.
    """
    stamp = getattr(chat, "for_module", None)
    return stamp(module) if callable(stamp) else chat


class ModelExtractor:
    """Adapts a chat model to `interpret`'s extraction contract.

    The prompt forbids calculation, and `interpret` independently verifies that
    the returned value appears in the cited quote -- so a model that computes
    anyway is caught rather than trusted. Instruction and enforcement, because
    the instruction alone is a request.
    """

    def __init__(self, chat: Any) -> None:
        self._chat = chat

    async def extract(self, passage: Passage, question: str, expect: Any) -> tuple[Any, str]:
        from app.agent_core.facts.adapter import extract_reply

        prompt = _EXTRACT_PROMPT % (
            getattr(expect, "value", expect),
            passage.slug,
            passage.excerpt,
            question,
        )
        from app.tracing.modules import INTERPRETER

        reply = await _stamped(self._chat, INTERPRETER).ainvoke([{"role": "user", "content": prompt}])
        payload = extract_reply(getattr(reply, "content", reply))
        if not isinstance(payload, dict) or "value" not in payload:
            # `extract_reply` only keeps loop-shaped replies, so parse the raw
            # content here rather than teaching it a second schema.
            payload = _loose_json(getattr(reply, "content", reply))
        return payload.get("value"), str(payload.get("quote") or "")

    async def extract_all(
        self, passage: Passage, question: str, expect: Any
    ) -> list[tuple[Any, str]]:
        """The plural of `extract`: every listed value, each with its own quote.

        `interpret_list` verifies each returned value against its quote, so a
        model that pads the list with invented codes is caught per element rather
        than trusted -- the same enforcement `extract` leans on, applied to a set.
        """
        prompt = _EXTRACT_ALL_PROMPT % (
            getattr(expect, "value", expect),
            passage.slug,
            passage.excerpt,
            question,
        )
        from app.tracing.modules import LIST_INTERPRETER

        reply = await _stamped(self._chat, LIST_INTERPRETER).ainvoke(
            [{"role": "user", "content": prompt}]
        )
        payload = _loose_json(getattr(reply, "content", reply))
        items = payload.get("items")
        if not isinstance(items, list):
            return []
        pairs: list[tuple[Any, str]] = []
        for item in items:
            if isinstance(item, dict) and "value" in item:
                pairs.append((item.get("value"), str(item.get("quote") or "")))
        return pairs


def _loose_json(content: Any) -> dict:
    import json
    import re

    text = str(content or "")
    for candidate in (text, *(m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S))):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start : end + 1])
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _edge_documents(engine: Any) -> list[dict[str, Any]]:
    """Every prerequisite edge in the graph, with its alternative group intact.

    `group` is what makes the result honest. The engine's own
    `prerequisite_course_ids` flattens the AND/OR tree into a plain set, so "A
    or B" comes back as two edges that both look mandatory -- and a caller
    counting missing prerequisites then double-counts a choice as two
    obligations. Walking the AST instead gives every OR-alternative its own
    group id, which `traverse` can carry through with `"carry": ["group"]`.

    Edges within one group are ALTERNATIVES (any one satisfies it); edges in
    different groups are all required.
    """
    documents: list[dict[str, Any]] = []
    for code in getattr(engine, "graph", {}).nodes:
        node = engine.graph.nodes[code]
        ast = node.get("prerequisites_ast")
        if not ast:
            continue
        for group, target in _walk_prerequisites(ast, path=str(code)):
            documents.append(
                {
                    "edge": f"{code}->{target}",
                    "course": str(code),
                    "requires": str(target),
                    "group": group,
                }
            )
    return documents


def _walk_prerequisites(ast: Mapping[str, Any], path: str) -> list[tuple[str, str]]:
    """(group, course) for every leaf, where a shared group means "either one"."""
    kind = ast.get("type")
    if kind == "COURSE":
        return [(path, ast["id"])]

    operands = ast.get("operands", [])
    if kind == "OR":
        # One group for the whole disjunction: all of its leaves are the same
        # single obligation, satisfiable by any one of them.
        return [leaf for operand in operands for leaf in _walk_prerequisites(operand, path)]
    # AND: each operand is its own obligation, so each gets its own group.
    return [
        leaf
        for position, operand in enumerate(operands)
        for leaf in _walk_prerequisites(operand, f"{path}.{position}")
    ]


def prerequisite_edges_source(engine: Any) -> DerivedSchema:
    """`find`'s view of the prerequisite graph.

    A source, not a tool. The model reaches it exactly like any collection --
    `{"tool": "find", "args": {"source": "prerequisite_edges", ...}}` -- which is
    the whole point: `traverse` was unusable not because it was unimplemented
    but because nothing the model could call produced edges.
    """
    return DerivedSchema(
        collection="prerequisite_edges",
        key="edge",
        fields={
            "edge": ScalarKind.IDENTIFIER,
            "course": ScalarKind.IDENTIFIER,
            "requires": ScalarKind.IDENTIFIER,
            "group": ScalarKind.IDENTIFIER,
        },
        # Parsed out of catalog prose, not read from a record. Weaker than an
        # official record, and the basis ordering carries that to any answer.
        basis=Basis.WIKI_DERIVED,
        # `course` and `requires` are course NUMBERS (00960211), not ObjectIds.
        # A live eval filtered `course` by a course's _id and got nothing --
        # `completed_courses` keys courses by ObjectId while this keys them by
        # number, and the model had no way to know the two differ.
        joins=(("course", "courses.courseNumber"), ("requires", "courses.courseNumber")),
        produce=lambda: _edge_documents(engine),
        yields=frozenset({"edges"}),
    )


def _curriculum_documents(engine: Any) -> list[dict[str, Any]]:
    """Every track->course membership edge the knowledge graph holds.

    The graph builds these from the wikilinks on a track's page: when the ISE
    track page links to a course, `build_graph` records a `contains` edge. So
    "which courses belong to my track" is already in the graph -- 2,944 edges of
    it -- and this exposes them as records the model can `find`.

    Membership only, NOT the required/elective classification: that split lives
    in the credit-breakdown table on the track's wiki PAGE (search_corpus +
    interpret), because the edge records the link, not the section it sat under.
    """
    documents: list[dict[str, Any]] = []
    graph = getattr(engine, "graph", None)
    if graph is None:
        return documents
    for source, target, data in graph.edges(data=True):
        if data.get("relation") != "contains":
            continue
        documents.append(
            {
                "edge": f"{source}->{target}",
                "track": str(source),
                "course": str(target),
            }
        )
    return documents


def curriculum_source(engine: Any) -> DerivedSchema:
    """`find`'s view of the track->course membership graph.

    The companion to `prerequisite_edges`, and the answer to "the knowledge is
    in the graph, why can't the agent reach it": the prerequisite edges were
    wired, the membership edges were not, so the model could ask what a course
    REQUIRES but not what a degree CONTAINS. Filter `track` by the student's
    `programSlug` to get their curriculum.
    """
    return DerivedSchema(
        collection="track_courses",
        key="edge",
        fields={
            "edge": ScalarKind.IDENTIFIER,
            "track": ScalarKind.IDENTIFIER,
            "course": ScalarKind.IDENTIFIER,
        },
        basis=Basis.WIKI_DERIVED,
        # `course` is a course NUMBER; join to the catalog for its credits/faculty.
        joins=(("course", "courses.courseNumber"),),
        produce=lambda: _curriculum_documents(engine),
    )


def build_wiring(settings: Any | None = None, audience: str | None = None) -> dict[str, Any]:
    """Retriever, extractor and derived sources for a `DispatchContext`.

    Returns only what could actually be built. A missing key means the catalog
    will not advertise the tool that needs it, which is the point: the model is
    never told about a capability nothing can serve.
    """
    wiring: dict[str, Any] = {"sources": {}}

    # The corpus comes from the precomputed artifact, not from a graph engine
    # holding 2,753 files. `prerequisite_edges` and `track_courses` are no longer
    # registered here either -- they are materialised tables in `REGISTRY` now.
    #
    # What used to be here was an import of `app.retrieval.graph_engine`, a
    # package this repo does not contain, inside a bare `except: pass`. It
    # silently succeeded at wiring NOTHING: no retriever, so `search_corpus`,
    # `interpret` and `extract_list` went unadvertised on every run, and no
    # derived sources, so `traverse` had no reachable input. A swallowed
    # ImportError is indistinguishable from a deliberate absence, which is
    # exactly why this one survived the port unnoticed.
    try:
        from app.retrieval.corpus import build_retriever

        retriever = build_retriever(settings=settings, audience=audience)
        if retriever is not None:
            wiring["retriever"] = retriever
    except Exception:  # noqa: BLE001 -- an unavailable corpus is a missing capability
        logger.exception("wiki corpus could not be wired; prose tools will not be advertised")

    try:
        from app.agent_core.reasoning.llm_client import build_chat_llm

        chat = build_chat_llm(settings=settings)
        if chat is not None:
            wiring["extractor"] = ModelExtractor(chat)
    except Exception:  # noqa: BLE001
        pass

    return wiring


def build_context(
    database: Any, settings: Any | None = None, audience: str | None = None, **overrides: Any
) -> Any:
    """The one place a fully wired `DispatchContext` is assembled.

    It did not exist before, and its absence was the actual defect behind two
    unusable tools: nothing in `app/` built a context, so every caller and every
    test assembled its own. The reachability test built one that seeded
    `context.facts["edges"]` by hand and passed -- proving a route no model
    could take, while the real one did not exist.

    `obtainable` is COMPUTED from what the registered sources declare they
    yield, never hand-listed, so adding or dropping a source moves the catalog
    with it and the prompt cannot drift from the wiring.
    """
    from app.agent_core.facts.dispatch import DispatchContext
    from app.agent_core.facts.sources import REGISTRY

    wiring = build_wiring(settings, audience)
    schemas = {**REGISTRY, **wiring.get("sources", {})}

    return DispatchContext(
        database=database,
        settings=settings,
        schemas=schemas,
        retriever=overrides.get("retriever", wiring.get("retriever")),
        extractor=overrides.get("extractor", wiring.get("extractor")),
        obtainable=obtainable_from(schemas),
    )


def obtainable_from(schemas: Mapping[str, Any]) -> frozenset[str]:
    """The tool input kinds these sources can supply.

    `slots` is the one that needs the extra sentence: no source yields slots
    directly, but a source declaring a nested array of them does, via `unnest`.
    That is a route the model can take with tools it already has, so the claim
    is honest -- and `test_reachability.py` walks it end to end rather than
    taking this function's word for it.
    """
    return frozenset(kind for schema in schemas.values() for kind in getattr(schema, "yields", ()))


__all__ = [
    "ModelExtractor",
    "WikiRetriever",
    "build_context",
    "build_wiring",
    "curriculum_source",
    "obtainable_from",
    "prerequisite_edges_source",
]
