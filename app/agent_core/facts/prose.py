"""The prose bridge -- phase 6 of docs/agent/tools_implementation_plan.md.

There is no algebra of text. So the prose side does not get its own parallel
vocabulary of operations -- it gets exactly two jobs, and both exist to hand
work over to the algebra as quickly as possible:

  `search_corpus`  find candidate passages          (relevance is not algebraic)
  `interpret`      turn ONE passage into a typed fact (text -> claim needs inference)

Kept as two steps rather than one on purpose. Retrieval returns candidates a
person can be shown; interpretation makes a claim about one of them. Merging
them would hide which passage an answer actually came from, which is the only
thing making a citation meaningful.

Ranking quality and the extraction model are injected. This module owns the TYPE
boundary: what comes out the far side must be a well-formed fact the algebra can
consume without re-parsing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Union

from app.agent_core.facts.operators import DataDefect
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)


@dataclass(frozen=True)
class Passage:
    slug: str
    title: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class Citation:
    """Where a claim came from. Mandatory -- an interpretation without one is
    indistinguishable from an assertion."""

    source: str
    quote: str
    title: str | None = None


@dataclass(frozen=True)
class Interpreted:
    value: Scalar
    citation: Citation
    basis: Basis = Basis.LLM_INTERPRETATION


@dataclass(frozen=True)
class InterpretedList:
    """The plural of `Interpreted`: the SET a passage enumerates, as a collection.

    A collection, not a bag of scalars, so the algebra applies unchanged -- the
    caller `select`s against it or uses its field for an `in` semi-join, exactly
    as with any `find` result.
    """

    value: Collection
    citation: Citation
    basis: Basis = Basis.LLM_INTERPRETATION


class Retriever(Protocol):
    async def search(self, query: str, limit: int) -> Any: ...


class Extractor(Protocol):
    async def extract(self, passage: Passage, question: str, expect: ScalarKind) -> Any: ...

    async def extract_all(
        self, passage: Passage, question: str, expect: ScalarKind
    ) -> Any: ...


def _appears_in(raw: Any, quote: str) -> bool:
    """Is the extracted value actually present in the text it cites?

    This is the only enforceable guard against interpretation COMPUTING rather
    than reading. Pattern-matching for arithmetic does not work: an unevaluated
    "155 - 62.5" already fails to parse as a quantity, so a regex for it catches
    nothing, while the failure that matters -- a model that quietly computes
    92.5 and returns a clean number -- looks identical to a real extraction.

    Requiring the value to appear in the cited text catches exactly that: a
    computed number is not in the passage it claims to come from. It also makes
    the citation load-bearing rather than decorative.
    """
    text = quote.replace(",", "")
    candidate = str(raw).strip().replace(",", "")
    if candidate in text:
        return True
    # "155.0" extracted from prose that says "155" is still grounded.
    try:
        number = float(candidate)
    except (TypeError, ValueError):
        return False
    # `%g` renders 155.0 as "155", which is how prose actually writes it.
    return f"{number:g}" in text


async def search_corpus(retriever: Retriever, query: str, limit: int = 5) -> Collection:
    """Candidate passages, as a Collection so the algebra applies to them.

    Search hits are records: they have a slug, a title, an excerpt and a score.
    Modelling them as anything else would force `sort`, `limit` and `select` to
    be reinvented for the prose side.

    Never complete. Retrieval is top-k by construction, so a hit list is almost
    never everything that matched -- claiming otherwise would let `count` over
    search results return a confident number that means nothing.
    """
    passages = await retriever.search(query, limit)
    records = tuple(
        Record(
            fields={
                "slug": Scalar(ScalarKind.IDENTIFIER, passage.slug),
                "title": Scalar(ScalarKind.TEXT, passage.title),
                "excerpt": Scalar(ScalarKind.TEXT, passage.excerpt),
                "score": Scalar(ScalarKind.QUANTITY, float(passage.score)),
            },
            basis=Basis.WIKI_DERIVED,
        )
        for passage in passages
    )
    return Collection(records=records, completeness=Completeness(complete=False, total=None))


async def interpret(
    extractor: Extractor,
    passage: Passage,
    question: str,
    expect: ScalarKind,
) -> Union[Interpreted, DataDefect]:
    """Read one value of a DECLARED kind out of one passage, with a citation.

    The kind is declared by the caller rather than chosen by the model. That
    removes a degree of freedom the model could get wrong, lets the claim be
    validated against the value, and means a passage with no value of that kind
    fails closed instead of producing a plausible-looking string.
    """
    raw, quote = await extractor.extract(passage, question, expect)

    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return _cannot_determine(passage, question, expect, why="it contains no such value")

    cited = quote or passage.excerpt
    if not _appears_in(raw, cited):
        return DataDefect(
            0,
            f"interpretation of '{passage.slug}' returned {raw!r}, which does not appear in the text "
            f"it cites ({cited!r}). Interpretation EXTRACTS; a value that is not in the passage was "
            "computed or inferred, and arithmetic belongs to the algebra where operands are grounded "
            "in refs and the result can be audited.",
        )

    value = _as_kind(raw, expect)
    if value is None:
        return _cannot_determine(
            passage, question, expect, why=f"the value found ({raw!r}) is not a {expect.value}"
        )

    return Interpreted(
        value=Scalar(expect, value),
        citation=Citation(source=passage.slug, quote=quote or passage.excerpt, title=passage.title),
    )


async def interpret_list(
    extractor: Extractor,
    passage: Passage,
    question: str,
    expect: ScalarKind,
) -> Union[InterpretedList, DataDefect]:
    """Read the SET of values of a declared kind that ONE passage lists.

    Where `interpret` turns a passage into one fact, this turns it into the
    collection of values it enumerates -- the courses a section lists, the codes
    in an elective group -- so the algebra can then classify against them (a
    remaining course is an elective iff its number is `in` this set). It exists
    because the alternative, one `interpret` per candidate, is dozens of model
    calls to answer one classification, and the model kept stalling there.

    Every value is held to the SAME bar as `interpret`: it must APPEAR in the
    text it cites, or it was invented rather than read, and is dropped. That is
    the whole grounding guarantee, applied per element -- a hallucinated course
    code cannot enter the set, so a classification built on the set stays honest.
    Duplicates collapse; a set with no verifiable member fails closed, naming the
    source, rather than returning an empty collection that reads as "none listed".
    """
    items = await extractor.extract_all(passage, question, expect)

    records: list[Record] = []
    seen: set = set()
    for raw, quote in items or ():
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        if not _appears_in(raw, quote or passage.excerpt):
            continue
        value = _as_kind(raw, expect)
        if value is None or value in seen:
            continue
        seen.add(value)
        records.append(Record(fields={"value": Scalar(expect, value)}, basis=Basis.LLM_INTERPRETATION))

    if not records:
        return _cannot_determine(passage, question, expect, why="it lists no such values")

    return InterpretedList(
        value=Collection(records=tuple(records), completeness=Completeness(complete=False, total=None)),
        citation=Citation(source=passage.slug, quote=passage.excerpt, title=passage.title),
    )


def _cannot_determine(passage: Passage, question: str, expect: ScalarKind, *, why: str) -> DataDefect:
    """A refusal that names the source and closes the door on re-reading it.

    Observed live: an agent got a bare `cannot_determine`, re-queried the same
    page, received the identical answer, and burned turns doing it. A refusal
    that does not say WHICH source was read invites exactly that.
    """
    return DataDefect(
        0,
        f"'{passage.slug}' does not answer {question!r} with a {expect.value}: {why}. "
        f"Reading '{passage.slug}' again will return the same thing -- a different source is needed.",
    )


def _as_kind(raw: Any, kind: ScalarKind) -> Any:
    if kind is ScalarKind.QUANTITY:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw.strip().replace(",", ""))
            except ValueError:
                return None
        return None

    if kind is ScalarKind.IDENTIFIER:
        return str(raw) if isinstance(raw, (str, int)) else None

    if kind is ScalarKind.TEXT:
        return raw if isinstance(raw, str) else None

    if kind is ScalarKind.BOOL:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("yes", "true", "no", "false"):
            return raw.strip().lower() in ("yes", "true")
        return None

    return None


__all__ = [
    "Citation",
    "Extractor",
    "Interpreted",
    "InterpretedList",
    "Passage",
    "Retriever",
    "interpret",
    "interpret_list",
    "search_corpus",
]
