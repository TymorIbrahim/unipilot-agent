"""Phase 6 gate for docs/agent/tools_implementation_plan.md.

The stated gate: an interpreted `Quantity` is directly consumable by `arith`
with NO coercion.

That is the whole point of the prose bridge. There is no algebra of text, so
`search_corpus` and `interpret` exist to convert prose into well-formed facts --
after which the algebra takes over. If an interpreted value arrives as a string
that arithmetic has to re-parse, the bridge has not done its job and every
downstream operator inherits a guessing problem.

Retrieval ranking and the extraction model are injected, not built here: this
phase owns the TYPE boundary, not the search quality.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.operators import Arith, ArithOp, DataDefect, Literal, PathRef, Pipeline, Stage
from app.agent_core.facts.predicate import Path
from app.agent_core.facts.prose import (
    Citation,
    Interpreted,
    InterpretedList,
    Passage,
    interpret,
    interpret_list,
    search_corpus,
)
from app.agent_core.facts.runner import Failed, Succeeded, run_pipelines
from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER
T = ScalarKind.TEXT

PASSAGES = (
    Passage(slug="track-ise", title="ISE Track", excerpt="The degree requires 155 credits in total.", score=0.91),
    Passage(slug="regulations", title="Regulations", excerpt="A student may appeal within 14 days.", score=0.77),
    Passage(slug="calendar", title="Calendar", excerpt="Spring begins in March.", score=0.55),
)


class _StubRetriever:
    def __init__(self, passages=PASSAGES):
        self.passages = passages
        self.calls = []

    async def search(self, query: str, limit: int):
        self.calls.append((query, limit))
        return self.passages[:limit]


class _StubExtractor:
    """Stands in for the model. Returns whatever it was told to return, so the
    TYPE contract can be tested without paying for a completion."""

    def __init__(self, value, kind=Q, quote="The degree requires 155 credits in total."):
        self.value, self.kind, self.quote = value, kind, quote

    async def extract(self, passage: Passage, question: str, expect: ScalarKind):
        return self.value, self.quote


class TestSearchCorpus:
    async def test_results_are_a_collection_the_algebra_can_operate_on(self) -> None:
        """Search hits ARE records. Making them a Collection means `sort`,
        `limit` and `select` work on them for free, instead of the prose side
        needing its own parallel vocabulary."""
        result = await search_corpus(_StubRetriever(), "how many credits", limit=3)
        assert isinstance(result, Collection)
        pipeline = Pipeline("top", "hits", (
            Stage("sort", {"path": Path.parse("score"), "dir": "desc"}),
            Stage("limit", {"n": 1}),
        ))
        outcome = run_pipelines((pipeline,), {"hits": result})["top"]
        assert isinstance(outcome, Succeeded)
        assert outcome.value.records[0].fields["slug"].value == "track-ise"

    async def test_a_hit_is_wiki_derived_not_official(self) -> None:
        """Verbatim corpus text is not an official record, and saying so is what
        keeps a policy claim from outranking a transcript."""
        result = await search_corpus(_StubRetriever(), "q", limit=3)
        assert all(r.basis is Basis.WIKI_DERIVED for r in result.records)

    async def test_score_is_a_quantity_and_slug_is_an_identifier(self) -> None:
        record = (await search_corpus(_StubRetriever(), "q", limit=1)).records[0]
        assert record.fields["score"].kind is Q
        assert record.fields["slug"].kind is I

    async def test_a_truncated_search_reports_incompleteness(self) -> None:
        """Retrieval is inherently top-k, so a hit list is almost never all of
        what matched. Claiming completeness would let `count` over search hits
        report a confident number that means nothing."""
        result = await search_corpus(_StubRetriever(), "q", limit=2)
        assert result.completeness.complete is False


class TestInterpret:
    async def test_an_interpreted_quantity_is_a_real_quantity(self) -> None:
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "how many credits", expect=Q)
        assert isinstance(fact, Interpreted)
        assert fact.value.kind is Q
        assert fact.value.value == 155.0
        assert isinstance(fact.value.value, float)

    async def test_the_gate_an_interpreted_quantity_feeds_arithmetic_with_no_coercion(self) -> None:
        """§16.4's open item, closed. Prose said 155; the transcript says 62.5
        earned. `155 - 62.5` must be expressible without anything re-parsing a
        string on the way."""
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "how many credits", expect=Q)
        assert isinstance(fact, Interpreted)

        transcript = Collection(
            records=(Record(fields={"earned": Scalar(Q, 62.5)}, basis=Basis.OFFICIAL_RECORD),),
            completeness=Completeness(complete=True, total=1),
        )
        pipeline = Pipeline("remaining", "transcript", (
            Stage("extend", {"fields": {"left": Arith(
                ArithOp.SUBTRACT, Literal(fact.value), PathRef(Path.parse("earned"))
            )}}),
            Stage("aggregate", {"op": "sum", "path": Path.parse("left")}),
        ))
        outcome = run_pipelines((pipeline,), {"transcript": transcript})["remaining"]
        assert isinstance(outcome, Succeeded)
        assert outcome.value.value == 92.5

    async def test_an_interpretation_carries_a_mandatory_citation(self) -> None:
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "q", expect=Q)
        assert isinstance(fact.citation, Citation)
        assert fact.citation.source == "track-ise"
        assert "155" in fact.citation.quote

    async def test_an_interpretation_is_weaker_than_an_official_record(self) -> None:
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "q", expect=Q)
        assert fact.basis is Basis.LLM_INTERPRETATION
        assert fact.basis.strength < Basis.OFFICIAL_RECORD.strength

    async def test_an_interpreted_value_taints_what_is_derived_from_it(self) -> None:
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "q", expect=Q)
        mixed = Collection(
            records=(Record(fields={"earned": Scalar(Q, 1.0)}, basis=fact.basis),),
            completeness=Completeness(complete=True, total=1),
        )
        pipeline = Pipeline("t", "mixed", (Stage("aggregate", {"op": "sum", "path": Path.parse("earned")}),))
        assert run_pipelines((pipeline,), {"mixed": mixed})["t"].basis is Basis.LLM_INTERPRETATION

    async def test_a_value_that_is_not_the_expected_kind_fails_closed(self) -> None:
        """The model claiming a quantity does not make one. Validating the claim
        against the value is cheap and catches the case where the passage simply
        had no number in it."""
        result = await interpret(_StubExtractor("not a number"), PASSAGES[1], "how many credits", expect=Q)
        assert isinstance(result, DataDefect)

    async def test_a_failure_names_the_source_so_it_is_not_re_queried(self) -> None:
        """Observed live: an agent re-asked the same page after a
        cannot-determine and burned turns getting the identical answer. The
        refusal has to say WHICH source it read and that re-reading is futile."""
        result = await interpret(_StubExtractor(None), PASSAGES[1], "how many credits", expect=Q)
        assert isinstance(result, DataDefect)
        assert "regulations" in result.message
        assert "again" in result.message.lower() or "re-read" in result.message.lower()

    async def test_a_computed_value_is_rejected_because_it_is_not_in_the_passage(self) -> None:
        """The failure that actually happens: the model quietly does the
        subtraction and returns a clean 92.5, indistinguishable from a real
        extraction by shape. It is caught by grounding, not by pattern-matching
        -- 92.5 is not in a passage that says 155.
        """
        result = await interpret(_StubExtractor("92.5"), PASSAGES[0], "how many remain", expect=Q)
        assert isinstance(result, DataDefect)
        assert "does not appear" in result.message

    async def test_a_value_actually_present_in_the_passage_is_accepted(self) -> None:
        """The other side of the same check -- it must not reject real extractions."""
        fact = await interpret(_StubExtractor("155"), PASSAGES[0], "how many credits", expect=Q)
        assert isinstance(fact, Interpreted)

    async def test_grounding_tolerates_formatting_differences(self) -> None:
        """Prose says '155'; a model may return '155.0'. Same value, and
        rejecting it would make the guard useless in practice."""
        fact = await interpret(_StubExtractor("155.0"), PASSAGES[0], "how many credits", expect=Q)
        assert isinstance(fact, Interpreted)
        assert fact.value.value == 155.0


ELECTIVES_PASSAGE = Passage(
    slug="track-ise",
    title="ISE Track",
    excerpt="Faculty Elective Requirements: 0960327, 0960324 and 0960311 are offered.",
    score=0.9,
)


class _ListExtractor:
    """Stands in for the model's SET extraction: returns whatever (value, quote)
    pairs it is told to, so `interpret_list`'s grounding can be tested cheaply."""

    def __init__(self, pairs):
        self.pairs = pairs

    async def extract(self, passage, question, expect):  # unused by interpret_list
        raise AssertionError("interpret_list must call extract_all, not extract")

    async def extract_all(self, passage, question, expect):
        return list(self.pairs)


class TestInterpretList:
    async def test_it_returns_the_listed_values_as_a_collection(self) -> None:
        """The plural of interpret: a section that lists three codes becomes a
        three-record collection the algebra can then classify against."""
        pairs = [(c, ELECTIVES_PASSAGE.excerpt) for c in ("0960327", "0960324", "0960311")]
        result = await interpret_list(_ListExtractor(pairs), ELECTIVES_PASSAGE, "elective codes", expect=I)
        assert isinstance(result, InterpretedList)
        assert [r.fields["value"].value for r in result.value.records] == ["0960327", "0960324", "0960311"]
        assert all(r.fields["value"].kind is I for r in result.value.records)

    async def test_a_hallucinated_value_is_dropped_not_trusted(self) -> None:
        """Same guarantee interpret gives, per element: a code the model invents
        is not in the passage it cites, so it cannot enter the set. Grounding is
        enforced, not requested."""
        pairs = [("0960327", ELECTIVES_PASSAGE.excerpt), ("9999999", "not in the passage")]
        result = await interpret_list(_ListExtractor(pairs), ELECTIVES_PASSAGE, "elective codes", expect=I)
        assert isinstance(result, InterpretedList)
        assert [r.fields["value"].value for r in result.value.records] == ["0960327"]

    async def test_duplicates_collapse(self) -> None:
        pairs = [("0960327", ELECTIVES_PASSAGE.excerpt), ("0960327", ELECTIVES_PASSAGE.excerpt)]
        result = await interpret_list(_ListExtractor(pairs), ELECTIVES_PASSAGE, "elective codes", expect=I)
        assert len(result.value.records) == 1

    async def test_a_set_with_no_verifiable_member_fails_closed(self) -> None:
        """An empty collection would read as 'the wiki lists no electives', a
        confident and wrong claim. A defect that names the source is honest and
        stops the model re-reading the same page."""
        result = await interpret_list(
            _ListExtractor([("9999999", "absent")]), ELECTIVES_PASSAGE, "elective codes", expect=I
        )
        assert isinstance(result, DataDefect)
        assert "track-ise" in result.message

    async def test_the_extracted_set_is_never_marked_complete(self) -> None:
        """Extraction is best-effort over a truncatable passage, so the set is
        incomplete by construction -- which is exactly why a caller must classify
        with a POSITIVE `in`, not a `difference` against it."""
        pairs = [("0960327", ELECTIVES_PASSAGE.excerpt)]
        result = await interpret_list(_ListExtractor(pairs), ELECTIVES_PASSAGE, "q", expect=I)
        assert result.value.completeness.complete is False

    async def test_a_member_is_wiki_grounded_and_carries_a_citation(self) -> None:
        pairs = [("0960327", ELECTIVES_PASSAGE.excerpt)]
        result = await interpret_list(_ListExtractor(pairs), ELECTIVES_PASSAGE, "q", expect=I)
        assert result.citation.source == "track-ise"
        assert result.basis is Basis.LLM_INTERPRETATION


class TestBridgeShape:
    async def test_search_then_interpret_is_the_whole_prose_path(self) -> None:
        """Two steps, not one: retrieval returns candidates a human can be shown,
        interpretation makes a claim. Merging them would hide which passage the
        answer actually came from."""
        hits = await search_corpus(_StubRetriever(), "how many credits", limit=1)
        best = hits.records[0]
        passage = Passage(
            slug=best.fields["slug"].value,
            title=best.fields["title"].value,
            excerpt=best.fields["excerpt"].value,
            score=best.fields["score"].value,
        )
        fact = await interpret(_StubExtractor("155"), passage, "how many credits", expect=Q)
        assert fact.citation.source == "track-ise"
