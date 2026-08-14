"""Dispatch -- phase 9d of docs/agent/tools_implementation_plan.md.

Where a parsed tool call becomes a held fact.

The property worth protecting: tool arguments name FACTS, never data. A tool
that accepted a payload would let the model hand-copy a transcript into an
argument, and the copy is where data gets quietly reshaped. Every test that
passes a collection here passes its NAME.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact, resolve_answer
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.operators import DataDefect, ExpressionDefect
from app.agent_core.facts.prose import Passage
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _rec(basis: Basis = Basis.OFFICIAL_RECORD, **fields) -> Record:
    typed = {
        k: Scalar(Q, v) if isinstance(v, (int, float)) else Scalar(I, str(v))
        for k, v in fields.items()
    }
    return Record(fields=typed, basis=basis)


def _coll(*records, complete: bool = True) -> Collection:
    return Collection(records=records, completeness=Completeness(complete=complete, total=len(records)))


TRANSCRIPT = _coll(_rec(id="00940224", credits=3.5), _rec(id="00960211", credits=3.0))
REQUIRED = _coll(_rec(id="00940224"), _rec(id="00960211"), _rec(id="00970800"))


def _context(**facts) -> DispatchContext:
    return DispatchContext(
        facts={name: HeldFact(value=value, basis=Basis.OFFICIAL_RECORD) for name, value in facts.items()}
    )


class TestRouting:
    async def test_an_unknown_tool_lists_the_real_ones(self) -> None:
        result = await dispatch({"tool": "frobnicate", "as": "x"}, _context())
        assert "compute" in result.defects["call"].message

    async def test_a_call_without_a_result_name_is_refused(self) -> None:
        """Unnamed results cannot be referenced by anything later, so the call
        would be work with nowhere to go."""
        result = await dispatch({"tool": "traverse", "args": {}}, _context())
        assert "as" in result.defects["call"].message

    async def test_compute_is_exempt_because_its_pipelines_name_themselves(self) -> None:
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "transcript", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}},
            _context(transcript=TRANSCRIPT),
        )
        assert "n" in result.facts


class TestArgumentsNameFactsNotData:
    async def test_a_tool_referring_to_an_unheld_fact_lists_what_is_held(self) -> None:
        result = await dispatch(
            {"tool": "traverse", "as": "reached", "args": {"edges": "nonexistent", "start": "a"}},
            _context(transcript=TRANSCRIPT),
        )
        assert "transcript" in result.defects["reached"].message

    async def test_a_scalar_where_a_collection_is_needed_is_refused(self) -> None:
        context = DispatchContext(facts={"total": HeldFact(Scalar(Q, 5.0), Basis.OFFICIAL_RECORD)})
        result = await dispatch(
            {"tool": "forecast", "as": "f", "args": {"observations": "total", "target": "spring"}},
            context,
        )
        assert "scalar" in result.defects["f"].message.lower()


class TestCompute:
    async def test_several_pipelines_land_as_several_facts(self) -> None:
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "remaining", "source": "required",
                 "stages": [{"op": "difference", "other": "transcript", "on": "id"}]},
                {"name": "how_many", "source": "remaining",
                 "stages": [{"op": "aggregate", "agg": "count"}]},
            ]}},
            _context(transcript=TRANSCRIPT, required=REQUIRED),
        )
        assert result.facts["how_many"].value.value == 1
        assert result.facts["remaining"].value.records[0].fields["id"].value == "00970800"

    async def test_one_failing_pipeline_does_not_discard_the_others(self) -> None:
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "good", "source": "transcript", "stages": [{"op": "aggregate", "agg": "count"}]},
                {"name": "bad", "source": "transcript",
                 "stages": [{"op": "aggregate", "agg": "sum", "path": "ghost"}]},
            ]}},
            _context(transcript=TRANSCRIPT),
        )
        assert "good" in result.facts
        assert "bad" in result.defects

    async def test_a_blocked_pipeline_says_what_it_was_waiting_for(self) -> None:
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "broken", "source": "transcript",
                 "stages": [{"op": "aggregate", "agg": "sum", "path": "ghost"}]},
                {"name": "downstream", "source": "transcript",
                 "stages": [{"op": "union", "other": "broken"}]},
            ]}},
            _context(transcript=TRANSCRIPT),
        )
        assert "broken" in result.defects["downstream"].message

    async def test_a_malformed_pipeline_is_reported_not_raised(self) -> None:
        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [{"source": "transcript"}]}},
            _context(transcript=TRANSCRIPT),
        )
        assert result.defects and "name" in next(iter(result.defects.values())).message


class TestProseChain:
    async def test_search_then_interpret_needs_no_retyped_text(self) -> None:
        """`interpret` is handed a slug, not a passage. Retyped prose is prose
        that can drift, and the citation would then point at text that differs
        from what was read."""

        class _Retriever:
            async def search(self, query, limit):
                return (Passage("track-ise", "ISE", "The degree requires 155 credits.", 0.9),)[:limit]

        class _Extractor:
            async def extract(self, passage, question, expect):
                return "155", passage.excerpt

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "credits"}}, context)

        result = await dispatch(
            {"tool": "interpret", "as": "required", "args": {"slug": "track-ise", "question": "how many", "expect": "quantity"}},
            context,
        )
        held = result.facts["required"]
        assert held.value.value == 155.0
        assert held.basis is Basis.LLM_INTERPRETATION
        assert held.citation.source == "track-ise"

    async def test_search_then_extract_list_yields_a_classifiable_collection(self) -> None:
        """The plural of interpret, end to end: a retrieved section's listed
        codes come back as records the algebra can `in`-classify against, each
        verified present so an invented code cannot enter the set."""

        excerpt = "Faculty electives: 0960327 and 0960324 are offered."

        class _Retriever:
            async def search(self, query, limit):
                return (Passage("track-ise", "ISE", excerpt, 0.9),)[:limit]

        class _Extractor:
            async def extract(self, passage, question, expect):
                raise AssertionError("extract_list must use extract_all")

            async def extract_all(self, passage, question, expect):
                return [("0960327", excerpt), ("0960324", excerpt), ("9999999", "absent")]

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "electives"}}, context)

        result = await dispatch(
            {"tool": "extract_list", "as": "elective_codes",
             "args": {"slug": "track-ise", "question": "elective course codes", "expect": "identifier"}},
            context,
        )
        held = result.facts["elective_codes"]
        # The wiki renders codes a digit short; the extracted set comes back
        # canonicalised to 8 digits so it joins to the catalog's courseNumber.
        assert [r.fields["value"].value for r in held.value.records] == ["00960327", "00960324"]
        assert held.citation.source == "track-ise"

    async def test_extract_list_restores_the_wiki_dropped_leading_zero(self) -> None:
        """The whole reason a live 'include electives' plan found zero electives:
        the wiki shows `0960600`, the catalog stores `00960600`, and the semi-join
        silently matched nothing. extract_list must hand back the 8-digit form."""

        class _Retriever:
            async def search(self, query, limit):
                return (Passage("track-ise", "ISE", "Faculty electives: 0960600, 0960620.", 0.9),)[:limit]

        class _Extractor:
            async def extract(self, p, q, e):
                return None, ""

            async def extract_all(self, passage, question, expect):
                return [("0960600", passage.excerpt), ("0960620", passage.excerpt)]

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "electives"}}, context)
        result = await dispatch(
            {"tool": "extract_list", "as": "codes",
             "args": {"slug": "track-ise", "question": "codes", "expect": "identifier"}},
            context,
        )
        assert [r.fields["value"].value for r in result.facts["codes"].value.records] == ["00960600", "00960620"]

    async def test_search_stashes_the_whole_page_when_the_retriever_serves_it(self) -> None:
        """The stash holds the FULL page, not just the chunk that made top-k -- so
        extract_list reads a section retrieval never surfaced. This is the fix for
        the live plan that classified against only 2 of ~40 required codes and
        then refused on a duplicate."""

        class _Retriever:
            async def search(self, query, limit):
                # Search surfaces only a tiny header chunk...
                return (Passage("track-ise", "ISE", "Information Systems Engineering track.", 0.9),)[:limit]

            def page(self, slug):
                # ...but the whole page, with the elective section, is held by slug.
                return "ISE track.\n## Faculty Electives\n0960600 and 0960620 are offered."

        class _Extractor:
            async def extract(self, p, q, e):
                return None, ""

            async def extract_all(self, passage, question, expect):
                import re

                return [(c, passage.excerpt) for c in re.findall(r"\d{7}", passage.excerpt)]

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "ise"}}, context)
        result = await dispatch(
            {"tool": "extract_list", "as": "codes",
             "args": {"slug": "track-ise", "question": "codes", "expect": "identifier"}},
            context,
        )
        # Codes that appear ONLY in the full page (never in the search chunk) are
        # extracted -- proof the stash carried the whole page, not the fragment.
        assert [r.fields["value"].value for r in result.facts["codes"].value.records] == ["00960600", "00960620"]

    async def test_interpret_restores_the_wiki_dropped_leading_zero(self) -> None:
        """A SINGLE course code read via interpret gets the same canonicalisation as
        extract_list, so it too joins to the catalog's 8-digit courseNumber."""

        class _Retriever:
            async def search(self, query, limit):
                return (Passage("track-ise", "ISE", "The commerce course uses code 0960221.", 0.9),)[:limit]

        class _Extractor:
            async def extract(self, passage, question, expect):
                return "0960221", passage.excerpt

            async def extract_all(self, p, q, e):
                raise AssertionError("interpret must use extract, not extract_all")

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "commerce"}}, context)
        result = await dispatch(
            {"tool": "interpret", "as": "code",
             "args": {"slug": "track-ise", "question": "the commerce course code", "expect": "identifier"}},
            context,
        )
        assert result.facts["code"].value.value == "00960221"

    async def test_chunks_of_one_page_accumulate_under_its_slug(self) -> None:
        """The wiki is heading-segmented, so one page returns several chunks that
        share its slug. Keyed by slug alone the last chunk clobbered the rest and
        a later extract_list saw only the final section -- a live plan lost the
        whole electives list this way. The stash must hold the WHOLE retrieved
        page."""

        class _Retriever:
            async def search(self, query, limit):
                return (
                    Passage("track-ise", "Required", "Required Courses: 0940314", 0.9),
                    Passage("track-ise", "Electives", "Faculty Elective Requirements: 0960327", 0.8),
                )[:limit]

        class _Extractor:
            async def extract(self, p, q, e):
                return None, ""

            async def extract_all(self, passage, question, expect):
                import re

                return [(c, passage.excerpt) for c in re.findall(r"\d{7}", passage.excerpt)]

        context = DispatchContext(retriever=_Retriever(), extractor=_Extractor())
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "ise"}}, context)
        result = await dispatch(
            {"tool": "extract_list", "as": "codes", "args": {"slug": "track-ise", "question": "codes", "expect": "identifier"}},
            context,
        )
        # Both the required code and the elective code survive -- proof neither
        # chunk overwrote the other. Canonicalised to 8 digits on the way out.
        assert {r.fields["value"].value for r in result.facts["codes"].value.records} == {"00940314", "00960327"}

    async def test_a_prose_tool_resolves_a_fact_ref_slug(self) -> None:
        """The model generalises the {"fact": name} idiom to `slug`, and it is
        right to -- the track's page slug IS a fact it just derived (the program
        slug). A live run passed {"fact": "program_slug"} and lost the whole type
        step because the slug was compared as a literal dict. It must resolve."""

        class _Retriever:
            async def search(self, query, limit):
                return (Passage("track-ise", "ISE", "Faculty electives: 0960327", 0.9),)[:limit]

        class _Extractor:
            async def extract(self, p, q, e):
                return None, ""

            async def extract_all(self, passage, question, expect):
                return [("0960327", passage.excerpt)]

        context = DispatchContext(
            retriever=_Retriever(),
            extractor=_Extractor(),
            facts={"program_slug": HeldFact(Scalar(I, "track-ise"), Basis.WIKI_DERIVED)},
        )
        await dispatch({"tool": "search_corpus", "as": "hits", "args": {"query": "ise"}}, context)
        result = await dispatch(
            {"tool": "extract_list", "as": "codes", "args": {"slug": {"fact": "program_slug"}, "question": "codes", "expect": "identifier"}},
            context,
        )
        assert [r.fields["value"].value for r in result.facts["codes"].value.records] == ["00960327"]

    async def test_extract_list_on_an_unretrieved_slug_is_refused(self) -> None:
        class _Extractor:
            async def extract(self, passage, question, expect):
                return "x", ""

            async def extract_all(self, passage, question, expect):
                return []

        result = await dispatch(
            {"tool": "extract_list", "as": "x", "args": {"slug": "never-searched", "expect": "identifier"}},
            DispatchContext(extractor=_Extractor()),
        )
        assert "search first" in result.defects["x"].message

    async def test_interpreting_an_unretrieved_slug_is_refused(self) -> None:
        """An extractor IS wired here -- otherwise this would pass for the wrong
        reason, on 'no interpreter configured' rather than on the slug."""

        class _Extractor:
            async def extract(self, passage, question, expect):
                return "1", passage.excerpt

        result = await dispatch(
            {"tool": "interpret", "as": "x", "args": {"slug": "never-searched", "expect": "quantity"}},
            DispatchContext(extractor=_Extractor()),
        )
        assert "search first" in result.defects["x"].message

    async def test_an_unwired_interpreter_says_so_rather_than_crashing(self) -> None:
        result = await dispatch(
            {"tool": "interpret", "as": "x", "args": {"slug": "s", "expect": "quantity"}},
            DispatchContext(),
        )
        assert "no interpreter is configured" in result.defects["x"].message

    async def test_an_unwired_retriever_says_so_rather_than_crashing(self) -> None:
        """It crashed with AttributeError before -- the catalog advertised a tool
        the wiring could not serve."""
        result = await dispatch(
            {"tool": "search_corpus", "as": "p", "args": {"query": "anything"}},
            DispatchContext(),
        )
        assert "no corpus is configured" in result.defects["p"].message


class TestPropose:
    async def test_grounds_must_name_held_facts(self) -> None:
        result = await dispatch(
            {"tool": "propose", "as": "p", "args": {
                "action": "register", "target": "00960211", "grounds": ["imaginary"]}},
            _context(transcript=TRANSCRIPT),
        )
        assert "imaginary" in result.defects["p"].message

    async def test_a_proposal_inherits_the_weakest_basis_of_its_grounds(self) -> None:
        """A proposal is only as sound as the weakest thing behind it, so one
        built on a simulated plan is marked speculative without anyone deciding
        to mark it."""
        context = DispatchContext(facts={
            "records": HeldFact(TRANSCRIPT, Basis.OFFICIAL_RECORD),
            "plan": HeldFact(REQUIRED, Basis.SIMULATED),
        })
        result = await dispatch(
            {"tool": "propose", "as": "p", "args": {
                "action": "register", "target": "00960211", "grounds": ["records", "plan"]}},
            context,
        )
        assert result.proposal.speculative is True

    async def test_a_proposal_is_not_a_fact(self) -> None:
        """Nothing has happened, so nothing should be derivable from it."""
        context = DispatchContext(facts={"records": HeldFact(TRANSCRIPT, Basis.OFFICIAL_RECORD)})
        result = await dispatch(
            {"tool": "propose", "as": "p", "args": {
                "action": "register", "target": "x", "grounds": ["records"]}},
            context,
        )
        assert result.facts == {}
        assert result.proposal is not None


class TestEndToEnd:
    async def test_a_question_goes_from_tool_calls_to_a_grounded_answer(self) -> None:
        """The whole layer in one test: dispatch admits facts, compute derives
        from them, and the answer boundary refuses anything the facts did not
        produce."""
        context = _context(transcript=TRANSCRIPT, required=REQUIRED)

        result = await dispatch(
            {"tool": "compute", "args": {"pipelines": [
                {"name": "remaining", "source": "required",
                 "stages": [{"op": "difference", "other": "transcript", "on": "id"}]},
                {"name": "how_many", "source": "remaining",
                 "stages": [{"op": "aggregate", "agg": "count"}]},
            ]}},
            context,
        )
        context.facts.update(result.facts)

        answer = resolve_answer("You have {how_many} required course left.", context.facts)
        assert answer.text == "You have 1 required course left."
        assert answer.basis is Basis.OFFICIAL_RECORD

        refused = resolve_answer("You have 1 required course left.", context.facts)
        assert "no fact" in refused.reason
