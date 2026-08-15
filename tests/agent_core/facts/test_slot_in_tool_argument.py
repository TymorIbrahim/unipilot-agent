"""A `{fact}` slot typed into a tool ARGUMENT is not filled by anything.

Slots belong to the answer. An argument is handed to the tool verbatim, so

    {"tool": "search_corpus", "args": {"query": "{program_slug} required courses"}}

searches the corpus for the literal characters "{program_slug}".

Measured on a live run of "Plan my winter semester.": that exact call was the
FIRST search of the run. It matched four unrelated tracks, `extract_list` then
could not find the student's own page among the retrieved passages, and the
model spent NINE of its sixteen turns re-searching with rephrasings that were
never the problem. The same question had taken eight turns and 116s; this one
took sixteen and 194s, and nothing in between said the query it sent was not
the query it wrote.

Nothing is substituted here. Guessing what the model meant is the grounding
failure this layer exists to prevent -- it says what went wrong and what the two
correct forms are.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Scalar, ScalarKind


def _detail(result) -> str:
    """Every defect message the call produced, as one string."""
    return " ".join(str(getattr(d, "detail", d)) for d in result.defects.values())


def _context() -> DispatchContext:
    return DispatchContext(
        facts={
            "program_slug": HeldFact(
                value=Scalar(ScalarKind.IDENTIFIER, "track-information-systems-engineering"),
                basis=Basis.OFFICIAL_RECORD,
            ),
            "me": HeldFact(
                value=Scalar(ScalarKind.IDENTIFIER, "6a578a2da43a2cfe1bcc791c"),
                basis=Basis.OFFICIAL_RECORD,
            ),
        }
    )


class TestTheLiveFailure:
    async def test_the_exact_call_that_cost_nine_turns(self) -> None:
        result = await dispatch(
            {
                "tool": "search_corpus",
                "as": "track_page",
                "args": {"query": "{program_slug} required courses by semester"},
            },
            _context(),
        )
        assert result.defects, "the poisoned query was dispatched anyway"
        assert "{program_slug}" in _detail(result)

    async def test_it_names_the_value_the_model_wanted(self) -> None:
        """The repair has to be one step. Told only "that is wrong", the model
        rephrases the query, which is what it did nine times."""
        result = await dispatch(
            {"tool": "search_corpus", "as": "p", "args": {"query": "{program_slug} electives"}},
            _context(),
        )
        assert "track-information-systems-engineering" in _detail(result)

    async def test_it_names_both_correct_forms(self) -> None:
        result = await dispatch(
            {"tool": "search_corpus", "as": "p", "args": {"query": "{program_slug}"}},
            _context(),
        )
        detail = _detail(result)
        assert '{"fact": "program_slug"}' in detail
        assert "ANSWERS" in detail

    async def test_it_fails_before_the_tool_runs(self) -> None:
        """The point is that the search never happens. A poisoned query that
        returns plausible-looking wrong pages is worse than no query."""
        result = await dispatch(
            {"tool": "search_corpus", "as": "p", "args": {"query": "{program_slug}"}},
            _context(),
        )
        assert not result.facts


class TestItLooksEverywhereAnArgumentCanHide:
    @pytest.mark.parametrize(
        "args",
        [
            {"query": "{program_slug}"},
            {"nested": {"query": "{program_slug}"}},
            {"items": ["fine", "{program_slug}"]},
            {"deep": [{"q": "{program_slug}"}]},
        ],
        ids=["flat", "nested-mapping", "in-a-list", "list-of-mappings"],
    )
    async def test_a_slot_is_found_at_any_depth(self, args: dict) -> None:
        result = await dispatch({"tool": "search_corpus", "as": "p", "args": args}, _context())
        assert result.defects, f"missed the slot in {args}"


class TestItDoesNotCryWolf:
    async def test_braces_naming_nothing_held_are_left_alone(self) -> None:
        """Braces are ordinary characters. Only a name that RESOLVES is evidence
        the model meant a substitution -- otherwise a query about set notation
        or a regex would be refused for no reason."""
        result = await dispatch(
            {"tool": "search_corpus", "as": "p", "args": {"query": "what is {x} in set builder"}},
            _context(),
        )
        assert "written into a tool argument" not in _detail(result)

    async def test_the_proper_fact_form_still_works(self) -> None:
        """`{"fact": ...}` is the supported idiom and must be untouched."""
        result = await dispatch(
            {
                "tool": "find",
                "as": "mine",
                "args": {
                    "source": "completed_courses",
                    "predicate": {"path": "userId", "op": "=", "value": {"fact": "me"}},
                },
            },
            _context(),
        )
        assert "written into a tool argument" not in _detail(result)
