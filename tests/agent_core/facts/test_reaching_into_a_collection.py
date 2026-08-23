"""Taking one field's value out of a collection is not an expression.

A live "shortest path to 00970135" ran 196 seconds and timed out. The waste was
10 parse failures, 14 "refers to a fact which is not held", and 12 "not run: it
depends on X, which failed" -- and all 26 of the latter cascade from the 10.

The model was trying, four different ways, to pull one field out of a one-row
collection so it could feed the next `find`:

    {"only": [{"fact": "target_prereq_edges", "field": "requires"}]}
    {"value": {"fact": "target_course", "field": "courseNumber"}}

That is not an expression and never will be. It is a CHAIN WALK, which
`traverse` exists for -- "a pipeline of N joins reaches exactly N levels, and
the chain's depth is a property of the data". The error it got instead listed
the arithmetic operators, pointing away from the answer at a model trying to
follow prerequisites.

One of those shapes was worse than an error: it PARSED.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.codec import ParseError, parse_pipelines


class TestTheSilentOne:
    """`{"fact": "course", "field": "courseNumber"}` parsed to `Held("course")`
    and dropped the field without a word, so a model asking for one course
    number was handed the whole collection and told nothing.

    A parse error costs one turn. A value that is quietly the wrong thing costs
    the answer, and nothing downstream can tell the difference.
    """

    def test_an_extra_key_on_a_fact_reference_is_refused(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"name": "x", "value": {"fact": "course", "field": "courseNumber"}}])
        assert "would be ignored" in str(caught.value)

    def test_the_refusal_names_the_moves_that_work(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"name": "x", "value": {"fact": "c", "field": "n"}}])
        message = str(caught.value)
        assert "project" in message and "traverse" in message

    def test_a_plain_fact_reference_still_parses(self) -> None:
        """The idiom this shares with predicate values, and it must keep working."""
        assert parse_pipelines([{"name": "x", "value": {"fact": "credits_needed"}}])

    def test_arithmetic_over_facts_still_parses(self) -> None:
        assert parse_pipelines(
            [{"name": "x", "value": {"ceil_div": [{"fact": "a"}, {"fact": "b"}]}}]
        )


class TestTheErrorPointsAtTheRightTool:
    def test_reaching_into_a_collection_is_told_about_traverse(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines(
                [{"name": "x", "value": {"only": [{"fact": "edges", "field": "requires"}]}}]
            )
        assert "traverse" in str(caught.value)

    def test_ordinary_bad_arithmetic_is_not_sent_to_traverse(self) -> None:
        """Naming `traverse` at a model that merely mistyped a sum would point
        it away from its actual mistake, which is how this started."""
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"name": "x", "value": {"nonsense_op": [1, 2]}}])
        assert "traverse" not in str(caught.value)
