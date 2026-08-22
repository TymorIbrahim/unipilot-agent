"""The algebra could divide but never round up.

Found on the deployed agent, asked "How many semesters will it take me to
graduate?" for the Data & Information Engineering student. It derived every
input correctly -- 155 required, 22 completed, 133 remaining, an 18-credit cap --
and then:

    step 4  compute {"ceil": [{"div": [...]}]}
            -> expression must be {'path': ...}, {'value': ...}, or one of [...]
    step 5  compute {"compare": [...]}          -> same
    step 6  compute {"value": ...} with no pipelines -> 0 facts
    step 6  call the tool "answer"              -> unknown tool
    step 8  compute {"max": [ratio, {"value": 8}]} -> 8

It shipped "155 more credits, so at least 9 semesters": a typed guess laundered
through `max`, with the completed credits dropped entirely. The truth is
ceil(133 / 18) = 8.

Every instruction it was following was correct. `check_periods_are_whole`
refuses "1.42 semesters", and the system prompt orders the count be derived as
ceil(credits needed / cap). The operation simply was not in the grammar -- the
same hole MAX filled when negative minimum grades had no arithmetic way out.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.codec import ParseError, parse_pipelines
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.operators import ArithOp
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY


def _compute(spec: dict, **facts: float):
    context = DispatchContext(facts={
        name: HeldFact(value=Scalar(Q, value), basis=Basis.OFFICIAL_RECORD)
        for name, value in facts.items()
    })
    outcome = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        dispatch({"tool": "compute", "as": "n",
                  "args": {"pipelines": [{"name": "n", "value": spec}]}}, context)
    )
    if outcome.defects:
        return outcome.defects
    return outcome.facts["n"].value.value


class TestTheNumberIsRight:
    def test_the_live_case(self) -> None:
        """133 remaining at 18 per semester is 8 terms, not the 9 it shipped."""
        assert _compute({"ceil_div": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=133.0, cap=18.0) == 8

    def test_the_primary_student(self) -> None:
        assert _compute({"ceil_div": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=25.5, cap=18.0) == 2

    def test_an_exact_multiple_does_not_gain_a_term(self) -> None:
        """36 credits at 18 is exactly 2, and rounding up must not make it 3."""
        assert _compute({"ceil_div": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=36.0, cap=18.0) == 2

    def test_a_remainder_of_one_credit_still_costs_a_term(self) -> None:
        assert _compute({"ceil_div": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=36.5, cap=18.0) == 3

    def test_division_by_zero_does_not_raise(self) -> None:
        assert _compute({"ceil_div": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=36.0, cap=0.0) == 0


class TestTheSpellingsAModelReachesFor:
    """The live run wrote `{"ceil": [{"div": [...]}]}` -- the unary ceiling of a
    division -- which is the form anyone writes first. It parses now, folded to
    ceil(x / 1), so the model's correct first instinct is not an arity error."""

    def test_unary_ceil_of_a_division(self) -> None:
        assert _compute({"ceil": [{"div": [{"fact": "needed"}, {"fact": "cap"}]}]},
                        needed=133.0, cap=18.0) == 8

    def test_unary_ceil_of_a_bare_value(self) -> None:
        assert _compute({"ceil": [{"value": 7.39}]}) == 8

    @pytest.mark.parametrize("spelling", ["ceil_div", "divide_round_up"])
    def test_binary_spellings(self, spelling: str) -> None:
        assert _compute({spelling: [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=133.0, cap=18.0) == 8

    @pytest.mark.parametrize("spelling", ["ceil", "ceiling", "round_up"])
    def test_unary_spellings(self, spelling: str) -> None:
        assert _compute({spelling: [{"value": 7.39}]}) == 8

    def test_two_operands_still_work_for_a_unary_spelling(self) -> None:
        """`{"ceil": [a, b]}` reads as ceil(a/b) and must not become an error."""
        assert _compute({"ceil": [{"fact": "needed"}, {"fact": "cap"}]},
                        needed=133.0, cap=18.0) == 8


class TestArityIsStillChecked:
    def test_a_binary_op_still_needs_two(self) -> None:
        with pytest.raises(ParseError, match="exactly two operands"):
            parse_pipelines([{"name": "n", "value": {"div": [{"value": 1}]}}])

    def test_three_operands_are_refused(self) -> None:
        with pytest.raises(ParseError, match="exactly two operands"):
            parse_pipelines([{"name": "n",
                              "value": {"ceil": [{"value": 1}, {"value": 2}, {"value": 3}]}}])

    def test_a_non_list_is_refused(self) -> None:
        with pytest.raises(ParseError, match="list of operands"):
            parse_pipelines([{"name": "n", "value": {"ceil": "nope"}}])


def test_the_operator_is_not_a_comparison() -> None:
    """It yields a quantity, so it must not be typed as a bool."""
    from app.agent_core.facts.operators import COMPARISONS

    assert ArithOp.CEIL_DIV not in COMPARISONS
