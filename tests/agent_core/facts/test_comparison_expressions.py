"""The algebra could add two counts but never compare them.

"Am I eligible" is `met_groups >= required_groups`. The expression grammar had
add, subtract, multiply, divide, min and max -- and no comparison -- so there was
no way to derive the one value the question asks for. A live run spent three
turns rediscovering that:

    'prereq_analysis': eligible stage 0: expression must be {'path': ...},
    {'value': ...}, or one of ['add', 'div', 'divide', 'max', ...]

and then answered with `{required_group_count}` slots for facts it had never
managed to derive, which the answer boundary correctly refused. The run failed,
and the cause was a missing operator rather than anything the model did wrong.

"Do I meet the minimum", "is my average above the threshold" and "have I earned
enough credits" are all the same shape.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Scalar, ScalarKind

Q = ScalarKind.QUANTITY


def _context() -> DispatchContext:
    return DispatchContext(
        facts={
            "met": HeldFact(value=Scalar(Q, 1), basis=Basis.OFFICIAL_RECORD),
            "needed": HeldFact(value=Scalar(Q, 1), basis=Basis.OFFICIAL_RECORD),
            "gpa": HeldFact(value=Scalar(Q, 78.5), basis=Basis.OFFICIAL_RECORD),
        }
    )


async def _value(expression: dict):
    result = await dispatch(
        {"tool": "compute", "args": {"pipelines": [{"name": "out", "value": expression}]}},
        _context(),
    )
    assert not result.defects, {n: d.message for n, d in result.defects.items()}
    return result.facts["out"].value


class TestComparisonsProduceATruthValue:
    async def test_the_eligibility_shape_works(self) -> None:
        scalar = await _value({"gte": [{"fact": "met"}, {"fact": "needed"}]})
        assert scalar.value is True

    async def test_a_threshold_check_works(self) -> None:
        scalar = await _value({"gt": [{"fact": "gpa"}, {"value": 65}]})
        assert scalar.value is True

    async def test_a_failing_threshold_is_false_not_an_error(self) -> None:
        scalar = await _value({"lt": [{"fact": "gpa"}, {"value": 65}]})
        assert scalar.value is False

    async def test_every_comparison_is_reachable(self) -> None:
        for name in ("gte", "gt", "lte", "lt", "eq"):
            scalar = await _value({name: [{"fact": "met"}, {"fact": "needed"}]})
            assert scalar.kind is ScalarKind.BOOL, f"{name} must yield a truth value"


class TestTheResultIsNotAQuantity:
    async def test_a_comparison_is_typed_bool(self) -> None:
        """Typing it QUANTITY would let it be summed, and would show the answer
        boundary a "number" that is really a yes."""
        scalar = await _value({"gte": [{"fact": "met"}, {"fact": "needed"}]})
        assert scalar.kind is ScalarKind.BOOL

    async def test_arithmetic_is_still_a_quantity(self) -> None:
        scalar = await _value({"sub": [{"fact": "gpa"}, {"value": 5}]})
        assert scalar.kind is ScalarKind.QUANTITY
        assert scalar.value == 73.5


class TestTheModelIsToldItExists:
    def test_the_catalog_teaches_the_comparison_form(self) -> None:
        """An operator the prompt never mentions is one the model cannot use --
        which is how this gap survived: the capability was simply absent, and
        nothing in the catalog said so either."""
        from app.agent_core.facts.catalog import render_catalog

        rendered = render_catalog()
        assert '"gte"' in rendered or "`gte`" in rendered
        assert "eligible" in rendered
