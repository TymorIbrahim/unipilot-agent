"""One literal, spelled two ways, and an error that could not be obeyed.

Asked "How many credits have I completed at each grade level -- 90+, 80s, 70s,
below?", the deployed agent spent six turns and returned nothing. The trace:

    step 2  {"path":"grade","op":">=","value":{"value":90}}
            -> cannot type {'value': 90} (dict); give an explicit 'kind' of [...]
    step 3  {"path":"grade","op":">=","value":{"kind":"quantity","value":90}}
            -> cannot type {'kind': 'quantity', 'value': 90} (dict); give an
               explicit 'kind' of [...]
    steps 4, 5, 6  the same two shapes again, alternating

The model did exactly what it was told and got the same message back, because
`kind` is read from the PREDICATE level and it had nested one inside the value.

The wrapped form is not a mistake, either. In a `compute` expression a literal
IS `{"value": 90}` -- the only way to write one, since a bare number there would
be an ungrounded typed digit. A model that has just written such an expression
carries the spelling across, which is the reasonable thing to do.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.codec import ParseError, parse_predicate
from app.agent_core.facts.types import ScalarKind


def _value(predicate):
    return predicate.value


class TestEverySpellingOfNinety:
    @pytest.mark.parametrize("spec", [
        {"path": "grade", "op": ">=", "value": 90},
        {"path": "grade", "op": ">=", "value": {"value": 90}},
        {"path": "grade", "op": ">=", "value": {"kind": "quantity", "value": 90}},
        {"path": "grade", "op": ">=", "value": 90, "kind": "quantity"},
    ])
    def test_all_four_parse_to_the_same_quantity(self, spec: dict) -> None:
        scalar = _value(parse_predicate(spec))
        assert scalar.value == 90
        assert scalar.kind is ScalarKind.QUANTITY

    def test_the_nested_kind_is_honoured_not_ignored(self) -> None:
        scalar = _value(parse_predicate(
            {"path": "code", "op": "=", "value": {"kind": "identifier", "value": "00940224"}}))
        assert scalar.kind is ScalarKind.IDENTIFIER
        assert scalar.value == "00940224"

    def test_wrapped_literals_inside_an_in_list(self) -> None:
        predicate = parse_predicate(
            {"path": "c", "op": "in", "value": [{"value": 1}, {"value": 2}]})
        assert [s.value for s in predicate.value] == [1, 2]


class TestTheOtherShapesStillMeanWhatTheyMeant:
    def test_a_fact_reference_is_not_a_literal(self) -> None:
        """`{"fact": name}` must keep resolving as a reference, not be unwrapped."""
        predicate = parse_predicate({"path": "userId", "op": "=", "value": {"fact": "me"}})
        assert getattr(predicate.value, "name", None) == "me"

    def test_a_fact_reference_with_a_field_survives(self) -> None:
        predicate = parse_predicate(
            {"path": "c", "op": "in", "value": {"fact": "completed", "field": "courseId"}})
        assert getattr(predicate.value, "field", None) == "courseId"

    def test_a_path_comparison_survives(self) -> None:
        predicate = parse_predicate(
            {"path": "earned", "op": "<", "value": {"path": "required"}})
        assert getattr(predicate.value, "dotted", None) == "required"


class TestTheErrorCanBeActedOn:
    def test_an_unrecognised_object_says_what_to_write(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_predicate({"path": "g", "op": "=", "value": {"nonsense": 1}})
        message = str(caught.value)
        assert "BESIDE the value, not inside it" in message, (
            "the old message asked for a 'kind' the model had already nested"
        )
        assert '{"value": 90}' in message
