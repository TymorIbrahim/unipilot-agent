"""JSON <-> pipeline -- phase 9a of docs/agent/tools_implementation_plan.md.

A complete algebra the model cannot emit is useless, so this is where
constructibility stops being a design claim and becomes a parser.

Two things it owes:

  - ROUND-TRIP fidelity, so what the model writes is what runs
  - PARSE ERRORS that name the mistake and the legal alternatives, because a
    model that gets "invalid pipeline" back has nothing to act on and will
    re-emit the same thing
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.codec import ParseError, parse_pipelines, parse_predicate
from app.agent_core.facts.operators import Arith, ArithOp, Literal, PathRef, Pipeline
from app.agent_core.facts.predicate import Always, And, Comparison, Not, Op, Or, Path
from app.agent_core.facts.types import ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


class TestPredicates:
    def test_a_simple_comparison(self) -> None:
        parsed = parse_predicate({"path": "grade", "op": ">", "value": 90})
        assert isinstance(parsed, Comparison)
        assert parsed.path == Path.parse("grade")
        assert parsed.op is Op.GT
        assert parsed.value.value == 90

    def test_a_json_number_is_a_quantity_and_a_string_is_an_identifier(self) -> None:
        """The inference rule, stated once and relied on everywhere: numbers are
        quantities, strings are identifiers. It gets the two common cases right
        -- `grade > 90` and `id == "00940224"` -- and anything else needs an
        explicit kind rather than a guess."""
        assert parse_predicate({"path": "grade", "op": ">", "value": 90}).value.kind is Q
        assert parse_predicate({"path": "id", "op": "=", "value": "00940224"}).value.kind is I

    def test_an_explicit_kind_overrides_inference(self) -> None:
        parsed = parse_predicate({"path": "note", "op": "=", "value": "elective", "kind": "text"})
        assert parsed.value.kind is ScalarKind.TEXT

    def test_a_field_to_field_comparison_uses_a_path_marker(self) -> None:
        parsed = parse_predicate({"path": "grade", "op": ">", "value": {"path": "passing"}})
        assert isinstance(parsed.value, Path)
        assert parsed.value.dotted == "passing"

    def test_boolean_composition(self) -> None:
        parsed = parse_predicate({"and": [
            {"path": "grade", "op": ">", "value": 90},
            {"not": {"path": "track", "op": "=", "value": "ee"}},
        ]})
        assert isinstance(parsed, And)
        assert isinstance(parsed.terms[1], Not)

    def test_or_and_always(self) -> None:
        assert isinstance(parse_predicate({"or": [{"path": "a", "op": "=", "value": 1}]}), Or)
        assert isinstance(parse_predicate({"always": True}), Always)

    def test_in_takes_a_list(self) -> None:
        parsed = parse_predicate({"path": "id", "op": "in", "value": ["a", "b"]})
        assert isinstance(parsed.value, tuple) and len(parsed.value) == 2


class TestPipelines:
    def test_a_full_pipeline_round_trips_to_the_right_shape(self) -> None:
        parsed = parse_pipelines([{
            "name": "remaining",
            "source": "required",
            "stages": [{"op": "difference", "other": "completed", "on": "id"}],
        }])
        assert len(parsed) == 1
        assert isinstance(parsed[0], Pipeline)
        assert parsed[0].stages[0].args["on"] == Path.parse("id")

    def test_paths_in_args_become_path_objects(self) -> None:
        parsed = parse_pipelines([{
            "name": "top", "source": "t",
            "stages": [{"op": "sort", "path": "score", "dir": "desc"}, {"op": "limit", "n": 1}],
        }])
        assert parsed[0].stages[0].args["path"] == Path.parse("score")
        assert parsed[0].stages[1].args["n"] == 1

    def test_project_field_maps_become_paths(self) -> None:
        parsed = parse_pipelines([{
            "name": "p", "source": "t",
            "stages": [{"op": "project", "fields": {"course": "id"}}],
        }])
        assert parsed[0].stages[0].args["fields"]["course"] == Path.parse("id")

    def test_aggregate_carries_its_op_and_path(self) -> None:
        parsed = parse_pipelines([{
            "name": "total", "source": "t",
            "stages": [{"op": "aggregate", "agg": "sum", "path": "credits"}],
        }])
        args = parsed[0].stages[0].args
        assert args["op"] == "sum" and args["path"] == Path.parse("credits")

    def test_arith_and_compare_functions_resolve_to_enums_not_strings(self) -> None:
        """The runner will not coerce a raw string on the model's behalf, so a
        stage carrying one fails at EVALUATION time -- far too late to tell the
        model anything it can act on. The codec resolves it here or not at all.
        """
        arith = parse_pipelines([{
            "name": "g", "source": "a", "stages": [{"op": "arith", "other": "b", "fn": "divide"}],
        }])
        assert arith[0].stages[0].args["op"] is ArithOp.DIVIDE

        compare = parse_pipelines([{
            "name": "h", "source": "a", "stages": [{"op": "compare", "other": "b", "fn": ">"}],
        }])
        assert compare[0].stages[0].args["op"] is Op.GT

    def test_an_unknown_arithmetic_function_lists_the_real_ones(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{
                "name": "g", "source": "a", "stages": [{"op": "arith", "other": "b", "fn": "exponentiate"}],
            }])
        assert "divide" in str(caught.value)


class TestScalarExpressions:
    def test_extend_parses_an_arithmetic_tree(self) -> None:
        parsed = parse_pipelines([{
            "name": "w", "source": "t",
            "stages": [{"op": "extend", "fields": {
                "points": {"multiply": [{"path": "grade"}, {"path": "credits"}]}
            }}],
        }])
        expression = parsed[0].stages[0].args["fields"]["points"]
        assert isinstance(expression, Arith)
        assert expression.op is ArithOp.MULTIPLY
        assert isinstance(expression.left, PathRef)

    def test_max_and_min_parse_as_arith_ops(self) -> None:
        parsed = parse_pipelines([{
            "name": "w", "source": "t",
            "stages": [{"op": "extend", "fields": {
                "floored": {"max": [{"value": 0}, {"path": "raw"}]},
                "capped": {"minimum": [{"value": 100}, {"path": "raw"}]},
            }}],
        }])
        fields = parsed[0].stages[0].args["fields"]
        assert fields["floored"].op is ArithOp.MAX
        assert fields["capped"].op is ArithOp.MIN

    def test_a_literal_inside_an_expression(self) -> None:
        parsed = parse_pipelines([{
            "name": "w", "source": "t",
            "stages": [{"op": "extend", "fields": {"x": {"add": [{"path": "a"}, {"value": 2}]}}}],
        }])
        expression = parsed[0].stages[0].args["fields"]["x"]
        assert isinstance(expression.right, Literal)
        assert expression.right.value.value == 2

    def test_nested_arithmetic(self) -> None:
        parsed = parse_pipelines([{
            "name": "w", "source": "t",
            "stages": [{"op": "extend", "fields": {
                "x": {"divide": [{"subtract": [{"path": "a"}, {"path": "b"}]}, {"path": "c"}]}
            }}],
        }])
        expression = parsed[0].stages[0].args["fields"]["x"]
        assert isinstance(expression.left, Arith)
        assert expression.left.op is ArithOp.SUBTRACT


class TestErrorsAreActionable:
    def test_an_unknown_operator_lists_the_real_ones(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"name": "p", "source": "t", "stages": [{"op": "frobnicate"}]}])
        assert "frobnicate" in str(caught.value)
        assert "difference" in str(caught.value), "must list the operators that DO exist"

    def test_an_unknown_comparator_lists_the_real_ones(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_predicate({"path": "a", "op": "≈", "value": 1})
        assert ">" in str(caught.value)

    def test_a_missing_name_says_which_pipeline(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"source": "t", "stages": []}])
        assert "name" in str(caught.value)

    def test_sugar_is_named_as_sugar_rather_than_rejected_blankly(self) -> None:
        """`intersection` is real, just not a basis operator. Saying so points at
        the fix; 'unknown operator' sends the model hunting."""
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{"name": "p", "source": "t", "stages": [{"op": "intersection", "other": "x"}]}])
        assert "sugar" in str(caught.value).lower() or "expand" in str(caught.value).lower()

    def test_a_non_list_payload_is_rejected_clearly(self) -> None:
        with pytest.raises(ParseError):
            parse_pipelines({"name": "p"})  # type: ignore[arg-type]

    def test_an_arith_node_with_the_wrong_arity_says_so(self) -> None:
        with pytest.raises(ParseError) as caught:
            parse_pipelines([{
                "name": "w", "source": "t",
                "stages": [{"op": "extend", "fields": {"x": {"add": [{"path": "a"}]}}}],
            }])
        assert "two" in str(caught.value).lower() or "2" in str(caught.value)


class TestJsonToAnswer:
    """Parsing is not the point; RUNNING what was parsed is.

    These go from a JSON document -- the literal shape a model would emit --
    through the parser, the runner, and out to a number. If any layer disagrees
    with another about the wire format, it shows up here rather than in a live
    run costing real money.
    """

    def test_a_weighted_average_from_json(self) -> None:
        """The case that exposed the missing `extend` operator, now driven end to
        end from the outside."""
        from app.agent_core.facts.runner import Succeeded, run_pipelines
        from app.agent_core.facts.types import (
            Basis,
            Collection,
            Completeness,
            Record,
            Scalar,
        )

        transcript = Collection(
            records=(
                Record(fields={"grade": Scalar(Q, 90.0), "credits": Scalar(Q, 4.0)}, basis=Basis.OFFICIAL_RECORD),
                Record(fields={"grade": Scalar(Q, 80.0), "credits": Scalar(Q, 1.0)}, basis=Basis.OFFICIAL_RECORD),
            ),
            completeness=Completeness(complete=True, total=2),
        )

        document = [
            {
                "name": "points",
                "source": "transcript",
                "stages": [
                    {"op": "extend", "fields": {"p": {"multiply": [{"path": "grade"}, {"path": "credits"}]}}},
                    {"op": "aggregate", "agg": "sum", "path": "p"},
                ],
            },
            {
                "name": "credits",
                "source": "transcript",
                "stages": [{"op": "aggregate", "agg": "sum", "path": "credits"}],
            },
            {
                "name": "gpa",
                "source": "points",
                "stages": [{"op": "arith", "other": "credits", "fn": "divide"}],
            },
        ]

        results = run_pipelines(parse_pipelines(document), {"transcript": transcript})
        assert isinstance(results["gpa"], Succeeded)
        assert results["gpa"].value.value == pytest.approx((90 * 4 + 80 * 1) / 5)

    def test_a_filter_and_count_from_json(self) -> None:
        from app.agent_core.facts.runner import Succeeded, run_pipelines
        from app.agent_core.facts.types import (
            Basis,
            Collection,
            Completeness,
            Record,
            Scalar,
        )

        courses = Collection(
            records=tuple(
                Record(fields={"id": Scalar(I, code), "grade": Scalar(Q, grade)}, basis=Basis.OFFICIAL_RECORD)
                for code, grade in (("00940224", 95.0), ("00960211", 60.0), ("00970800", 91.0))
            ),
            completeness=Completeness(complete=True, total=3),
        )

        pipelines = parse_pipelines([{
            "name": "distinctions",
            "source": "courses",
            "stages": [
                {"op": "select", "predicate": {"path": "grade", "op": ">", "value": 90}},
                {"op": "aggregate", "agg": "count"},
            ],
        }])

        results = run_pipelines(pipelines, {"courses": courses})
        assert isinstance(results["distinctions"], Succeeded)
        assert results["distinctions"].value.value == 2
