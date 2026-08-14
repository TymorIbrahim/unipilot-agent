"""Phase 3 gate for docs/agent/tools_implementation_plan.md.

Two static assertions, and passing only the first proves very little:

  1. TYPE CLOSURE -- every operator's result type is accepted as an input type
     somewhere, and every declared sugar expands to the basis.
  2. RELATIONAL COMPLETENESS -- each of Codd's six is exhibited as a concrete
     expression in this basis, plus generalized projection via `extend`.

A basis can be perfectly closed and still unable to express a join, which is why
closure alone is not the gate. Neither test needs a model or a database: if the
basis is unsound, that is knowable for free.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.operators import (
    OPERATORS,
    SUGAR,
    Arith,
    ArithOp,
    CollectionShape,
    DataDefect,
    ExpressionDefect,
    Literal,
    OperandPosition,
    PathRef,
    Pipeline,
    Stage,
    Ty,
    check_pipeline,
    expand_sugar,
)
from app.agent_core.facts.predicate import Always, Comparison, Op, Path
from app.agent_core.facts.types import InputRole, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER

TRANSCRIPT = CollectionShape(fields={"id": I, "grade": Q, "credits": Q, "semester": I})
REQUIRED = CollectionShape(fields={"id": I, "kind": I})
ENV = {"transcript": TRANSCRIPT, "required": REQUIRED}


class TestTypeClosure:
    def test_every_result_type_is_consumed_somewhere(self) -> None:
        produced = {spec.result for spec in OPERATORS.values()}
        consumed = {
            operand.ty
            for spec in OPERATORS.values()
            for operand in spec.operands
            if operand.position is OperandPosition.DATA
        }
        orphans = produced - consumed
        assert not orphans, f"these result types cannot feed any operator: {orphans}"

    def test_every_operator_consumes_something_it_can_be_given(self) -> None:
        produced = {spec.result for spec in OPERATORS.values()}
        for name, spec in OPERATORS.items():
            data_inputs = [o.ty for o in spec.operands if o.position is OperandPosition.DATA]
            assert data_inputs, f"{name} consumes no data -- it cannot participate in a pipeline"
            for ty in data_inputs:
                assert ty in produced, f"{name} needs a {ty} that no operator produces"

    def test_every_sugar_expands_to_basis_operators_only(self) -> None:
        """§3.8: sugar that cannot expand is not sugar -- it is new capability,
        which means the basis was incomplete."""
        for name in SUGAR:
            pipelines = expand_sugar(name, "transcript", {"other": "required", "path": Path.parse("grade")})
            for pipeline in pipelines:
                for stage in pipeline.stages:
                    assert stage.op in OPERATORS, f"sugar {name} expands to unknown op {stage.op}"
                    assert stage.op not in SUGAR, f"sugar {name} expands to other sugar ({stage.op})"

    def test_collection_operands_declare_a_completeness_role(self) -> None:
        """The §4.1 table is only enforceable if every collection input says how
        it reacts to incompleteness."""
        for name, spec in OPERATORS.items():
            for index, operand in enumerate(spec.operands):
                if operand.ty is Ty.COLLECTION and operand.position is OperandPosition.DATA:
                    assert operand.role is not None, f"{name} operand {index} has no InputRole"

    def test_difference_is_the_asymmetric_one(self) -> None:
        roles = [o.role for o in OPERATORS["difference"].operands if o.ty is Ty.COLLECTION]
        assert roles == [InputRole.MONOTONE, InputRole.REQUIRES_ALL]


class TestRelationalCompleteness:
    """Codd's six, exhibited. If any of these cannot be written, the basis is
    incomplete and capability will have to come back as pre-solved shortcuts."""

    def test_selection(self) -> None:
        pipeline = Pipeline("r", "transcript", (
            Stage("select", {"predicate": Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90))}),
        ))
        assert not isinstance(check_pipeline(pipeline, ENV), (ExpressionDefect, DataDefect))

    def test_projection(self) -> None:
        pipeline = Pipeline("r", "transcript", (Stage("project", {"fields": {"id": Path.parse("id")}}),))
        assert not isinstance(check_pipeline(pipeline, ENV), (ExpressionDefect, DataDefect))

    def test_product_via_join_on_a_constant_true_predicate(self) -> None:
        """The reason `Always` exists in the grammar. Without product, the basis
        is not relationally complete."""
        pipeline = Pipeline("r", "transcript", (Stage("join", {"other": "required", "predicate": Always()}),))
        assert not isinstance(check_pipeline(pipeline, ENV), (ExpressionDefect, DataDefect))

    def test_union(self) -> None:
        pipeline = Pipeline("r", "transcript", (Stage("union", {"other": "required"}),))
        assert not isinstance(check_pipeline(pipeline, ENV), (ExpressionDefect, DataDefect))

    def test_difference(self) -> None:
        pipeline = Pipeline("r", "required", (Stage("difference", {"other": "transcript"}),))
        assert not isinstance(check_pipeline(pipeline, ENV), (ExpressionDefect, DataDefect))

    def test_rename_via_project(self) -> None:
        """Codd's rho. Needed for self-joins, where field names collide."""
        pipeline = Pipeline("r", "transcript", (
            Stage("project", {"fields": {"course": Path.parse("id")}}),
        ))
        result = check_pipeline(pipeline, ENV)
        assert not isinstance(result, (ExpressionDefect, DataDefect))
        assert "course" in result.fields and "id" not in result.fields

    def test_generalized_projection_via_extend(self) -> None:
        """The case that exposed the missing operator: a credit-weighted average
        needs `grade x credits` per record before anything can sum it."""
        pipeline = Pipeline("r", "transcript", (
            Stage("extend", {"fields": {"points": Arith(
                ArithOp.MULTIPLY, PathRef(Path.parse("grade")), PathRef(Path.parse("credits"))
            )}}),
            Stage("aggregate", {"op": "sum", "path": Path.parse("points")}),
        ))
        result = check_pipeline(pipeline, ENV)
        assert not isinstance(result, (ExpressionDefect, DataDefect))
        assert result is Ty.SCALAR or getattr(result, "ty", None) is Ty.SCALAR

    def test_self_join_qualifies_colliding_names(self) -> None:
        pipeline = Pipeline("r", "transcript", (Stage("join", {"other": "transcript", "predicate": Always()}),))
        result = check_pipeline(pipeline, ENV)
        assert not isinstance(result, (ExpressionDefect, DataDefect))
        assert "left.id" in result.fields and "right.id" in result.fields


class TestTypeChecking:
    def test_a_scalar_cannot_feed_a_collection_stage(self) -> None:
        pipeline = Pipeline("r", "transcript", (
            Stage("aggregate", {"op": "count"}),
            Stage("select", {"predicate": Always()}),
        ))
        result = check_pipeline(pipeline, ENV)
        assert isinstance(result, ExpressionDefect)
        assert result.stage == 1
        assert "collection" in result.message.lower()

    def test_an_unknown_field_names_the_fields_that_exist(self) -> None:
        """An error a model cannot act on is retried verbatim, so it must say
        what to switch to."""
        pipeline = Pipeline("r", "transcript", (
            Stage("select", {"predicate": Comparison(Path.parse("deficit"), Op.GT, Scalar(Q, 0))}),
        ))
        result = check_pipeline(pipeline, ENV)
        assert isinstance(result, ExpressionDefect)
        assert "deficit" in result.message
        assert "grade" in result.message, "must name available fields"

    def test_arithmetic_on_a_non_quantity_field_is_rejected(self) -> None:
        pipeline = Pipeline("r", "transcript", (
            Stage("extend", {"fields": {"x": Arith(ArithOp.ADD, PathRef(Path.parse("id")), Literal(Scalar(Q, 1)))}}),
        ))
        result = check_pipeline(pipeline, ENV)
        assert isinstance(result, ExpressionDefect)
        assert "id" in result.message

    def test_a_collection_with_no_numeric_field_is_a_DATA_defect_not_an_expression_one(self) -> None:
        """Nothing to switch to means no edit can fix it -- a repair loop that
        retries here burns its budget re-deriving a pipeline that was right."""
        no_numbers = CollectionShape(fields={"id": I, "kind": I})
        pipeline = Pipeline("r", "codes", (Stage("aggregate", {"op": "sum", "path": Path.parse("id")}),))
        result = check_pipeline(pipeline, {"codes": no_numbers})
        assert isinstance(result, DataDefect)

    def test_an_unknown_source_is_an_expression_defect_naming_what_exists(self) -> None:
        pipeline = Pipeline("r", "nonexistent", (Stage("select", {"predicate": Always()}),))
        result = check_pipeline(pipeline, ENV)
        assert isinstance(result, ExpressionDefect)
        assert "transcript" in result.message

    def test_an_unknown_operator_is_rejected(self) -> None:
        pipeline = Pipeline("r", "transcript", (Stage("frobnicate", {}),))
        result = check_pipeline(pipeline, ENV)
        assert isinstance(result, ExpressionDefect)


class TestSugarExpansion:
    def test_intersection_expands_to_two_differences_over_two_pipelines(self) -> None:
        """A ∩ B = A − (A − B) reads the source TWICE, so it cannot be a linear
        stage list. Expansion is to pipelines, with the intermediate named --
        which is exactly what §3.6's named-pipeline model already provides."""
        pipelines = expand_sugar("intersection", "transcript", {"other": "required"})
        assert len(pipelines) == 2
        assert [s.op for p in pipelines for s in p.stages] == ["difference", "difference"]
        assert pipelines[1].stages[0].args["other"] == pipelines[0].name

    def test_argmax_expands_to_sort_then_limit(self) -> None:
        pipelines = expand_sugar("argmax", "transcript", {"path": Path.parse("grade")})
        assert [s.op for p in pipelines for s in p.stages] == ["sort", "limit"]

    @pytest.mark.parametrize("name", ["intersection", "argmax"])
    def test_sugar_is_not_in_the_basis(self, name: str) -> None:
        assert name not in OPERATORS
