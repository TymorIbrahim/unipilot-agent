"""Cross-pipeline scalar comparison -- closing the stub left in phase 4.

"Is my spring load heavier than autumn?" is two aggregates and a comparison.
Each aggregate collapses a collection to a scalar, so the comparison needs TWO
named scalar results -- and until now only collections were publishable, so a
scalar could not be referenced at all.

This is also where the operand-position rule (§3.2) stops being theory:
`arith` forbids literals because `155 - 62.5` typed directly is the laundering
bug, while `compare` admits one on the right because a threshold genuinely can
come from the question.
"""

from __future__ import annotations

from app.agent_core.facts.operators import ArithOp, ExpressionDefect, Pipeline, Stage
from app.agent_core.facts.predicate import Comparison, Op, Path
from app.agent_core.facts.runner import Blocked, Failed, Succeeded, run_pipelines
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


def _plan(*rows: tuple[str, float], basis: Basis = Basis.OFFICIAL_RECORD) -> Collection:
    return Collection(
        records=tuple(
            Record(fields={"term": Scalar(I, term), "credits": Scalar(Q, credits)}, basis=basis)
            for term, credits in rows
        ),
        completeness=Completeness(complete=True, total=len(rows)),
    )


SPRING = _plan(("spring", 4.0), ("spring", 3.5), ("spring", 3.0))   # 10.5
AUTUMN = _plan(("autumn", 3.0), ("autumn", 3.0))                    # 6.0

SPRING_TOTAL = Pipeline("spring_total", "spring", (
    Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),
))
AUTUMN_TOTAL = Pipeline("autumn_total", "autumn", (
    Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),
))

ENV = {"spring": SPRING, "autumn": AUTUMN}


class TestScalarsAreReferenceable:
    def test_a_scalar_result_can_be_the_source_of_another_pipeline(self) -> None:
        passthrough = Pipeline("copy", "spring_total", ())
        results = run_pipelines((SPRING_TOTAL, passthrough), ENV)
        assert isinstance(results["copy"], Succeeded)
        assert results["copy"].value.value == 10.5

    def test_two_aggregates_compare_in_one_call(self) -> None:
        """The whole point: three pipelines, one call, one turn."""
        heavier = Pipeline("heavier", "spring_total", (
            Stage("compare", {"other": "autumn_total", "op": Op.GT}),
        ))
        results = run_pipelines((SPRING_TOTAL, AUTUMN_TOTAL, heavier), ENV)
        assert isinstance(results["heavier"], Succeeded)
        assert results["heavier"].value.kind is ScalarKind.BOOL
        assert results["heavier"].value.value is True

    def test_arithmetic_between_two_pipeline_results(self) -> None:
        difference = Pipeline("gap", "spring_total", (
            Stage("arith", {"other": "autumn_total", "op": ArithOp.SUBTRACT}),
        ))
        results = run_pipelines((SPRING_TOTAL, AUTUMN_TOTAL, difference), ENV)
        assert results["gap"].value.value == 4.5

    def test_declaration_order_still_does_not_matter(self) -> None:
        heavier = Pipeline("heavier", "spring_total", (
            Stage("compare", {"other": "autumn_total", "op": Op.GT}),
        ))
        results = run_pipelines((heavier, AUTUMN_TOTAL, SPRING_TOTAL), ENV)
        assert isinstance(results["heavier"], Succeeded)


class TestOperandPositions:
    def test_arith_refuses_a_literal_operand(self) -> None:
        """§3.2: a literal in DATA position is a laundered computed value. This
        is the `155 - 62.5` bug, and it must be refused even though the number
        looks perfectly reasonable."""
        laundered = Pipeline("bad", "spring_total", (
            Stage("arith", {"value": Scalar(Q, 62.5), "op": ArithOp.SUBTRACT}),
        ))
        results = run_pipelines((SPRING_TOTAL, laundered), ENV)
        assert isinstance(results["bad"], Failed)
        assert "ref" in results["bad"].defect.message.lower()

    def test_compare_admits_a_literal_threshold(self) -> None:
        """A threshold from the question is a CRITERION, not data about the
        world -- and the answer boundary independently catches an ungrounded
        number that reaches the answer, so this is not the last line of defence."""
        over = Pipeline("over_ten", "spring_total", (
            Stage("compare", {"value": Scalar(Q, 10.0), "op": Op.GT}),
        ))
        results = run_pipelines((SPRING_TOTAL, over), ENV)
        assert isinstance(results["over_ten"], Succeeded)
        assert results["over_ten"].value.value is True


class TestCertaintyAcrossPipelines:
    def test_a_comparison_is_only_as_strong_as_its_weaker_side(self) -> None:
        simulated_autumn = Pipeline("autumn_total", "sim", (
            Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),
        ))
        heavier = Pipeline("heavier", "spring_total", (
            Stage("compare", {"other": "autumn_total", "op": Op.GT}),
        ))
        env = {"spring": SPRING, "sim": _plan(("autumn", 3.0), ("autumn", 3.0), basis=Basis.SIMULATED)}
        results = run_pipelines((SPRING_TOTAL, simulated_autumn, heavier), env)
        assert results["heavier"].basis is Basis.SIMULATED


class TestFailurePropagation:
    def test_a_comparison_against_a_failed_pipeline_is_blocked(self) -> None:
        broken = Pipeline("autumn_total", "autumn", (
            Stage("aggregate", {"op": "sum", "path": Path.parse("ghost")}),
        ))
        heavier = Pipeline("heavier", "spring_total", (
            Stage("compare", {"other": "autumn_total", "op": Op.GT}),
        ))
        results = run_pipelines((SPRING_TOTAL, broken, heavier), ENV)
        assert isinstance(results["autumn_total"], Failed)
        assert isinstance(results["heavier"], Blocked)

    def test_comparing_a_scalar_against_a_collection_is_a_type_error(self) -> None:
        confused = Pipeline("nonsense", "spring_total", (
            Stage("compare", {"other": "spring", "op": Op.GT}),
        ))
        results = run_pipelines((SPRING_TOTAL, confused), ENV)
        assert isinstance(results["nonsense"], Failed)
        assert isinstance(results["nonsense"].defect, ExpressionDefect)
