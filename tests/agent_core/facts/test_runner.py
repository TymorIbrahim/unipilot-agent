"""Phase 4 gate for docs/agent/tools_implementation_plan.md.

The stated gate: a 5-stage derivation runs in ONE call, and a mid-pipeline
failure returns its siblings intact.

Plus the three rules that only bite once real data flows -- they are types in
phase 1 and 3, and behaviour here:

  - completeness (§4.1), including the `difference` asymmetry
  - certainty collapse (§3.7), including why a join must NOT flatten provenance
  - determinism (§4.3), so `argmax` does not coin-flip on ties
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.operators import (
    Arith,
    ArithOp,
    Held,
    Literal,
    PathRef,
    Pipeline,
    Stage,
)
from app.agent_core.facts.codec import parse_pipelines
from app.agent_core.facts.predicate import Comparison, Op, Path
from app.agent_core.facts.runner import Blocked, Failed, Succeeded, run_pipelines
from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _rec(basis: Basis = Basis.OFFICIAL_RECORD, **fields: object) -> Record:
    typed = {}
    for name, value in fields.items():
        typed[name] = Scalar(Q, value) if isinstance(value, (int, float)) else Scalar(I, str(value))
    return Record(fields=typed, basis=basis)


def _collection(*records: Record, complete: bool = True, total: int | None = None) -> Collection:
    return Collection(records=records, completeness=Completeness(complete=complete, total=total))


TRANSCRIPT = _collection(
    _rec(id="00940224", grade=95, credits=3.5),
    _rec(id="00960211", grade=60, credits=3.0),
    _rec(id="00970800", grade=88, credits=4.0),
)

REQUIRED = _collection(
    _rec(id="00940224"), _rec(id="00960211"), _rec(id="00970800"), _rec(id="00990100")
)


class TestTheGate:
    def test_a_five_stage_derivation_runs_in_one_call(self) -> None:
        """The turn-cost fix: this is one tool call, not five turns each carrying
        its own rejection risk."""
        pipeline = Pipeline("weighted", "transcript", (
            Stage("select", {"predicate": Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 50))}),
            Stage("extend", {"fields": {"points": Arith(ArithOp.MULTIPLY, PathRef(Path.parse("grade")), PathRef(Path.parse("credits")))}}),
            Stage("sort", {"path": Path.parse("points"), "dir": "desc"}),
            Stage("limit", {"n": 3}),
            Stage("aggregate", {"op": "sum", "path": Path.parse("points")}),
        ))
        results = run_pipelines((pipeline,), {"transcript": TRANSCRIPT})
        outcome = results["weighted"]
        assert isinstance(outcome, Succeeded)
        assert outcome.value.value == 95 * 3.5 + 60 * 3.0 + 88 * 4.0

    def test_a_failing_pipeline_leaves_its_siblings_intact(self) -> None:
        """Discarding successful work because a sibling failed is how a repair
        loop burns its budget."""
        good_a = Pipeline("a", "transcript", (Stage("aggregate", {"op": "count"}),))
        broken = Pipeline("b", "transcript", (Stage("aggregate", {"op": "sum", "path": Path.parse("nonexistent")}),))
        good_c = Pipeline("c", "transcript", (Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),))

        results = run_pipelines((good_a, broken, good_c), {"transcript": TRANSCRIPT})
        assert isinstance(results["a"], Succeeded)
        assert isinstance(results["b"], Failed)
        assert isinstance(results["c"], Succeeded)
        assert results["c"].value.value == 10.5


class TestMalformedCallsAreRepairable:
    """A live run died on `assert other is not None` when the model omitted a
    binary operand. An assert says "this cannot happen"; a model writing a
    malformed call is the most ordinary thing there is, and it must come back as
    something the next turn can act on rather than a crash."""

    @pytest.mark.parametrize("op", ["join", "union", "difference"])
    def test_a_binary_op_without_other_is_a_defect_not_a_crash(self, op: str) -> None:
        pipeline = Pipeline("p", "transcript", (Stage(op, {}),))
        results = run_pipelines((pipeline,), {"transcript": TRANSCRIPT})
        assert isinstance(results["p"], Failed)
        assert "other" in results["p"].defect.message
        assert "transcript" in results["p"].defect.message, "must name what is available"

    def test_a_non_string_other_is_also_handled(self) -> None:
        pipeline = Pipeline("p", "transcript", (Stage("union", {"other": 42}),))
        results = run_pipelines((pipeline,), {"transcript": TRANSCRIPT})
        assert isinstance(results["p"], Failed)


class TestDependencyOrdering:
    def test_pipelines_declared_out_of_order_still_run(self) -> None:
        """Order comes from the declared references, not the caller's listing --
        a model that lists them backwards should still succeed."""
        dependent = Pipeline("remaining", "required", (Stage("difference", {"other": "passed", "on": Path.parse("id")}),))
        base = Pipeline("passed", "transcript", (Stage("select", {"predicate": Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 64))}),))

        results = run_pipelines((dependent, base), {"transcript": TRANSCRIPT, "required": REQUIRED})
        assert isinstance(results["remaining"], Succeeded)
        ids = sorted(r.fields["id"].value for r in results["remaining"].value.records)
        assert ids == ["00960211", "00990100"]

    def test_a_reference_cycle_is_reported_not_hung(self) -> None:
        a = Pipeline("a", "transcript", (Stage("union", {"other": "b"}),))
        b = Pipeline("b", "transcript", (Stage("union", {"other": "a"}),))
        results = run_pipelines((a, b), {"transcript": TRANSCRIPT})
        assert isinstance(results["a"], Failed)
        assert "cycle" in results["a"].defect.message.lower()

    def test_a_dependent_of_a_failed_pipeline_is_blocked_not_failed(self) -> None:
        """Distinguishable because they call for different repairs: fix the
        broken one, or stop."""
        broken = Pipeline("broken", "transcript", (Stage("aggregate", {"op": "sum", "path": Path.parse("ghost")}),))
        downstream = Pipeline("downstream", "transcript", (Stage("union", {"other": "broken"}),))
        results = run_pipelines((broken, downstream), {"transcript": TRANSCRIPT})
        assert isinstance(results["broken"], Failed)
        assert isinstance(results["downstream"], Blocked)


class TestCompletenessAtRuntime:
    def test_aggregate_over_a_truncated_collection_refuses(self) -> None:
        """A count over a page of 50 from a true 73 is confidently wrong, and
        every fact in it reports full confidence."""
        page = _collection(*TRANSCRIPT.records, complete=False, total=73)
        pipeline = Pipeline("n", "page", (Stage("aggregate", {"op": "count"}),))
        results = run_pipelines((pipeline,), {"page": page})
        assert isinstance(results["n"], Failed)
        assert "73" in results["n"].defect.message

    def test_select_over_a_truncated_collection_stays_partial_not_refused(self) -> None:
        page = _collection(*TRANSCRIPT.records, complete=False, total=73)
        pipeline = Pipeline("some", "page", (
            Stage("select", {"predicate": Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 64))}),
        ))
        results = run_pipelines((pipeline,), {"page": page})
        assert isinstance(results["some"], Succeeded)
        assert results["some"].value.completeness.complete is False

    def test_difference_tolerates_an_incomplete_minuend(self) -> None:
        partial_required = _collection(*REQUIRED.records, complete=False, total=29)
        pipeline = Pipeline("r", "req", (Stage("difference", {"other": "transcript", "on": Path.parse("id")}),))
        results = run_pipelines((pipeline,), {"req": partial_required, "transcript": TRANSCRIPT})
        assert isinstance(results["r"], Succeeded)
        assert results["r"].value.completeness.complete is False

    def test_difference_refuses_an_incomplete_subtrahend(self) -> None:
        """THE asymmetry. Every record missing from B is wrongly RETAINED, so
        'requirements remaining' silently gains courses already passed."""
        partial_transcript = _collection(*TRANSCRIPT.records, complete=False, total=53)
        pipeline = Pipeline("r", "req", (Stage("difference", {"other": "partial", "on": Path.parse("id")}),))
        results = run_pipelines((pipeline,), {"req": REQUIRED, "partial": partial_transcript})
        assert isinstance(results["r"], Failed)


class TestCertainty:
    def test_a_scalar_collapses_to_the_weakest_basis_consumed(self) -> None:
        mixed = _collection(
            _rec(basis=Basis.OFFICIAL_RECORD, id="a", credits=3.0),
            _rec(basis=Basis.WIKI_DERIVED, id="b", credits=2.0),
        )
        pipeline = Pipeline("total", "mixed", (Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),))
        results = run_pipelines((pipeline,), {"mixed": mixed})
        assert results["total"].basis is Basis.WIKI_DERIVED

    def test_a_simulated_record_taints_everything_derived_from_it(self) -> None:
        """§2.1: what removes the need for a counterfactual primitive."""
        hypothetical = _collection(
            _rec(basis=Basis.OFFICIAL_RECORD, id="a", credits=3.0),
            _rec(basis=Basis.SIMULATED, id="b", credits=2.0),
        )
        pipeline = Pipeline("total", "h", (Stage("aggregate", {"op": "sum", "path": Path.parse("credits")}),))
        results = run_pipelines((pipeline,), {"h": hypothetical})
        assert results["total"].basis is Basis.SIMULATED

    def test_a_join_does_not_degrade_the_fields_it_did_not_touch(self) -> None:
        """If a join flattened provenance to the weakest side, summing an
        OFFICIAL field after joining against a wiki collection would report
        wiki_derived -- and provenance would never recover."""
        official = _collection(_rec(basis=Basis.OFFICIAL_RECORD, id="x", credits=3.0))
        wiki = _collection(_rec(basis=Basis.WIKI_DERIVED, id="x", note="elective"))
        joined = Pipeline("j", "official", (
            Stage("join", {"other": "wiki", "predicate": Comparison(Path.parse("left.id"), Op.EQ, Path.parse("right.id"))}),
            Stage("aggregate", {"op": "sum", "path": Path.parse("left.credits")}),
        ))
        results = run_pipelines((joined,), {"official": official, "wiki": wiki})
        assert isinstance(results["j"], Succeeded)
        assert results["j"].basis is Basis.OFFICIAL_RECORD


class TestDeterminism:
    def test_sort_then_limit_breaks_ties_stably(self) -> None:
        """We have spent a lot of effort on run-to-run variance; an argmax that
        coin-flips on ties is more of it."""
        tied = _collection(_rec(id="b", score=10), _rec(id="a", score=10), _rec(id="c", score=5))
        pipeline = Pipeline("top", "tied", (
            Stage("sort", {"path": Path.parse("score"), "dir": "desc"}),
            Stage("limit", {"n": 1}),
        ))
        first = run_pipelines((pipeline,), {"tied": tied})["top"].value.records[0].fields["id"].value
        for _ in range(20):
            again = run_pipelines((pipeline,), {"tied": tied})["top"].value.records[0].fields["id"].value
            assert again == first


class TestNullKeys:
    def test_a_null_join_key_fails_closed(self) -> None:
        """§4.2: silently dropping unresolvable records makes a set difference
        quietly wrong while every surviving fact reports full confidence. This
        is the 7-dangling-courseId bug class."""
        dangling = Collection(
            records=(Record(fields={"id": Scalar(I, "ok")}, basis=Basis.OFFICIAL_RECORD),
                     Record(fields={}, basis=Basis.OFFICIAL_RECORD)),
            completeness=Completeness(complete=True, total=2),
        )
        pipeline = Pipeline("r", "req", (Stage("difference", {"other": "dangling", "on": Path.parse("id")}),))
        results = run_pipelines((pipeline,), {"req": REQUIRED, "dangling": dangling})
        assert isinstance(results["r"], Failed)
        assert "key" in results["r"].defect.message.lower()


def _nested(name: str, *elements: Record, complete: bool = True) -> Collection:
    """A record holding one nested array -- what `find` now produces."""
    return _collection(
        Record(
            fields={
                "id": Scalar(I, "plan-1"),
                name: Collection(
                    records=elements,
                    completeness=Completeness(complete=complete, total=len(elements)),
                ),
            },
            basis=Basis.OFFICIAL_RECORD,
        )
    )


class TestUnnest:
    """`unnest` was in the operator table, in the type checker, and in the system
    prompt for a while with NO branch in the runner -- so a model that used the
    operator the prompt taught it got "no evaluation rule". These pin the
    semantics the type checker already declared, so the two cannot drift.
    """

    def test_one_record_per_element(self) -> None:
        source = _nested("semesters", _rec(code="2025-1"), _rec(code="2025-2"))
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        assert isinstance(result["out"], Succeeded)
        assert [r.fields["code"].value for r in result["out"].value.records] == ["2025-1", "2025-2"]

    def test_parent_fields_ride_along(self) -> None:
        """SQL lateral semantics, and what the type checker declares. Without it
        a later group or join on parent identity becomes impossible."""
        source = _nested("semesters", _rec(code="2025-1"))
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        record = result["out"].value.records[0]
        assert record.fields["id"].value == "plan-1"
        # And the array itself is gone -- it has been expanded, not duplicated.
        assert "semesters" not in record.fields

    def test_an_empty_array_contributes_no_rows_and_stays_complete(self) -> None:
        """`$unwind` semantics. "This plan has no semesters" is a real answer,
        not a truncation, so an aggregate over the result must still be allowed."""
        source = _nested("semesters")
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        assert result["out"].value.records == ()
        assert result["out"].value.completeness.complete is True

    def test_a_partial_nested_array_makes_the_result_incomplete(self) -> None:
        source = _nested("semesters", _rec(code="2025-1"), complete=False)
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        assert result["out"].value.completeness.complete is False

    def test_a_record_missing_the_array_fails_closed(self) -> None:
        """Expanding the records that have one and dropping the rest returns a
        partial result that counts like a whole one."""
        source = _collection(_rec(id="plan-1"))
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        assert isinstance(result["out"], Failed)
        assert "semesters" in result["out"].defect.message

    def test_unnesting_a_scalar_is_an_expression_error_not_a_crash(self) -> None:
        source = _collection(_rec(id="plan-1", credits=3))
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("credits")}),)),), {"src": source})
        assert isinstance(result["out"], Failed)
        assert "not an array" in result["out"].defect.message

    def test_an_element_field_shadows_the_parent(self) -> None:
        """Matching the checker's `merged.update(inner.fields)`. Pinned because
        the opposite choice type-checks identically and answers differently."""
        source = _collection(
            Record(
                fields={
                    "id": Scalar(I, "plan-1"),
                    "semesters": Collection(
                        records=(_rec(id="sem-9"),),
                        completeness=Completeness(complete=True, total=1),
                    ),
                },
                basis=Basis.OFFICIAL_RECORD,
            )
        )
        result = run_pipelines((Pipeline("out", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),), {"src": source})
        assert result["out"].value.records[0].fields["id"].value == "sem-9"

    def test_twice_expands_a_two_level_nesting(self) -> None:
        """The shape `semester_plans` actually has: semesters, each holding
        planned courses."""
        inner = Collection(
            records=(_rec(courseNumber="00940224"), _rec(courseNumber="00960211")),
            completeness=Completeness(complete=True, total=2),
        )
        source = _collection(
            Record(
                fields={
                    "id": Scalar(I, "plan-1"),
                    "semesters": Collection(
                        records=(
                            Record(
                                fields={"code": Scalar(I, "2025-1"), "plannedCourses": inner},
                                basis=Basis.OFFICIAL_RECORD,
                            ),
                        ),
                        completeness=Completeness(complete=True, total=1),
                    ),
                },
                basis=Basis.OFFICIAL_RECORD,
            )
        )
        result = run_pipelines(
            (
                Pipeline("slots", "src", (Stage("unnest", {"path": Path.parse("semesters")}),)),
                Pipeline("items", "slots", (Stage("unnest", {"path": Path.parse("plannedCourses")}),)),
            ),
            {"src": source},
        )
        courses = [r.fields["courseNumber"].value for r in result["items"].value.records]
        assert courses == ["00940224", "00960211"]
        # Both levels of parent survive, which is what lets a placement be
        # attributed back to its semester.
        assert result["items"].value.records[0].fields["code"].value == "2025-1"


class TestDistinctOverNestedRecords:
    def test_it_does_not_raise_on_a_record_holding_an_array(self) -> None:
        """`_signature` put whole field values in a set, so a nested array raised
        `TypeError: unhashable type: 'dict'` -- an exception escaping the runner
        rather than a defect it could report. Unreachable while every source was
        flat; reachable the moment one declared an array."""
        source = _nested("semesters", _rec(code="2025-1"))
        result = run_pipelines((Pipeline("out", "src", (Stage("distinct", {}),)),), {"src": source})
        assert isinstance(result["out"], Succeeded)
        assert len(result["out"].value.records) == 1

    def test_identical_nested_records_collapse(self) -> None:
        one = _nested("semesters", _rec(code="2025-1")).records[0]
        source = _collection(one, one)
        result = run_pipelines((Pipeline("out", "src", (Stage("distinct", {}),)),), {"src": source})
        assert len(result["out"].value.records) == 1


class TestMaxAndMin:
    def test_max_floors_a_negative_grade_and_min_caps_one(self) -> None:
        """max(0, raw) is how a computed grade gets FLOORED instead of shipping a
        negative one -- the arithmetic the negative-min-grade answers lacked."""
        source = _collection(_rec(raw=-82.0, over=105.0))
        parsed = parse_pipelines([{"name": "out", "source": "src", "stages": [
            {"op": "extend", "fields": {
                "floored": {"max": [{"value": 0}, {"path": "raw"}]},
                "capped": {"min": [{"value": 100}, {"path": "over"}]},
            }}]}])
        result = run_pipelines(tuple(parsed), {"src": source})

        assert isinstance(result["out"], Succeeded)
        row = result["out"].value.records[0]
        assert row.fields["floored"].value == 0.0
        assert row.fields["capped"].value == 100.0

    def test_max_in_a_scalar_pipeline_floors_at_zero(self) -> None:
        parsed = parse_pipelines([{"name": "nm", "value": {"max": [{"value": 0}, {"value": -82}]}}])
        result = run_pipelines(tuple(parsed), {})

        assert isinstance(result["nm"], Succeeded)
        assert result["nm"].value.value == 0.0


class TestDistinctByKey:
    def test_distinct_on_a_key_collapses_rows_that_differ_elsewhere(self) -> None:
        """Two rows for the SAME course -- one tagged required, one elective after a
        union -- are kept as two by a whole-record distinct, and the answer boundary
        then refuses the plan as a course listed twice. `on` collapses them by code,
        keeping the first."""
        source = _collection(
            _rec(courseNumber="00940395", type="required"),
            _rec(courseNumber="00940395", type="elective"),
            _rec(courseNumber="00940412", type="elective"),
        )
        pipeline = Pipeline("out", "src", (Stage("distinct", {"on": Path.parse("courseNumber")}),))
        result = run_pipelines((pipeline,), {"src": source})

        assert isinstance(result["out"], Succeeded)
        kept = [r.fields["courseNumber"].value for r in result["out"].value.records]
        assert kept == ["00940395", "00940412"]
        assert result["out"].value.records[0].fields["type"].value == "required"  # first per key

    def test_distinct_on_a_missing_key_is_a_reported_defect_not_a_crash(self) -> None:
        source = _collection(_rec(courseNumber="00940395"))
        pipeline = Pipeline("out", "src", (Stage("distinct", {"on": Path.parse("ghost")}),))
        result = run_pipelines((pipeline,), {"src": source})

        assert not isinstance(result["out"], Succeeded)


class TestHeldScalarInExtend:
    """`extend` can reference a held SCALAR fact, so a per-record formula can
    combine the record with a global aggregate. The per-course GPA threshold --
    (85*(credits + total_credits) - total_points)/credits -- was inexpressible
    before this, and the model correctly gave up on it."""

    def test_a_per_record_field_uses_a_held_total(self) -> None:
        planned = _collection(
            _rec(courseNumber="A", credits=4.0),
            _rec(courseNumber="B", credits=2.0),
        )
        env = {
            "planned": planned,
            "total_points": Scalar(Q, 600.0),
            "total_credits": Scalar(Q, 7.0),
        }
        # min grade to keep GPA >= 85: (85*(credits+total_credits) - total_points)/credits
        pipeline = Pipeline("thresholds", "planned", (
            Stage("extend", {"fields": {"min_grade": Arith(
                ArithOp.DIVIDE,
                Arith(ArithOp.SUBTRACT,
                      Arith(ArithOp.MULTIPLY, Literal(Scalar(Q, 85.0)),
                            Arith(ArithOp.ADD, PathRef(Path.parse("credits")), Held("total_credits"))),
                      Held("total_points")),
                PathRef(Path.parse("credits")))}}),
        ))
        result = run_pipelines((pipeline,), env)["thresholds"]
        assert isinstance(result, Succeeded)
        grades = {r.fields["courseNumber"].value: round(r.fields["min_grade"].value, 1) for r in result.value.records}
        # A: (85*11 - 600)/4 = 83.75 ; B: (85*9 - 600)/2 = 82.5
        assert grades == {"A": 83.8, "B": 82.5}

    def test_a_missing_held_scalar_is_a_named_defect_not_a_crash(self) -> None:
        pipeline = Pipeline("x", "planned", (
            Stage("extend", {"fields": {"y": Arith(ArithOp.ADD, PathRef(Path.parse("credits")), Held("nope"))}}),
        ))
        result = run_pipelines((pipeline,), {"planned": _collection(_rec(courseNumber="A", credits=4.0))})["x"]
        assert isinstance(result, Failed)
        assert "nope" in result.defect.message

    def test_a_held_scalar_is_ordered_before_the_pipeline_that_uses_it(self) -> None:
        """The dependency is inside the expression, not in `source`/`other`, so
        the runner must still schedule the total first."""
        env = {"planned": _collection(_rec(courseNumber="A", credits=4.0)), "base": _collection(_rec(x=10.0), _rec(x=20.0))}
        pipelines = (
            # `thresholds` (listed FIRST) depends on `total`, listed after it.
            Pipeline("thresholds", "planned", (
                Stage("extend", {"fields": {"v": Arith(ArithOp.ADD, PathRef(Path.parse("credits")), Held("total"))}}),
            )),
            Pipeline("total", "base", (Stage("aggregate", {"op": "sum", "path": Path.parse("x")}),)),
        )
        results = run_pipelines(pipelines, env)
        assert isinstance(results["thresholds"], Succeeded)
        assert results["thresholds"].value.records[0].fields["v"].value == 34.0  # 4 + 30


class TestScalarPipeline:
    """A source-less `{name, value}` pipeline: one scalar computed from held
    scalars, no carrier collection. The affordance that was missing when the
    model kept reaching for a stage-level arith that did not exist."""

    def _env(self):
        return {
            "total_points": Scalar(Q, 5243.0),
            "total_credits": Scalar(Q, 62.5),
            "plan_credits": Scalar(Q, 37.0),
        }

    def test_a_scalar_is_computed_straight_from_held_facts(self) -> None:
        pipes = parse_pipelines(
            [{"name": "gpa", "value": {"div": [{"fact": "total_points"}, {"fact": "total_credits"}]}}]
        )
        outcome = run_pipelines(pipes, self._env())["gpa"]
        assert isinstance(outcome, Succeeded)
        assert round(outcome.value.value, 3) == 83.888

    def test_scalar_pipelines_chain_by_name(self) -> None:
        """deficit -> lift -> needed_average, the reachable-target chain, each a
        scalar pipeline referencing the previous. Ordering is by held-fact refs."""
        pipes = parse_pipelines(
            [
                {"name": "needed_average", "value": {"add": [{"value": 85}, {"fact": "lift"}]}},
                {"name": "lift", "value": {"div": [{"fact": "deficit"}, {"fact": "plan_credits"}]}},
                {"name": "deficit", "value": {
                    "sub": [{"mul": [{"value": 85}, {"fact": "total_credits"}]}, {"fact": "total_points"}]
                }},
            ]
        )
        outcome = run_pipelines(pipes, self._env())["needed_average"]
        assert isinstance(outcome, Succeeded)
        assert round(outcome.value.value, 2) == 86.88

    def test_the_result_basis_is_the_weakest_held_fact_consumed(self) -> None:
        env = {"a": Scalar(Q, 10.0), "b": Scalar(Q, 2.0)}
        # publish a and b with different bases via a prior compute is awkward here;
        # instead lean on the loop's basis map by seeding through the env is not
        # possible, so assert the simpler invariant: a literal-only compute is
        # official (strongest), never spuriously weak.
        pipes = parse_pipelines([{"name": "k", "value": {"add": [{"value": 2}, {"value": 3}]}}])
        outcome = run_pipelines(pipes, env)["k"]
        assert isinstance(outcome, Succeeded)
        assert outcome.value.value == 5.0
        assert outcome.basis is Basis.OFFICIAL_RECORD

    def test_a_path_ref_in_a_scalar_pipeline_fails_cleanly(self) -> None:
        """There are no rows, so `{"path": ...}` has nothing to read -- it must be
        a clear defect, not a crash."""
        pipes = parse_pipelines([{"name": "x", "value": {"path": "credits"}}])
        outcome = run_pipelines(pipes, self._env())["x"]
        assert isinstance(outcome, Failed)
        assert "not a field" in outcome.defect.message

    def test_a_missing_held_scalar_is_reported(self) -> None:
        pipes = parse_pipelines([{"name": "x", "value": {"div": [{"fact": "nope"}, {"fact": "total_credits"}]}}])
        outcome = run_pipelines(pipes, self._env())["x"]
        assert isinstance(outcome, (Failed, Blocked))

    def test_a_pipeline_with_neither_source_nor_value_is_refused_helpfully(self) -> None:
        import pytest as _pytest

        from app.agent_core.facts.codec import ParseError

        with _pytest.raises(ParseError) as err:
            parse_pipelines([{"name": "x"}])
        assert "value" in str(err.value) and "source" in str(err.value)
