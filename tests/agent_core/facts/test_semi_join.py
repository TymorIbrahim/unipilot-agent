"""The semi-join and the nested declarations -- pure, and previously untested here.

Ported from `tests/pending_supabase/test_sources.py`. Neither needs a database,
and neither had any coverage in the active suite: `test_dispatch.py` mentions the
semi-join only in a comment.

`_id IN {"fact": collection, "field": f}` is how the model fetches the records a
held collection points at. Two live evals stalled without it -- the model held 17
course ids and no way to reach the courses except `find(courses, limit=3000)` and
an in-memory join, which it never discovered. The natural spelling did not parse.

The nested declarations are what `optimize` walks: `semester_plans.semesters[]`
is the only nested source, and unnesting it twice is the only route to slots.
"""

from __future__ import annotations

import pytest


class TestNestedDeclarations:
    """`semester_plans` is the only source declaring nested structure, and it is
    the one `optimize` depends on. These pin the shape the route relies on."""

    def test_semesters_is_declared_as_an_array_of_sub_documents(self) -> None:
        from app.agent_core.facts.find import ArrayOf, Sub
        from app.agent_core.facts.sources import SEMESTER_PLANS

        semesters = SEMESTER_PLANS.fields["semesters"]
        assert isinstance(semesters, ArrayOf) and isinstance(semesters.element, Sub)
        # `order` is the slot index and `goalCredits` the capacity. Without both,
        # a plan unnests into slots that cannot be placed into.
        assert {"order", "goalCredits", "semesterCode"} <= set(semesters.element.fields)

    def test_nested_paths_are_reachable_by_name(self) -> None:
        """The unknown-field message lists these, so a model can find them."""
        from app.agent_core.facts.find import declared_paths
        from app.agent_core.facts.sources import SEMESTER_PLANS

        paths = declared_paths(SEMESTER_PLANS)
        assert "semesters.order" in paths
        assert "semesters.plannedCourses.courseNumber" in paths



class TestSemiJoin:
    """`_id IN {"fact": collection, "field": f}` -- fetch the records a held
    collection points at, without pulling a whole catalog to join in memory.

    Two live evals stalled here: the model held 17 course ids and had no way to
    fetch the courses they referenced except `find(courses, limit=3000)` then a
    join, which it never discovered. The natural spelling did not parse.
    """

    def _completed(self):
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        records = tuple(
            Record(fields={"courseId": Scalar(ScalarKind.IDENTIFIER, cid)}, basis=Basis.OFFICIAL_RECORD)
            for cid in ("507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012", "507f1f77bcf86cd799439011")
        )
        return HeldFact(
            value=Collection(records=records, completeness=Completeness(complete=True, total=3)),
            basis=Basis.OFFICIAL_RECORD,
        )

    def _resolve(self, predicate_json, facts):
        from app.agent_core.facts.codec import parse_predicate
        from app.agent_core.facts.dispatch import _resolve_fact_refs

        return _resolve_fact_refs(parse_predicate(predicate_json), facts)

    def test_it_resolves_to_the_distinct_set_of_field_values(self) -> None:
        from app.agent_core.facts.predicate import Comparison, Op

        resolved = self._resolve(
            {"path": "_id", "op": "in", "value": {"fact": "completed", "field": "courseId"}},
            {"completed": self._completed()},
        )
        assert isinstance(resolved, Comparison) and resolved.op is Op.IN
        # Three records, two distinct ids -- the duplicate collapses.
        assert sorted(s.value for s in resolved.value) == ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]

    def test_in_against_a_held_fact_without_a_field_is_a_repairable_error(self) -> None:
        from app.agent_core.facts.codec import ParseError

        with pytest.raises(ParseError) as caught:
            self._resolve({"path": "_id", "op": "in", "value": {"fact": "completed"}}, {"completed": self._completed()})
        assert "field" in str(caught.value)

    def test_a_scalar_op_on_a_single_record_extracts_the_one_value(self) -> None:
        """The shape the model reached for turn after turn: `course = {"fact":
        "next_course", "field": "courseNumber"}` where next_course holds one
        record. This is one-record extraction, like `only`."""
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.predicate import Comparison
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        one = HeldFact(
            value=Collection(
                records=(Record(fields={"courseNumber": Scalar(ScalarKind.IDENTIFIER, "00960211")}, basis=Basis.OFFICIAL_RECORD),),
                completeness=Completeness(complete=True, total=1),
            ),
            basis=Basis.OFFICIAL_RECORD,
        )
        resolved = self._resolve(
            {"path": "course", "op": "=", "value": {"fact": "next_course", "field": "courseNumber"}},
            {"next_course": one},
        )
        assert isinstance(resolved, Comparison)
        assert resolved.value.value == "00960211"

    def test_a_scalar_op_on_a_multi_record_fact_is_refused_with_guidance(self) -> None:
        from app.agent_core.facts.operators import ExpressionDefect

        resolved = self._resolve(
            {"path": "_id", "op": "=", "value": {"fact": "completed", "field": "courseId"}},
            {"completed": self._completed()},  # two distinct ids
        )
        assert isinstance(resolved, ExpressionDefect)
        assert "in" in resolved.message and "one" in resolved.message

    def test_a_missing_field_fails_closed_rather_than_shrinking_the_set(self) -> None:
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.operators import ExpressionDefect
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        held = HeldFact(
            value=Collection(
                records=(Record(fields={"other": Scalar(ScalarKind.IDENTIFIER, "x")}, basis=Basis.OFFICIAL_RECORD),),
                completeness=Completeness(complete=True, total=1),
            ),
            basis=Basis.OFFICIAL_RECORD,
        )
        resolved = self._resolve(
            {"path": "_id", "op": "in", "value": {"fact": "c", "field": "courseId"}}, {"c": held}
        )
        assert isinstance(resolved, ExpressionDefect)
        assert "courseId" in resolved.message
