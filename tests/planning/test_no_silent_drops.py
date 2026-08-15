"""Every candidate handed to the planner must come back accounted for.

The catalog tells the model `plan_term` returns the courses actually PLACED
"plus the ones that did not fit and why". Already-completed candidates were
skipped from both lists, so a model that handed over a whole curriculum -- the
obvious thing to do -- got a plan that simply did not mention them. Measured
live: six passed courses neither placed nor reported, indistinguishable from
candidates that were lost.

A silent omission is the one outcome a student cannot interrogate. "Why isn't X
in my plan" had no answer.
"""

from __future__ import annotations

from app.planning.term_plan import _unscheduled


class _Candidate:
    def __init__(self, number: str, course_id: str) -> None:
        self.course = {"_id": course_id, "courseNumber": number}
        self.category = "mandatory"


class TestNothingIsDroppedSilently:
    def test_a_completed_candidate_is_reported_not_omitted(self) -> None:
        rows = _unscheduled(
            [_Candidate("00940224", "id-1")],
            placed_ids=set(),
            completed_course_ids={"id-1"},
            satisfied_numbers={"00940224"},
            last_reason={},
        )
        assert [r["courseNumber"] for r in rows] == ["00940224"]
        assert "Already completed" in rows[0]["reason"]

    def test_a_completed_candidate_matched_only_by_number_is_reported(self) -> None:
        """The placement loop skips by id OR number, and 28% of transcript rows
        reference a catalog course that no longer exists -- so number-only
        matches are the common case, not the exotic one."""
        rows = _unscheduled(
            [_Candidate("00940224", "stale-id")],
            placed_ids=set(),
            completed_course_ids=set(),
            satisfied_numbers={"00940224"},
            last_reason={},
        )
        assert len(rows) == 1
        assert "Already completed" in rows[0]["reason"]

    def test_every_candidate_appears_exactly_once(self) -> None:
        candidates = [
            _Candidate("A", "id-a"),  # placed
            _Candidate("B", "id-b"),  # completed
            _Candidate("C", "id-c"),  # did not fit, with a reason
            _Candidate("D", "id-d"),  # did not fit, no reason recorded
        ]
        rows = _unscheduled(
            candidates,
            placed_ids={"id-a"},
            completed_course_ids={"id-b"},
            satisfied_numbers={"B"},
            last_reason={"C": "Exceeds the term credit cap"},
        )
        reported = {r["courseNumber"] for r in rows}
        assert reported == {"B", "C", "D"}, "placed courses belong to the plan, not this list"
        by_number = {r["courseNumber"]: r["reason"] for r in rows}
        assert "Already completed" in by_number["B"]
        assert by_number["C"] == "Exceeds the term credit cap"
        assert by_number["D"] == "Did not fit the requested term(s)"


class TestExistingBehaviourHolds:
    def test_a_placed_course_is_not_also_reported_unscheduled(self) -> None:
        rows = _unscheduled(
            [_Candidate("A", "id-a")],
            placed_ids={"id-a"},
            completed_course_ids=set(),
            satisfied_numbers=set(),
            last_reason={},
        )
        assert rows == []

    def test_a_duplicate_candidate_number_is_reported_once(self) -> None:
        rows = _unscheduled(
            [_Candidate("A", "id-1"), _Candidate("A", "id-2")],
            placed_ids=set(),
            completed_course_ids=set(),
            satisfied_numbers=set(),
            last_reason={},
        )
        assert len(rows) == 1
