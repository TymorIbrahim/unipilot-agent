"""Ordinary term plans were never verified at all.

The whole post-condition layer was written for the MIN-GRADE planner, and
`_plan_collections` encoded that by requiring every record to carry both
`credits` and `min_grade`. An ordinary term plan -- courseNumber, title,
category, credits -- has no minimum, so it returned nothing and no check ran.

A live answer to "how many semesters will it take me to graduate" shipped:

    Winter — 23 credits
    - 00940704 · 1.5   - 00960578 · 2.5   - 00960606 · 3     ...

against a cap of 18, and three guards missed it in sequence:

  1. `_plan_collections` skipped the collection entirely (no min_grade)
  2. `check_term_load` would have been silent anyway -- it is a 40-credit
     SANITY ceiling, "a number no real semester reaches", by design
  3. nothing compared the plan to `maxCreditsPerSemester` at all

The third had no equivalent until the profile started being seeded, which made
the student's own cap reachable from the answer layer for the first time.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.answer_verify import _plan_collections, verify_answer
from app.agent_core.facts.postconditions import (
    MAX_TERM_CREDITS,
    check_term_load,
    check_term_within_cap,
)
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
CAP = 18.0

# The eight courses that really shipped in that winter term.
SHIPPED = [1.5, 2.5, 3.0, 3.5, 2.5, 3.0, 3.0, 4.0]  # 23.0


def _plan(credits: list[float], *, with_min_grade: bool = False) -> Collection:
    records = []
    for index, amount in enumerate(credits):
        fields = {
            "number": Scalar(ScalarKind.IDENTIFIER, f"009{index:05d}"),
            "credits": Scalar(Q, amount),
            "name": Scalar(ScalarKind.TEXT, "a course"),
        }
        if with_min_grade:
            fields["min_grade"] = Scalar(Q, 80.0)
        records.append(Record(fields=fields, basis=Basis.SIMULATED))
    return Collection(
        records=tuple(records), completeness=Completeness(complete=True, total=len(records))
    )


class _Answer:
    def __init__(self, used: list[str]) -> None:
        self.used = used
        self.text = "Winter — a plan\n{winter:detail}"


def _facts(credits: list[float], *, with_min_grade: bool = False, cap: float | None = CAP) -> dict:
    facts = {"winter": HeldFact(value=_plan(credits, with_min_grade=with_min_grade),
                                basis=Basis.SIMULATED)}
    if cap is not None:
        # Seeded from student_profiles at the start of every real run.
        facts["max_credits_per_semester"] = HeldFact(
            value=Scalar(Q, cap), basis=Basis.OFFICIAL_RECORD
        )
    return facts


class TestAnOrdinaryPlanIsSeenAtAll:
    def test_a_plan_without_min_grade_is_collected(self) -> None:
        found = _plan_collections(_Answer(["winter"]), _facts(SHIPPED))
        assert len(found) == 1, "an ordinary term plan was invisible to every check"

    def test_its_courses_carry_no_minimum(self) -> None:
        (_, courses), = _plan_collections(_Answer(["winter"]), _facts(SHIPPED))
        assert all(c.min_grade is None for c in courses)

    def test_a_min_grade_plan_still_works(self) -> None:
        (_, courses), = _plan_collections(
            _Answer(["winter"]), _facts(SHIPPED, with_min_grade=True)
        )
        assert all(c.min_grade == 80.0 for c in courses)

    def test_a_collection_without_credits_is_still_ignored(self) -> None:
        """A prerequisite list or an offerings table is not a plan."""
        records = (Record(fields={"number": Scalar(ScalarKind.IDENTIFIER, "00940224")},
                          basis=Basis.OFFICIAL_RECORD),)
        facts = {"prereqs": HeldFact(
            value=Collection(records=records, completeness=Completeness(complete=True, total=1)),
            basis=Basis.OFFICIAL_RECORD)}
        assert _plan_collections(_Answer(["prereqs"]), facts) == []


class TestTheStudentsOwnCap:
    def test_the_shipped_term_is_caught(self) -> None:
        violations = verify_answer(_Answer(["winter"]), _facts(SHIPPED), "How many semesters?")
        assert [v.kind for v in violations] == ["term_over_cap"]

    def test_the_refusal_names_the_likely_cause(self) -> None:
        """A model told only "too many credits" drops a course; the real fault
        was asking for the same term name twice and merging the two."""
        violations = check_term_within_cap(23.0, CAP, "winter")
        assert "same term name twice" in violations[0].message
        assert "max_credits" in violations[0].message

    @pytest.mark.parametrize("credits", [[18.0], [17.5], [1.5, 2.5, 3.0, 3.5, 2.5, 3.0]])
    def test_a_term_within_the_cap_is_silent(self, credits: list[float]) -> None:
        assert not verify_answer(_Answer(["winter"]), _facts(credits), "How many semesters?")

    def test_exactly_at_the_cap_is_allowed(self) -> None:
        """A full load is a full load, not a violation."""
        assert not check_term_within_cap(18.0, 18.0, "winter")

    def test_no_cap_fact_means_no_cap_check(self) -> None:
        """Unverifiable is not the same as violated -- a run that never learned
        the cap must not be blocked."""
        violations = verify_answer(
            _Answer(["winter"]), _facts(SHIPPED, cap=None), "How many semesters?"
        )
        assert not [v for v in violations if v.kind == "term_over_cap"]

    def test_a_nonsense_cap_is_ignored(self) -> None:
        assert not check_term_within_cap(23.0, 0.0, "winter")


class TestTheTwoLoadChecksAreDifferent:
    def test_the_sanity_ceiling_does_not_cover_the_cap(self) -> None:
        """23 credits is over this student's limit and nowhere near the 40-credit
        overflow ceiling. Neither check subsumes the other."""
        assert not check_term_load(23.0, "winter")
        assert check_term_within_cap(23.0, CAP, "winter")

    def test_the_overflow_ceiling_still_fires(self) -> None:
        assert check_term_load(MAX_TERM_CREDITS + 1, "winter")

    def test_an_overflow_term_trips_both(self) -> None:
        kinds = {v.kind for v in verify_answer(
            _Answer(["winter"]), _facts([83.0]), "How many semesters?")}
        assert kinds == {"term_load", "term_over_cap"}


class TestTheCapIsCheckedPerTermNotPerCollection:
    """A `:detail` collection is not always ONE term.

    A multi-semester answer slots a SUMMARY -- one row per term, each carrying
    that term's total. Summing those and comparing to a per-semester cap
    compares the whole degree plan to one semester's limit, and it refused:

        "term_summary totals 38.5 credits, over this student's limit of 18"

    where 38.5 was 16 + 13.5 + 9 across three terms, every one of them legal.
    The model had already spent a rejection on an unrelated refusal, spent its
    remaining ones here, and the run returned nothing at all -- so a check added
    to stop a wrong answer was instead preventing a right one.
    """

    def _facts(self, rows: list[tuple[str, float, str | None]]) -> dict:
        records = tuple(
            Record(
                fields={
                    "number": Scalar(ScalarKind.IDENTIFIER, code),
                    "credits": Scalar(Q, credits),
                    **({"term": Scalar(ScalarKind.TEXT, term)} if term else {}),
                },
                basis=Basis.SIMULATED,
            )
            for code, credits, term in rows
        )
        return {
            "plan": HeldFact(
                value=Collection(
                    records=records,
                    completeness=Completeness(complete=True, total=len(records)),
                ),
                basis=Basis.SIMULATED,
            ),
            "max_credits_per_semester": HeldFact(
                value=Scalar(Q, CAP), basis=Basis.OFFICIAL_RECORD
            ),
        }

    def test_a_multi_term_summary_is_not_summed(self) -> None:
        facts = self._facts([("t1", 16.0, "2026-1"), ("t2", 13.5, "2026-2"), ("t3", 9.0, "2026-3")])
        assert not verify_answer(_Answer(["plan"]), facts, "How many semesters?")

    def test_one_term_over_the_cap_is_still_caught_inside_a_summary(self) -> None:
        facts = self._facts([("t1", 22.0, "2026-1"), ("t2", 10.0, "2026-2")])
        kinds = [v.kind for v in verify_answer(_Answer(["plan"]), facts, "How many semesters?")]
        assert kinds == ["term_over_cap"]

    def test_the_violation_names_the_offending_term(self) -> None:
        facts = self._facts([("t1", 22.0, "2026-1"), ("t2", 10.0, "2026-2")])
        message = verify_answer(_Answer(["plan"]), facts, "q")[0].message
        assert "2026-1" in message and "2026-2" not in message

    def test_a_single_term_of_courses_is_unchanged(self) -> None:
        """The shape the cap was written for: 8 courses, all one term, 23
        credits. Grouping must not lose it."""
        rows = [(f"009{i:05d}", c, "winter")
                for i, c in enumerate([1.5, 2.5, 3.0, 3.5, 2.5, 3.0, 3.0, 4.0])]
        kinds = [v.kind for v in verify_answer(_Answer(["plan"]), self._facts(rows), "q")]
        assert kinds == ["term_over_cap"]

    def test_rows_with_no_term_are_treated_as_one(self) -> None:
        rows = [(f"009{i:05d}", c, None)
                for i, c in enumerate([1.5, 2.5, 3.0, 3.5, 2.5, 3.0, 3.0, 4.0])]
        kinds = [v.kind for v in verify_answer(_Answer(["plan"]), self._facts(rows), "q")]
        assert kinds == ["term_over_cap"]
