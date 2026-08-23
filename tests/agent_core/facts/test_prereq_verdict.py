"""An eligibility verdict must be computed, not asserted.

Live, and wrong in the direction that costs a student a semester:

    "Before you can take 00970135, you need to complete 00960324 first. I also
     checked 00960324 itself, and its prerequisites are not yet satisfied by
     your passed courses, so the chain stops there for now."

Checked against SQL rather than against the agent: `prerequisite_edges` gives
00960324 ONE group with two members, 00940314 and 00980413 -- alternatives, so
either satisfies it -- and `passed_courses` holds 00940314 at 57, above the 55
pass mark. The student could have registered that day.

The run had already fetched both halves and then wrote the verdict in prose
instead of computing it. That is the hole: the grounding invariant refuses a
typed DIGIT, so "you need 25.5 credits" cannot be invented, while "its
prerequisites are not satisfied" carries no number and costs nothing to make
up. The system's strongest guarantee does not reach claims without numbers, and
an eligibility verdict is exactly such a claim.
"""

from __future__ import annotations

from app.agent_core.facts.answer import Answer, HeldFact
from app.agent_core.facts.answer_verify import _satisfied_courses, verify_answer
from app.agent_core.facts.postconditions import check_prereq_verdict_matches_the_edges
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

I = ScalarKind.IDENTIFIER

# The live answer, verbatim.
SHIPPED = (
    "Before you can take 00970135, you need to complete 00960324 first. I also "
    "checked 00960324 itself, and its prerequisites are not yet satisfied by "
    "your passed courses, so the chain stops there for now."
)


def edges(*rows: tuple[str, str, str]) -> Collection:
    return Collection(
        records=tuple(
            Record(
                fields={
                    "course": Scalar(I, course),
                    "requires": Scalar(I, requires),
                    "group": Scalar(I, group),
                },
                basis=Basis.OFFICIAL_RECORD,
            )
            for course, requires, group in rows
        ),
        completeness=Completeness(complete=True, total=len(rows)),
    )


def passed(*codes: str) -> Collection:
    return Collection(
        records=tuple(
            Record(fields={"courseNumber": Scalar(I, code)}, basis=Basis.OFFICIAL_RECORD)
            for code in codes
        ),
        completeness=Completeness(complete=True, total=len(codes)),
    )


def held(value: Collection, derivation: str) -> HeldFact:
    return HeldFact(value, Basis.OFFICIAL_RECORD, derivation=derivation)


# Exactly what SQL returned for this student.
REAL_FACTS = {
    "blocker_prereqs": held(
        edges(
            ("00960324", "00940314", "00960324"),
            ("00960324", "00980413", "00960324"),
            ("00970135", "00960324", "00970135"),
        ),
        "read from prerequisite_edges",
    ),
    "my_passed_courses": held(
        passed("00940314", "00940412"), "read from passed_courses"
    ),
}


class TestReplayingTheGroupAlgebra:
    def test_one_passed_alternative_satisfies_the_group(self) -> None:
        satisfied = _satisfied_courses(REAL_FACTS)
        assert satisfied.get("00960324") == "00940314"

    def test_a_course_whose_own_requirement_is_unpassed_is_not_satisfied(self) -> None:
        """00970135 needs 00960324, which is not on the transcript."""
        assert "00970135" not in _satisfied_courses(REAL_FACTS)

    def test_every_group_must_be_met_not_just_one(self) -> None:
        """Two GROUPS are both mandatory; two members of one are alternatives."""
        facts = {
            "e": held(
                edges(("00900001", "00900010", "g1"), ("00900001", "00900020", "g2")),
                "read from prerequisite_edges",
            ),
            "p": held(passed("00900010"), "read from passed_courses"),
        }
        assert "00900001" not in _satisfied_courses(facts)

    def test_holding_no_edges_yields_no_opinion(self) -> None:
        assert _satisfied_courses({"p": held(passed("00940314"), "passed_courses")}) == {}


class TestTheShippedAnswerIsRefused:
    def test_the_live_failure_is_caught(self) -> None:
        violations = check_prereq_verdict_matches_the_edges(
            SHIPPED, _satisfied_courses(REAL_FACTS)
        )
        assert [v.kind for v in violations] == ["prereq_verdict_contradicts_the_edges"]

    def test_the_message_names_the_course_that_satisfies_it(self) -> None:
        """A reason the model cannot act on wastes the retry it costs."""
        message = check_prereq_verdict_matches_the_edges(
            SHIPPED, _satisfied_courses(REAL_FACTS)
        )[0].message
        assert "00960324" in message and "00940314" in message

    def test_it_reaches_through_verify_answer(self) -> None:
        answer = Answer(
            text=SHIPPED, basis=Basis.OFFICIAL_RECORD, used=(), citations=()
        )
        kinds = [v.kind for v in verify_answer(answer, REAL_FACTS, "")]
        assert "prereq_verdict_contradicts_the_edges" in kinds


class TestItDoesNotRefuseCorrectAnswers:
    def test_the_answer_the_prompt_asks_for_passes(self) -> None:
        text = (
            "You need 00960324 first, and you are already eligible for it "
            "because you passed 00940314."
        )
        assert check_prereq_verdict_matches_the_edges(
            text, _satisfied_courses(REAL_FACTS)
        ) == []

    def test_a_true_negative_about_a_different_course_passes(self) -> None:
        """The claim is scoped to the code NEAREST BEFORE it.

        An answer naming two courses -- one satisfied, one not -- must not be
        refused for the one it got right.
        """
        text = (
            "00960324 is takeable because you passed 00940314. "
            "00970135's prerequisites are not yet satisfied."
        )
        assert check_prereq_verdict_matches_the_edges(
            text, _satisfied_courses(REAL_FACTS)
        ) == []

    def test_no_claim_means_no_violation(self) -> None:
        assert check_prereq_verdict_matches_the_edges(
            "You have completed 129.5 credits.", _satisfied_courses(REAL_FACTS)
        ) == []

    def test_an_unknown_course_is_left_alone(self) -> None:
        """Silence beats guessing: the check only speaks where it holds edges."""
        text = "00999999's prerequisites are not satisfied."
        assert check_prereq_verdict_matches_the_edges(
            text, _satisfied_courses(REAL_FACTS)
        ) == []


class TestTheCatalogLookupIsNotTheAnswer:
    """A successful existence check is not news, and it crowds out the verdict.

    Two live answers to "will 00940412 be offered next spring?":

        "00940412 exists in the catalog, and yes."
        "Yes -- it exists in the catalog, and yes."

    The student typed the course number, so they know it exists. The verdict is
    the last word of both. The second is also why `_tidy_affirmations` could not
    repair the doubled yes: that repair spans separators, and here a whole
    clause sits between the two.

    Asked for in the prompt first, in the same voice as the rules that do hold,
    and the next runs narrated anyway.
    """

    def kinds(self, text: str) -> list[str]:
        from app.agent_core.facts.postconditions import (
            check_answer_does_not_narrate_the_catalog_lookup as check,
        )

        return [v.kind for v in check(text)]

    def test_the_two_live_answers_are_refused(self) -> None:
        for text in (
            "00940412 exists in the catalog, and yes.",
            "Yes -- it exists in the catalog, and yes.",
        ):
            assert self.kinds(text) == ["narrates_the_catalog_lookup"], text

    def test_a_failed_lookup_survives(self) -> None:
        """Existence IS the answer here, and reasoning past an empty lookup is
        how a course that does not exist got a confident eligibility verdict."""
        for text in (
            "00999999 is not in the catalog, so I cannot say anything about it.",
            "That course number does not exist in the catalog.",
            "I could not find 00999999 in the catalog.",
        ):
            assert self.kinds(text) == [], text

    def test_an_ordinary_answer_is_untouched(self) -> None:
        for text in (
            "Yes -- it has run every spring on record.",
            "You need 25.5 more credits.",
        ):
            assert self.kinds(text) == [], text

    def test_it_reaches_through_verify_answer(self) -> None:
        from app.agent_core.facts.answer_verify import verify_answer

        answer = Answer(
            text="00940412 exists in the catalog, and yes.",
            basis=Basis.OFFICIAL_RECORD,
            used=(),
            citations=(),
        )
        kinds = [v.kind for v in verify_answer(answer, {}, "")]
        assert "narrates_the_catalog_lookup" in kinds
