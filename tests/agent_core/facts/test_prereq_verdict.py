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


class TestAPassClaimedWithAPronoun:
    """"you already passed it" is the same dangerous claim, wearing a pronoun.

    Live, asked "Am I eligible to take 00960211?":

        "No -- the course exists in the catalog, and you already passed it in 41
         recorded attempts, so you are not eligible to take it again."

    Checked against SQL: the student has NOT passed 00960211, and 41 is their
    TOTAL passed-course count -- a `find` on `passed_courses` filtered only by
    userId. Every number is real and the sentence is false, which is exactly
    what `check_claimed_pass_is_on_the_transcript` exists for. It missed because
    it was looking for digits next to the verb.

    Worse than a wrong answer: the same question asked in Hebrew answered
    "yes, you meet 1 of 1 groups". Two opposite verdicts, one of them telling a
    student not to register for a course they need.
    """

    LIVE = (
        "No — the course exists in the catalog, and you already passed it in 41 "
        "recorded attempts, so you are not eligible to take it again."
    )
    ASKED = "Am I eligible to take 00960211?"

    def kinds(self, text: str, passed, question: str) -> list[str]:
        from app.agent_core.facts.postconditions import (
            check_claimed_pass_is_on_the_transcript as check,
        )

        return [v.kind for v in check(text, passed, question)]

    def test_the_live_answer_is_refused(self) -> None:
        assert self.kinds(self.LIVE, ["00940314"] * 41, self.ASKED) == ["unearned_pass"]

    def test_a_true_claim_survives(self) -> None:
        assert self.kinds("You already passed it.", ["00960211"], self.ASKED) == []

    def test_two_courses_asked_is_never_guessed_at(self) -> None:
        """The pronoun's referent is only unambiguous when ONE course was asked."""
        assert self.kinds(
            "You already passed it.",
            ["00940314"],
            "Is 00960211 or 01040174 easier for me?",
        ) == []

    def test_the_hebrew_phrasing_is_caught(self) -> None:
        assert self.kinds("כבר עברת אותו.", ["00940314"] * 41, self.ASKED) == [
            "unearned_pass"
        ]

    def test_no_transcript_held_means_no_opinion(self) -> None:
        assert self.kinds(self.LIVE, [], self.ASKED) == []


class TestTheCatalogDenialIsAboutTheCatalog:
    """The exemption was twice too loose, and exempted the worst answer twice.

    A leading verdict -- "No -- the course exists in the catalog..." -- satisfied
    both "a negator within 30 characters of 'catalog'" and "a negator within 12
    characters before the phrase". The negation has to belong to the existence
    predicate, not merely appear near it.
    """

    def kinds(self, text: str) -> list[str]:
        from app.agent_core.facts.postconditions import (
            check_answer_does_not_narrate_the_catalog_lookup as check,
        )

        return [v.kind for v in check(text)]

    def test_a_leading_verdict_no_does_not_exempt_the_narration(self) -> None:
        assert self.kinds(
            "No — the course exists in the catalog, and you meet 0 of 1 groups."
        ) == ["narrates_the_catalog_lookup"]

    def test_hebrew_narration_is_caught(self) -> None:
        assert self.kinds("כן — הקורס קיים בקטלוג, ואתה עומד ב-1 מתוך 1.") == [
            "narrates_the_catalog_lookup"
        ]

    def test_a_real_denial_of_existence_still_survives(self) -> None:
        for text in (
            "00999999 is not in the catalog, so I cannot say anything about it.",
            "That course number does not exist in the catalog.",
            "I could not find 00999999 in the catalog.",
            "הקורס לא קיים בקטלוג.",
        ):
            assert self.kinds(text) == [], text


class TestNoRankingByAGradeNobodyHasEarned:
    """Live, asked "which courses next semester would raise my GPA the most?":

        "These are the courses next semester that would raise your GPA the most:
         - course 00960620 · credits 3.5
         - course 00960606 · credits 3
         - course 00970325 · credits 3 ..."

    The course list sorted by CREDITS, presented as a GPA-impact ranking. GPA
    impact is grade x credits and the grade is not a record -- it does not exist
    yet for any of them. The agent took the one field it had, ordered by it, and
    labelled the result something else.

    Every other gate passes it, which is why it needs its own: the credits are
    real derived facts, no digit was typed, no course was invented. Only the
    CLAIM ABOUT WHAT THE ORDERING MEANS is fabricated, and nothing else examines
    a claim carrying no number of its own.

    Worse than a wrong number because it is actionable -- a student reads the
    top of that list and registers, believing it was computed. The ground truth
    for this question says so outright: "inventing a ranking here would be the
    worst outcome".
    """

    ASKED = "Which courses next semester would raise my GPA the most?"
    SHIPPED = (
        "These are the courses next semester that would raise your GPA the most:\n"
        "- course 00960620 · credits 3.5\n- course 00960606 · credits 3\n"
        "- course 00970325 · credits 3"
    )

    def kinds(self, text: str, question: str) -> list[str]:
        from app.agent_core.facts.postconditions import (
            check_no_ranking_by_an_unearned_grade as check,
        )

        return [v.kind for v in check(text, question)]

    def test_the_invented_ranking_is_refused(self) -> None:
        assert self.kinds(self.SHIPPED, self.ASKED) == ["ranked_by_an_unearned_grade"]

    def test_saying_it_cannot_be_derived_passes(self) -> None:
        """The correct answer, and offering what CAN be done alongside it is
        encouraged rather than penalised."""
        honest = (
            "I can't work that out: raising a GPA depends on the grade you earn, and "
            "no grade exists yet for a course you have not taken. What I can tell you "
            "is that credits weight a grade's effect — 00960620 is 3.5 credits, "
            "00940704 is 1.5."
        )
        assert self.kinds(honest, self.ASKED) == []

    def test_a_hebrew_disclaimer_passes(self) -> None:
        assert self.kinds(
            "אי אפשר לגזור את זה: ההשפעה תלויה בציון שתקבל. 00960620, 00940704.",
            "איזה קורסים יעלו לי הכי הרבה את הממוצע?",
        ) == []

    def test_other_questions_are_untouched(self) -> None:
        for question, answer in (
            ("Plan my winter semester.", self.SHIPPED),
            ("What is my GPA?", "Your GPA is 74.45."),
            ("Which three courses did I do worst in?", "01030015, 03240053, 01040166."),
        ):
            assert self.kinds(answer, question) == [], question

    def test_one_course_is_not_an_ordering(self) -> None:
        assert self.kinds("Consider 00960620.", self.ASKED) == []
