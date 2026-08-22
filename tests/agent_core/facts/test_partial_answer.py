"""A run out of budget was throwing away everything it had learned.

`config.py` has promised for a long time that the agent "ships a grounded
partial answer rather than being killed". It never did. Every non-answer --
refused, stalled, exhausted -- became the same sentence:

    I wasn't able to work that out from your records with confidence.

Measured on production after the ceiling was corrected to 60s: "How many
semesters will it take me to graduate, and what should I take each semester?"
returned exactly that at 46.7s, having already derived the student's completed
credits, the credits they still need, and their per-semester cap. All three were
in hand and none reached the student.

It matters more at a 60s ceiling than it would have at 300s, because the
questions that exhaust the budget are the substantial ones -- and those are the
runs holding the most when the clock stops.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.loop import LoopResult
from app.agent_core.facts.service import _COULD_NOT_ANSWER, to_advice
from app.agent_core.facts.types import Basis, Collection, Completeness, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _facts(**kw) -> dict:
    return {n: HeldFact(value=v, basis=Basis.OFFICIAL_RECORD) for n, v in kw.items()}


def _exhausted(**facts) -> LoopResult:
    return LoopResult(outcome="exhausted", reason="the budget was spent", facts=_facts(**facts))


class TestWhatItSalvages:
    def test_the_live_case_reports_what_it_held(self) -> None:
        advice = to_advice(_exhausted(
            completed_credits=Scalar(Q, 129.5),
            credits_needed=Scalar(Q, 25.5),
            max_credits_per_semester=Scalar(Q, 18.0),
        ))
        assert "129.5" in advice.answer
        assert "25.5" in advice.answer
        assert advice.answer != _COULD_NOT_ANSWER

    def test_it_says_plainly_that_it_did_not_finish(self) -> None:
        advice = to_advice(_exhausted(completed_credits=Scalar(Q, 129.5)))
        assert "ran out of time" in advice.answer
        assert "finish" in advice.answer

    def test_a_whole_number_is_not_shown_as_a_float(self) -> None:
        advice = to_advice(_exhausted(credits_needed=Scalar(Q, 18.0)))
        assert "18" in advice.answer and "18.0" not in advice.answer

    def test_the_outcome_is_still_reported_honestly(self) -> None:
        """A partial is not a success -- the status must not claim one."""
        advice = to_advice(_exhausted(completed_credits=Scalar(Q, 129.5)))
        assert advice.outcome == "exhausted"
        assert advice.status == "incomplete"


class TestWhatItRefusesToSalvage:
    def test_seeded_identity_alone_is_not_a_partial(self) -> None:
        """`me` and the profile columns are seeded before the run starts.
        Echoing them back restates the question rather than answering any of
        it."""
        advice = to_advice(_exhausted(
            me=Scalar(I, "6a578a2da43a2cfe1bcc791c"),
            program_slug=Scalar(I, "track-information-systems-engineering"),
            catalog_year=Scalar(Q, 2025.0),
            current_semester=Scalar(I, "2025-2"),
            max_credits_per_semester=Scalar(Q, 18.0),
        ))
        assert advice.answer == _COULD_NOT_ANSWER

    def test_a_run_holding_nothing_says_so(self) -> None:
        assert to_advice(_exhausted()).answer == _COULD_NOT_ANSWER

    def test_collections_are_not_dumped_at_the_student(self) -> None:
        """A 49-course curriculum is not a partial answer, it is a data dump."""
        result = LoopResult(outcome="exhausted", facts={
            "track": HeldFact(value=Collection(records=(), completeness=Completeness(True, 0)),
                              basis=Basis.WIKI_DERIVED),
        })
        assert to_advice(result).answer == _COULD_NOT_ANSWER

    def test_nothing_is_combined_or_inferred(self) -> None:
        """Combining is the step that ran out of time; doing it here would be
        inventing the answer the loop declined to give."""
        advice = to_advice(_exhausted(
            credits_needed=Scalar(Q, 25.5), completed_credits=Scalar(Q, 129.5)))
        assert "2 semesters" not in advice.answer
        assert "semesters:" not in advice.answer.lower()


class TestTheOtherOutcomesAreUnchanged:
    def test_an_answered_run_still_returns_its_answer(self) -> None:
        from app.agent_core.facts.answer import Answer

        result = LoopResult(outcome="answered",
                            answer=Answer(text="You have completed 129.5 credits.",
                                          basis=Basis.OFFICIAL_RECORD, used=(),
                                          citations=()),
                            facts=_facts(completed_credits=Scalar(Q, 129.5)))
        assert to_advice(result).answer == "You have completed 129.5 credits."

    def test_a_decline_still_uses_the_models_words(self) -> None:
        result = LoopResult(outcome="declined", reason="Not about your studies.",
                            facts=_facts(completed_credits=Scalar(Q, 129.5)))
        assert to_advice(result).answer == "Not about your studies."


class TestTheCreditStandingIsSeededButWorthReporting:
    """`SEEDED_FACT_NAMES` and `IDENTITY_FACTS` split because their two callers
    want different things.

    The decline guard asks "did the model FETCH anything", so everything the
    route seeded belongs outside it -- including the credit standing, or a
    weather question would look like it had records in hand.

    A partial answer asks "what did the run ESTABLISH", and there the credit
    standing is exactly what a student wants when the clock ran out. "You have
    completed 129.5 of 155 credits, 25.5 to go" is a real answer to part of the
    question; "your id is 6a578a..." is not.
    """

    def test_the_credit_standing_survives_into_a_partial(self) -> None:
        advice = to_advice(_exhausted(
            credits_completed=Scalar(Q, 129.5),
            credits_required=Scalar(Q, 155.0),
            credits_needed=Scalar(Q, 25.5),
        ))
        assert "129.5" in advice.answer
        assert "155" in advice.answer
        assert "25.5" in advice.answer

    def test_identity_is_still_excluded_beside_it(self) -> None:
        advice = to_advice(_exhausted(
            me=Scalar(I, "6a578a2da43a2cfe1bcc791c"),
            max_credits_per_semester=Scalar(Q, 18.0),
            credits_needed=Scalar(Q, 25.5),
        ))
        assert "25.5" in advice.answer
        assert "6a578a" not in advice.answer
        assert "18" not in advice.answer

    def test_identity_alone_is_still_not_a_partial(self) -> None:
        from app.agent_core.facts.service import IDENTITY_FACTS, SEEDED_FACT_NAMES

        assert IDENTITY_FACTS < SEEDED_FACT_NAMES, "the credit standing is seeded too"
        advice = to_advice(_exhausted(
            me=Scalar(I, "6a578a"), max_credits_per_semester=Scalar(Q, 18.0)))
        assert advice.answer == _COULD_NOT_ANSWER
