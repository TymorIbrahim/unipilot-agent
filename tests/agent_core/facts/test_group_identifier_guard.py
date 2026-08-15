"""A prerequisite GROUP label must never be shown as if it were a course.

Groups are labelled `<course>.<n>` -- `00970800.0`, `00970800.1` -- which is
bookkeeping. A live answer read "the alternatives I derived are 00970800.0,
00970800.1", naming two things a student cannot register for instead of the four
course codes behind them.

Nothing else catches it. The tokens are slotted from a real fact, so the
grounding invariant passes them, and they LOOK like course codes to a reader --
which is what makes them worse than a visible error. So it is a post-condition:
the answer is refused and the reason handed back, the same way an impossible
grade is.
"""

from __future__ import annotations

from app.agent_core.facts.answer import Answer
from app.agent_core.facts.answer_verify import verify_answer
from app.agent_core.facts.postconditions import check_no_group_identifiers
from app.agent_core.facts.types import Basis


def _answer(text: str) -> Answer:
    return Answer(text=text, basis=Basis.OFFICIAL_RECORD, used=("edges",))


class TestGroupLabelsAreCaught:
    def test_the_live_failure_is_flagged(self) -> None:
        violations = check_no_group_identifiers(
            "You need 2 prerequisite groups. The alternatives I derived are "
            "00970800.0, 00970800.1."
        )
        assert violations and violations[0].kind == "group_identifier_shown"

    def test_the_message_names_what_to_do_instead(self) -> None:
        violations = check_no_group_identifiers("alternatives: 00970800.0")
        assert "requires" in violations[0].message, "the model needs the fix, not just the fault"

    def test_it_runs_on_a_non_plan_answer(self) -> None:
        """A prerequisite question is not a plan, and that is exactly where this
        was happening -- so it cannot live behind the plan-shaped checks."""
        violations = verify_answer(_answer("alternatives: 00970800.0, 00970800.1"), {}, "q")
        assert violations and violations[0].kind == "group_identifier_shown"


class TestRealAnswersAreNotDisturbed:
    def test_course_codes_pass(self) -> None:
        assert not check_no_group_identifiers(
            "any one of 00940423, 00940594, and any one of 00940424, 00940591."
        )

    def test_credits_and_gpa_pass(self) -> None:
        assert not check_no_group_identifiers(
            "You have completed 129.5 of 155 credits. Your GPA is 72.64."
        )

    def test_a_sentence_ending_in_a_course_code_passes(self) -> None:
        assert not check_no_group_identifiers("It requires one of 00940224, 00940226.")

    def test_a_sound_answer_verifies_clean(self) -> None:
        assert verify_answer(_answer("It requires one of 00940224, 00940226."), {}, "q") == []
