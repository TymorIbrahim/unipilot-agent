"""The plan was never checked against what the degree still needs.

Every other post-condition is PER TERM. On 2026-08-20 two live runs of "how many
semesters will it take me to graduate", minutes apart on identical data, scored
`correct` and answered **2** and **4**:

    run 0   10 courses   28.0 credits   ->  2 semesters
    run 1   13 courses   38.5 credits   ->  4 semesters

Every term inside run 1 was legal -- 16, 12, 7, 3.5 against an 18 cap -- so
`check_term_load` and `check_term_within_cap` both passed it, and nothing else
looked at the plan whole.

The data, checked by SQL rather than against the agent:

    completed (creditsCounted, passed)   129.5
    degree requires                      155.0
    remaining REQUIREMENT                 25.5
    remaining track courses    21 courses  50.0   (17.5 mandatory + 32.5 elective)

The student must take the 6 mandatory courses and then ANY 8 credits of the
32.5 on offer. Run 1 scheduled all 21, which is the "remaining means two
different things" defect one layer further in than where it was last fixed: not
in what the answer REPORTS, but in what the planner is FED.

The threshold is deliberately loose. Overshoot alone is normal -- courses are
indivisible, so 28 credits against a 25.5 requirement is simply the cost of
finishing, and run 0 is a correct answer. Only an overshoot that changes the
term count is a violation, because that is the one that reaches the student as a
wrong number.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.answer_verify import _remaining_required, verify_answer
from app.agent_core.facts.postconditions import check_plan_within_requirement
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
REQUIRED = 155.0
COMPLETED = 129.5
REMAINING = 25.5

# The two live plans, as their per-term totals.
RUN_0_TERMS = [16.0, 12.0]                # 28.0 -> answered 2
RUN_1_TERMS = [16.0, 12.0, 7.0, 3.5]      # 38.5 -> answered 4


class TestTheCheckItself:
    def test_the_four_semester_plan_is_caught(self) -> None:
        violations = check_plan_within_requirement(38.5, REMAINING, CAP)
        assert [v.kind for v in violations] == ["plan_exceeds_requirement"]

    def test_the_two_semester_plan_is_not(self) -> None:
        """28 credits against a 25.5 requirement still finishes in 2 terms."""
        assert check_plan_within_requirement(28.0, REMAINING, CAP) == []

    def test_a_plan_that_exactly_meets_the_requirement_passes(self) -> None:
        assert check_plan_within_requirement(REMAINING, REMAINING, CAP) == []

    def test_an_under_target_plan_passes(self) -> None:
        """Planning only the next term is a legitimate answer to a different
        question, and must not be flagged for being short."""
        assert check_plan_within_requirement(16.0, REMAINING, CAP) == []

    def test_the_message_names_the_number_to_stop_at(self) -> None:
        message = check_plan_within_requirement(38.5, REMAINING, CAP)[0].message
        assert "25.5" in message
        assert "38.5" in message
        assert "MANDATORY" in message, "a model told only 'too many' drops the wrong courses"

    def test_the_message_names_the_cost_in_semesters(self) -> None:
        """The harm is the term count, so the refusal says so.

        Three, not the four the live run reported: this check measures the terms
        the scheduled CREDITS force (ceil(38.5/18)), which is the part the plan's
        own contents are responsible for. The live answer reached four because
        `plan_term` spread those credits across four terms once offerings were
        applied -- and spreading like that is legitimate, so it is deliberately
        not what the threshold is built on. Counting the plan's actual terms
        instead would refuse a correct three-term answer that the ground truth
        explicitly allows."""
        message = check_plan_within_requirement(38.5, REMAINING, CAP)[0].message
        assert "2 semester(s) to 3" in message

    def test_a_missing_cap_skips_rather_than_guesses(self) -> None:
        assert check_plan_within_requirement(38.5, REMAINING, 0.0) == []

    def test_a_missing_requirement_skips(self) -> None:
        assert check_plan_within_requirement(38.5, 0.0, CAP) == []


class TestFindingTheRequirement:
    """`remaining` is the one word here that means two things, so the
    subtraction is preferred over any fact that merely claims the name."""

    def test_it_is_derived_from_required_minus_completed(self) -> None:
        facts = {
            "credits_required": HeldFact(value=Scalar(Q, REQUIRED), basis=Basis.OFFICIAL_RECORD),
            "completed_credits": HeldFact(value=Scalar(Q, COMPLETED), basis=Basis.OFFICIAL_RECORD),
        }
        assert _remaining_required(facts) == REMAINING

    def test_the_subtraction_beats_a_misnamed_remaining_fact(self) -> None:
        """The live failure: `remaining_credits` held 50.0 -- the credits still
        on OFFER -- while the student needed 25.5. Trusting the name would let
        the 38.5-credit plan through, since 38.5 < 50."""
        facts = {
            "credits_required": HeldFact(value=Scalar(Q, REQUIRED), basis=Basis.OFFICIAL_RECORD),
            "completed_credits": HeldFact(value=Scalar(Q, COMPLETED), basis=Basis.OFFICIAL_RECORD),
            "remaining_credits": HeldFact(value=Scalar(Q, 50.0), basis=Basis.OFFICIAL_RECORD),
        }
        assert _remaining_required(facts) == REMAINING

    def test_a_named_fact_is_used_when_the_subtraction_is_unavailable(self) -> None:
        facts = {"remaining_credits": HeldFact(value=Scalar(Q, REMAINING),
                                               basis=Basis.OFFICIAL_RECORD)}
        assert _remaining_required(facts) == REMAINING

    def test_nothing_held_means_no_opinion(self) -> None:
        assert _remaining_required({}) is None

    def test_a_completed_total_above_the_requirement_is_not_negative(self) -> None:
        facts = {
            "credits_required": HeldFact(value=Scalar(Q, 100.0), basis=Basis.OFFICIAL_RECORD),
            "completed_credits": HeldFact(value=Scalar(Q, 120.0), basis=Basis.OFFICIAL_RECORD),
        }
        assert _remaining_required(facts) is None


def _summary(term_credits: list[float]) -> Collection:
    """A multi-term answer's summary collection -- one row per term."""
    records = tuple(
        Record(
            fields={
                "term": Scalar(ScalarKind.IDENTIFIER, f"2026-{index + 1}"),
                "credits": Scalar(Q, amount),
            },
            basis=Basis.SIMULATED,
        )
        for index, amount in enumerate(term_credits)
    )
    return Collection(records=records,
                      completeness=Completeness(complete=True, total=len(records)))


class _Answer:
    def __init__(self, used: list[str], text: str = "It will take some semesters.\n{plan:detail}"):
        self.used = used
        self.text = text


def _facts(term_credits: list[float], **extra: float) -> dict:
    facts = {
        "plan": HeldFact(value=_summary(term_credits), basis=Basis.SIMULATED),
        "max_credits_per_semester": HeldFact(value=Scalar(Q, CAP), basis=Basis.OFFICIAL_RECORD),
        "credits_required": HeldFact(value=Scalar(Q, REQUIRED), basis=Basis.OFFICIAL_RECORD),
        "completed_credits": HeldFact(value=Scalar(Q, COMPLETED), basis=Basis.OFFICIAL_RECORD),
    }
    for name, value in extra.items():
        facts[name] = HeldFact(value=Scalar(Q, value), basis=Basis.OFFICIAL_RECORD)
    return facts


class TestThroughVerifyAnswer:
    QUESTION = "How many semesters will it take me to graduate?"

    def test_run_1_is_refused(self) -> None:
        violations = verify_answer(_Answer(["plan"]), _facts(RUN_1_TERMS), self.QUESTION)
        assert "plan_exceeds_requirement" in [v.kind for v in violations]

    def test_run_0_ships(self) -> None:
        """The correct answer of the pair must still pass, every check."""
        assert verify_answer(_Answer(["plan"]), _facts(RUN_0_TERMS), self.QUESTION) == []

    def test_the_per_term_checks_still_fire(self) -> None:
        """A 23-credit term is over the cap whatever the total comes to."""
        violations = verify_answer(_Answer(["plan"]), _facts([23.0]), self.QUESTION)
        assert "term_over_cap" in [v.kind for v in violations]

    def test_a_summary_and_a_listing_are_not_added_together(self) -> None:
        """A multi-term answer slots BOTH the per-term summary and the
        course-by-course listing. They describe the same credits twice, so
        summing every collection would double a correct plan into a violation."""
        facts = _facts(RUN_0_TERMS)
        facts["listing"] = HeldFact(value=_summary(RUN_0_TERMS), basis=Basis.SIMULATED)
        assert verify_answer(_Answer(["plan", "listing"]), facts, self.QUESTION) == []

    def test_without_the_requirement_the_check_skips(self) -> None:
        """Unverifiable is not the same as violated: an answer we cannot judge
        ships, which is the behaviour every other check here has."""
        facts = _facts(RUN_1_TERMS)
        del facts["credits_required"]
        del facts["completed_credits"]
        assert [v.kind for v in verify_answer(_Answer(["plan"]), facts, self.QUESTION)] == []
