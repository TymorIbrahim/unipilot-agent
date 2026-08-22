"""Telling a student they passed a course they have not taken.

Found by asking the deployed agent something the eval never asks -- a
counterfactual:

    "If I fail 00970800, how does that change my graduation timeline?"

    You already passed 00970800 in 41 transcript rows, so failing it on a
    re-take would not change your graduation timeline. A passed course does not
    add any further credit when repeated.

00970800 is one of this student's SIX REMAINING mandatory courses. They have not
taken it. The advice is not merely wrong, it is the kind a student acts on.

Every gate passed, correctly. The `find` carried no course filter --

    find(passed_courses where userId = me) -> 41 records

-- so the fact held all 41 passed courses, and 41 is a real, official,
non-empty, correctly-derived number. The grounding invariant guarantees a digit
came from a fact. It cannot guarantee the SENTENCE means what the fact means,
because the fact's name is prose the model wrote, and here it wrote
`passed_00970800`.

That is the failure `HeldFact.derivation` was introduced for, one step on.
Showing "41 (read from passed_courses)" beside the answer lets a developer see
it; the student reading "you already passed 00970800" cannot.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.answer_verify import _passed_codes, verify_answer
from app.agent_core.facts.postconditions import check_claimed_pass_is_on_the_transcript
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

# The six mandatory courses this student has NOT taken, and three they have.
NOT_TAKEN = "00970800"
TAKEN = ("00940314", "00940345", "00960210")

LIVE_ANSWER = (
    "You already passed 00970800 in 41 transcript rows, so failing it on a "
    "re-take would not change your graduation timeline."
)


def _passed(*codes: str) -> Collection:
    records = tuple(
        Record(fields={"courseNumber": Scalar(ScalarKind.IDENTIFIER, code)},
               basis=Basis.OFFICIAL_RECORD)
        for code in codes
    )
    return Collection(records=records,
                      completeness=Completeness(complete=True, total=len(records)))


class TestTheCheck:
    def test_the_live_answer_is_refused(self) -> None:
        violations = check_claimed_pass_is_on_the_transcript(LIVE_ANSWER, TAKEN)
        assert [v.kind for v in violations] == ["unearned_pass"]

    def test_the_refusal_names_the_actual_cause(self) -> None:
        """A model told only "wrong" re-runs the same unfiltered find."""
        message = check_claimed_pass_is_on_the_transcript(LIVE_ANSWER, TAKEN)[0].message
        assert "filtered only by userId" in message
        assert NOT_TAKEN in message

    def test_a_true_claim_passes(self) -> None:
        assert check_claimed_pass_is_on_the_transcript(
            "You already passed 00940314, so you are eligible.", TAKEN) == []

    def test_no_transcript_held_means_no_opinion(self) -> None:
        """Unverifiable is not violated -- the rule every check here follows."""
        assert check_claimed_pass_is_on_the_transcript(LIVE_ANSWER, ()) == []

    def test_a_future_pass_is_not_a_claim(self) -> None:
        """"once you pass X" is advice, not an assertion about the record."""
        for answer in ("Once you pass 00970800 you can take 00960324.",
                       "You need to pass 00970800 first.",
                       "If you pass 00970800 this winter, you finish on time."):
            assert check_claimed_pass_is_on_the_transcript(answer, TAKEN) == [], answer

    def test_a_denial_is_not_a_claim(self) -> None:
        assert check_claimed_pass_is_on_the_transcript(
            "You have not taken 00970800 yet.", TAKEN) == []


class TestFindingTheTranscript:
    """Read off the DERIVATION, never the fact's name -- the name is the thing
    that lied."""

    def test_codes_come_from_a_passed_courses_fact(self) -> None:
        facts = {
            "anything_at_all": HeldFact(value=_passed(*TAKEN), basis=Basis.OFFICIAL_RECORD,
                                        derivation="read from passed_courses matching a filter"),
        }
        assert sorted(_passed_codes(facts)) == sorted(TAKEN)

    def test_the_misleading_name_does_not_matter(self) -> None:
        """The live fact was called `passed_00970800` and held all 41."""
        facts = {
            "passed_00970800": HeldFact(value=_passed(*TAKEN), basis=Basis.OFFICIAL_RECORD,
                                        derivation="read from passed_courses matching a filter"),
        }
        assert NOT_TAKEN not in _passed_codes(facts)

    def test_an_unrelated_collection_is_not_a_transcript(self) -> None:
        facts = {
            "track": HeldFact(value=_passed(NOT_TAKEN), basis=Basis.WIKI_DERIVED,
                              derivation="read from track_courses matching a filter"),
        }
        assert _passed_codes(facts) == []

    def test_nothing_held_is_empty(self) -> None:
        assert _passed_codes({}) == []


class TestThroughVerifyAnswer:
    QUESTION = "If I fail 00970800, how does that change my graduation timeline?"

    def _facts(self) -> dict:
        return {"passed_00970800": HeldFact(
            value=_passed(*TAKEN), basis=Basis.OFFICIAL_RECORD,
            derivation="read from passed_courses matching a filter")}

    class _Answer:
        used = ["passed_00970800"]
        text = LIVE_ANSWER

    def test_the_live_answer_would_now_be_sent_back(self) -> None:
        violations = verify_answer(self._Answer(), self._facts(), self.QUESTION)
        assert "unearned_pass" in [v.kind for v in violations]

    def test_a_correct_answer_still_ships(self) -> None:
        class _Ok:
            used = ["passed_00970800"]
            text = "You have not taken 00970800, so failing it would add a semester."

        assert verify_answer(_Ok(), self._facts(), self.QUESTION) == []
