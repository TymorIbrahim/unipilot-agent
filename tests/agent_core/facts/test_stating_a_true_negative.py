"""A true negative about a thing that does not exist could not be said.

The system prompt orders it explicitly:

    ASKED ABOUT A NAMED COURSE, CONFIRM IT EXISTS FIRST ... Nothing back means
    the code is not in the catalog, and NO conclusion about it is available --
    say that.

And `resolve_answer` refused every phrasing. Live, asked "Am I eligible for
course 00999999?", the agent searched the catalog twice, wrote three correct
answers, had all three rejected as "every fact the answer cites is empty", and
returned the give-up sentence to the student.

The rule itself is right and stays: "I can't determine: (none), (none)" is a
non-answer wearing the shape of a verified one, and it shipped once. What
separates that from a real negative is not completeness -- the comment in
`answer.py` says so, correctly -- it is HOW the fact is cited. `{name}` over an
empty collection renders "(none)" and asserts nothing. `{name:count}` renders
"0": a real derived number, about a real query, and the only way to state that
a thing was looked for and not found.
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact, Ungrounded, resolve_answer
from app.agent_core.facts.types import (
    Basis, Collection, Completeness, Record, Scalar, ScalarKind,
)

Q = ScalarKind.QUANTITY


def _empty() -> Collection:
    return Collection(records=(), completeness=Completeness(complete=True, total=0))


def _one() -> Collection:
    return Collection(
        records=(Record(fields={"courseNumber": Scalar(ScalarKind.IDENTIFIER, "00960211")},
                        basis=Basis.OFFICIAL_RECORD),),
        completeness=Completeness(complete=True, total=1),
    )


def _facts(**kw) -> dict:
    return {n: HeldFact(value=v, basis=Basis.OFFICIAL_RECORD) for n, v in kw.items()}


QUESTION = "Am I eligible for course 00999999?"


class TestTheNegativeCanBeSaid:
    def test_a_count_over_an_empty_collection_is_an_answer(self) -> None:
        result = resolve_answer(
            "I found {catalog:count} catalog records for that course, so it is not in the catalog.",
            _facts(catalog=_empty()), QUESTION)
        assert not isinstance(result, Ungrounded), getattr(result, "reason", "")
        assert "0" in result.text

    def test_the_live_phrasing_that_was_refused_three_times(self) -> None:
        result = resolve_answer(
            "I can’t determine eligibility for 00999999 because I found "
            "{course_catalog_entry:count} catalog records for that course.",
            _facts(course_catalog_entry=_empty()), QUESTION)
        assert not isinstance(result, Ungrounded), getattr(result, "reason", "")


class TestTheNonAnswerIsStillRefused:
    def test_bare_slots_over_empty_facts_still_fail(self) -> None:
        """The shape this rule was written for, and which shipped once:
        "I can't determine ... (none), (none), (none) are all empty"."""
        result = resolve_answer(
            "I can't determine this: {a}, {b} are all I have.",
            _facts(a=_empty(), b=_empty()), QUESTION)
        assert isinstance(result, Ungrounded)
        assert "empty" in result.reason

    def test_a_detail_render_of_nothing_still_fails(self) -> None:
        result = resolve_answer("Here is your plan:\n{plan:detail}",
                                _facts(plan=_empty()), QUESTION)
        assert isinstance(result, Ungrounded)

    def test_mixing_a_count_in_does_not_launder_a_bare_empty_slot(self) -> None:
        """A count of a POPULATED fact must not excuse bare empty ones -- only a
        count of the empty fact itself is the negative being asserted."""
        result = resolve_answer("I have {found:count} and also {a}, {b}.",
                                _facts(found=_one(), a=_empty(), b=_empty()), QUESTION)
        assert not isinstance(result, Ungrounded), "a populated fact is cited, so this is allowed"

    def test_an_answer_on_no_facts_at_all_still_fails(self) -> None:
        result = resolve_answer("That course does not exist.", _facts(), QUESTION)
        assert isinstance(result, Ungrounded)
        assert "no facts at all" in result.reason
