"""The answer boundary -- phase 9d of docs/agent/tools_implementation_plan.md.

The invariant in one line: no number reaches a reader unless a fact produced it.

The most important test here is the LAST one. The boundary this replaces had a
structural gap -- interpreted prose fitted neither of its two categories, so
some answers had no legal form and the loop exhausted itself discovering that.
Typed facts remove the category rather than widening it, and that test is what
proves it.
"""

from __future__ import annotations

from app.agent_core.facts.answer import Answer, HeldFact, Ungrounded, resolve_answer
from app.agent_core.facts.prose import Citation
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _held(value, basis=Basis.OFFICIAL_RECORD, citation=None) -> HeldFact:
    return HeldFact(value=value, basis=basis, citation=citation)


def _courses(*codes: str) -> Collection:
    return Collection(
        records=tuple(Record(fields={"id": Scalar(I, c)}, basis=Basis.OFFICIAL_RECORD) for c in codes),
        completeness=Completeness(complete=True, total=len(codes)),
    )


FACTS = {
    "remaining": _held(Scalar(Q, 16.0)),
    "missing_courses": _held(_courses("00940224", "00960211")),
    "eligible": _held(Scalar(ScalarKind.BOOL, True)),
}


class TestSubstitution:
    def test_a_scalar_slot_is_filled_from_the_fact(self) -> None:
        result = resolve_answer("You have {remaining} credits left.", FACTS)
        assert isinstance(result, Answer)
        assert result.text == "You have 16 credits left."

    def test_a_whole_number_does_not_render_as_a_float(self) -> None:
        """'16.0 credits' reads as a rounding artefact; '16' reads as an answer."""
        assert "16 credits" in resolve_answer("{remaining} credits", FACTS).text

    def test_a_boolean_reads_as_language_not_a_python_literal(self) -> None:
        assert resolve_answer("Eligible: {eligible}", FACTS).text == "Eligible: yes"

    def test_a_collection_can_be_counted_or_listed(self) -> None:
        counted = resolve_answer("{missing_courses:count} courses remain.", FACTS)
        assert counted.text == "2 courses remain."
        listed = resolve_answer("Missing: {missing_courses:list}", FACTS)
        assert "00940224" in listed.text and "00960211" in listed.text

    def test_the_same_fact_twice_is_reported_once_as_used(self) -> None:
        result = resolve_answer("{remaining} and again {remaining}", FACTS)
        assert result.used == ("remaining",)


class TestGrounding:
    def test_a_number_typed_into_the_prose_is_refused(self) -> None:
        """THE invariant. The model may not state a figure it did not derive,
        however plausible it looks."""
        result = resolve_answer("You have 16 credits left.", FACTS)
        assert isinstance(result, Ungrounded)
        assert "no fact" in result.reason

    def test_the_refusal_shows_where_the_number_is(self) -> None:
        """A refusal that does not locate the problem gets the same answer back."""
        result = resolve_answer("Long preamble about your degree. You need 42 more.", FACTS)
        assert "42" in result.reason

    def test_a_slot_naming_no_fact_lists_what_exists(self) -> None:
        result = resolve_answer("You need {credits_remaining}.", FACTS)
        assert isinstance(result, Ungrounded)
        assert "credits_remaining" in result.reason
        assert "remaining" in result.reason

    def test_an_answer_standing_on_nothing_is_refused(self) -> None:
        """Qualitative answers still have to cite something, or nothing separates
        them from a guess."""
        result = resolve_answer("You are doing fine.", FACTS)
        assert isinstance(result, Ungrounded)
        assert "no facts" in result.reason

    def test_an_answer_citing_only_empty_facts_is_refused(self) -> None:
        """Found on the first live run against a real model.

        It wrote "I can't determine how many credits you still need: (none),
        (none), (none) are all empty" -- and the boundary PASSED it, because
        slots were present and no digit was typed. Citing empty facts is not
        citing facts, and accepting it lets a non-answer wear the shape of a
        verified one.
        """
        empty = {
            "a": _held(_courses()),
            "b": _held(_courses()),
        }
        result = resolve_answer("Cannot tell: {a} and {b} are empty.", empty)
        assert isinstance(result, Ungrounded)
        assert "empty" in result.reason

    def test_a_mix_of_empty_and_populated_facts_is_allowed(self) -> None:
        """'You have finished everything, nothing remains' is a real answer, and
        it legitimately cites an empty collection alongside a populated one."""
        facts = {
            "done": _held(_courses("00940224")),
            "remaining": _held(_courses()),
        }
        result = resolve_answer("Completed {done}; remaining: {remaining}.", facts)
        assert isinstance(result, Answer)

    def test_substituted_values_are_not_mistaken_for_typed_numbers(self) -> None:
        """The check reads the TEMPLATE, not the filled text -- otherwise every
        successful answer would fail the moment a number was substituted in."""
        assert isinstance(resolve_answer("{remaining} credits", FACTS), Answer)


class TestCertainty:
    def test_the_answer_is_only_as_strong_as_its_weakest_fact(self) -> None:
        facts = {
            "a": _held(Scalar(Q, 1.0), Basis.OFFICIAL_RECORD),
            "b": _held(Scalar(Q, 2.0), Basis.WIKI_DERIVED),
        }
        assert resolve_answer("{a} and {b}", facts).basis is Basis.WIKI_DERIVED

    def test_an_answer_from_a_simulated_plan_is_marked_speculative(self) -> None:
        facts = {"a": _held(Scalar(Q, 1.0), Basis.SIMULATED)}
        result = resolve_answer("{a} semesters", facts)
        assert result.speculative is True

    def test_only_the_facts_actually_USED_affect_certainty(self) -> None:
        """Holding a weak fact must not degrade an answer that never touched it."""
        facts = {
            "used": _held(Scalar(Q, 1.0), Basis.OFFICIAL_RECORD),
            "ignored": _held(Scalar(Q, 2.0), Basis.SIMULATED),
        }
        assert resolve_answer("{used}", facts).basis is Basis.OFFICIAL_RECORD


class TestCitations:
    def test_citations_travel_with_the_answer(self) -> None:
        facts = {
            "credits": _held(
                Scalar(Q, 155.0),
                Basis.LLM_INTERPRETATION,
                Citation(source="track-ise", quote="The degree requires 155 credits."),
            )
        }
        result = resolve_answer("The degree requires {credits} credits.", facts)
        assert len(result.citations) == 1
        assert result.citations[0].source == "track-ise"

    def test_only_cited_facts_contribute_citations(self) -> None:
        result = resolve_answer("{remaining} left", FACTS)
        assert result.citations == ()


class TestTheOldStructuralGapIsGone:
    def test_an_interpreted_value_is_slottable_like_any_other(self) -> None:
        """The failure this boundary was rebuilt to remove.

        Previously an interpreted claim fitted neither category the boundary
        had: too load-bearing to be free prose, too verbatim to be a slot. The
        answer had no legal form, so the loop was rejected every turn until its
        budget ran out and it shipped a raw fact dump.

        Now interpretation yields a typed SCALAR plus a separate CITATION. The
        value slots exactly like a fetched number, and the prose it came from
        rides alongside the answer instead of inside it -- so the third category
        does not need handling, because it no longer exists.
        """
        interpreted = _held(
            Scalar(Q, 155.0),
            Basis.LLM_INTERPRETATION,
            Citation(source="regulations-undergraduate", quote="A degree requires 155 credits."),
        )
        earned = _held(Scalar(Q, 62.5), Basis.OFFICIAL_RECORD)

        result = resolve_answer(
            "The degree requires {required} credits and you have earned {earned}.",
            {"required": interpreted, "earned": earned},
        )

        assert isinstance(result, Answer)
        assert result.text == "The degree requires 155 credits and you have earned 62.5."
        assert result.basis is Basis.LLM_INTERPRETATION, "an interpreted number must not pass as official"
        assert result.citations[0].source == "regulations-undergraduate"


class TestDerivationTravelsWithTheAnswer:
    """The fix for a live failure no gate could catch.

    The model named a fact `remaining_credits`, filled it with the degree's
    TOTAL, and answered "you still need 155 credits". Grounding passed and was
    right to: the number came from a real, official, non-empty fact. What was
    wrong was the NAME, which is prose the model wrote.

    Nothing below the model can check that. So the derivation rides along, and a
    reader sees where 155 actually came from.
    """

    def test_an_answer_reports_how_each_slot_was_derived(self) -> None:
        facts = {
            "total": HeldFact(
                value=Scalar(Q, 155.0),
                basis=Basis.OFFICIAL_RECORD,
                derivation="degree_programs -> aggregate:only(totalCredits)",
            )
        }
        result = resolve_answer("You need {total} credits.", facts)
        assert result.derivations == (("total", "degree_programs -> aggregate:only(totalCredits)"),)

    def test_the_derivation_exposes_a_misleading_name(self) -> None:
        """The exact live case: the name says 'remaining', the derivation says
        'the degree total'. A reader catches in one glance what no check could."""
        facts = {
            "remaining_credits": HeldFact(
                value=Scalar(Q, 155.0),
                basis=Basis.OFFICIAL_RECORD,
                derivation="degree_record -> aggregate:only(totalCredits)",
            )
        }
        result = resolve_answer("You still need {remaining_credits} credits.", facts)
        assert isinstance(result, Answer)
        name, how = result.derivations[0]
        assert name == "remaining_credits"
        assert "totalCredits" in how, "the derivation must contradict the name visibly"

    def test_facts_without_a_derivation_are_simply_omitted(self) -> None:
        facts = {"given": HeldFact(value=Scalar(Q, 1.0), basis=Basis.OFFICIAL_RECORD)}
        assert resolve_answer("{given}", facts).derivations == ()

    def test_only_used_facts_contribute_derivations(self) -> None:
        facts = {
            "used": HeldFact(Scalar(Q, 1.0), Basis.OFFICIAL_RECORD, derivation="a -> b"),
            "unused": HeldFact(Scalar(Q, 2.0), Basis.OFFICIAL_RECORD, derivation="c -> d"),
        }
        assert [n for n, _ in resolve_answer("{used}", facts).derivations] == ["used"]


class TestNumeralsEchoedFromTheQuestion:
    """Course codes are numerals, and the grounding rule made them unsayable.

    A live run wrote "course 00960211 is not offered in the summer", had it
    refused as a typed number, rephrased twice, and ran out of attempts. The
    number was the user's own reference, quoted back.
    """

    def test_a_code_from_the_question_may_be_named(self) -> None:
        result = resolve_answer(
            "Course 00960211 has {remaining} credits left.",
            FACTS,
            question="Am I eligible to take course 00960211?",
        )
        assert isinstance(result, Answer)

    def test_a_number_absent_from_the_question_is_still_refused(self) -> None:
        """The invariant itself. Echoing is narrow -- it is not a licence to type."""
        result = resolve_answer(
            "You need 92.5 credits.", FACTS, question="How many credits do I need?"
        )
        assert isinstance(result, Ungrounded)

    def test_a_single_shared_digit_does_not_admit_a_whole_number(self) -> None:
        """The trap in the obvious implementation: matching digit-by-digit, "0"
        appears in almost every question mentioning a course code, so every
        numeral in every answer would have passed."""
        result = resolve_answer(
            "You need 155 credits.", FACTS, question="What about course 00960211?"
        )
        assert isinstance(result, Ungrounded)

    def test_with_no_question_the_rule_is_unchanged(self) -> None:
        assert isinstance(resolve_answer("You need 16 credits.", FACTS), Ungrounded)

    def test_a_code_followed_by_a_comma_is_still_recognised_as_echoed(self) -> None:
        """The trailing-punctuation trap. A greedy numeric token captured
        "00960211," WITH the comma in "...course 00960211, and none...", which is
        not the "00960211" in the question -- so a correct negative answer was
        refused three times on a live run for naming the course the user asked
        about."""
        result = resolve_answer(
            "I checked {courses}, and course 00960211, offered in summer, is not among them.",
            {"courses": _held(_courses("00940224"))},
            question="Is course 00960211 offered in the summer semester?",
        )
        assert isinstance(result, Answer)

    def test_a_decimal_is_still_matched_whole(self) -> None:
        """The fix must not break the ordinary case: 92.5 is one token, refused
        when it is typed rather than slotted."""
        result = resolve_answer("You need 92.5 credits.", FACTS, question="How many credits?")
        assert isinstance(result, Ungrounded)


class TestRawObjectIdsAreRefused:
    """A 24-hex ObjectId in a finished answer is an internal key leaking to the
    reader. A live run slotted a transcript keyed by `courseId` and rendered two
    dozen of them into prose; every one was grounded and every one was useless.
    """

    def test_a_slotted_objectid_is_refused_with_a_route_to_the_fix(self) -> None:
        held = {
            "completed": _held(
                Collection(
                    records=(Record(fields={"courseId": Scalar(I, "6a3db0e382df7b7cb04552be")}, basis=Basis.OFFICIAL_RECORD),),
                    completeness=Completeness(complete=True, total=1),
                )
            )
        }
        result = resolve_answer("You completed {completed}.", held, question="Which courses have I completed?")
        assert isinstance(result, Ungrounded)
        assert "courseNumber" in result.reason and "join" in result.reason

    def test_a_course_number_is_not_mistaken_for_an_id(self) -> None:
        result = resolve_answer(
            "You completed {codes}.",
            {"codes": _held(_courses("00940224", "00960211"))},
            question="Which courses have I completed?",
        )
        assert isinstance(result, Answer)


class TestCollectionRenderSkipsInternalIds:
    """`{offerings}` used to render each record's FIRST field -- the ObjectId
    `_id` -- dumping internal keys into prose the ObjectId guard then refused.
    A readable field is preferred."""

    def test_a_collection_renders_its_readable_field_not_its_id(self) -> None:
        offerings = Collection(
            records=(
                Record(
                    fields={"_id": Scalar(I, "6a3db0e482df7b7cb0455380"), "courseNumber": Scalar(I, "00960211")},
                    basis=Basis.OFFICIAL_RECORD,
                ),
            ),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer("I checked {offerings}.", {"offerings": _held(offerings)}, question="offered?")
        assert isinstance(result, Answer)
        assert "00960211" in result.text
        assert "6a3db0e4" not in result.text


class TestMalformedSlotsAreCaught:
    """A `{fact.field}` -- a dotted projection the grammar does not allow --
    matched no slot, so it was neither rendered nor flagged, and the raw braces
    SHIPPED in an accepted answer on a live run. It must be refused."""

    def test_a_dotted_field_slot_is_refused(self) -> None:
        result = resolve_answer(
            "Your codes are {codes.courseNumber}.",
            {"codes": _held(_courses("00940704"))},
            question="which courses?",
        )
        assert isinstance(result, Ungrounded)
        assert "not a valid slot" in result.reason

    def test_a_bare_collection_slot_still_works(self) -> None:
        result = resolve_answer(
            "Your codes are {codes}.",
            {"codes": _held(_courses("00940704", "03240033"))},
            question="which courses?",
        )
        assert isinstance(result, Answer)
        assert "00940704" in result.text


class TestLargeCollectionSlotsAreCapped:
    """A slot holding a big collection dumped every record into prose -- a live
    partial answer listed 117 prerequisite edges inline. Cap it."""

    def test_a_long_list_is_capped_with_a_remainder(self) -> None:
        many = _courses(*[f"009{n:05d}" for n in range(40)])
        result = resolve_answer("Remaining: {rem}.", {"rem": _held(many)}, question="which?")
        assert isinstance(result, Answer)
        assert "and 25 more" in result.text  # 40 - 15
        assert result.text.count(",") < 40, "it must not list all 40"

    def test_a_short_list_is_shown_in_full(self) -> None:
        few = _courses("00940224", "00960211", "00970800")
        result = resolve_answer("You have {c}.", {"c": _held(few)}, question="which?")
        assert isinstance(result, Answer)
        assert "and" not in result.text.replace("00940224", "") or "more" not in result.text
        assert "00970800" in result.text


class TestDetailRender:
    """`:detail` -- one line per record, every field as "label value".

    The bare `{fact}` slot shows ONE field per record. A plan a reader must act
    on needs each course's number AND title AND credits AND the grade computed
    for it, so a two-semester schedule was unrenderable through the boundary
    until this modifier existed. It stays domain-blind: the labels are whatever
    field names the caller projected, not anything this module knows.
    """

    def test_it_lists_every_field_per_record(self) -> None:
        plan = Collection(
            records=(
                Record(
                    fields={
                        "courseNumber": Scalar(I, "0960327"),
                        "title": Scalar(ScalarKind.TEXT, "Nonlinear OR"),
                        "credits": Scalar(Q, 3.5),
                        "type": Scalar(ScalarKind.TEXT, "elective"),
                    },
                    basis=Basis.SIMULATED,
                ),
            ),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer("Winter:\n{plan:detail}", {"plan": _held(plan, Basis.SIMULATED)})
        assert isinstance(result, Answer)
        for shown in ("courseNumber 0960327", "title Nonlinear OR", "credits 3.5", "type elective"):
            assert shown in result.text

    def test_one_line_per_record(self) -> None:
        plan = Collection(
            records=(
                Record(fields={"courseNumber": Scalar(I, "0960327")}, basis=Basis.SIMULATED),
                Record(fields={"courseNumber": Scalar(I, "0940314")}, basis=Basis.SIMULATED),
            ),
            completeness=Completeness(complete=True, total=2),
        )
        result = resolve_answer("{plan:detail}", {"plan": _held(plan, Basis.SIMULATED)})
        assert isinstance(result, Answer)
        assert [line for line in result.text.split("\n") if line.startswith("- ")] == [
            "- courseNumber 0960327",
            "- courseNumber 0940314",
        ]

    def test_it_drops_object_ids_so_the_finished_answer_guard_does_not_reject_it(self) -> None:
        """A placed row can still carry an internal courseId. If :detail printed
        it, the ObjectId guard would refuse the whole plan -- so it skips any
        24-hex value, keeping the readable number."""
        plan = Collection(
            records=(
                Record(
                    fields={"courseId": Scalar(I, "a" * 24), "courseNumber": Scalar(I, "0960327")},
                    basis=Basis.SIMULATED,
                ),
            ),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer("{plan:detail}", {"plan": _held(plan, Basis.SIMULATED)})
        assert isinstance(result, Answer)
        assert "a" * 24 not in result.text
        assert "courseNumber 0960327" in result.text

    def test_a_course_placed_in_two_semesters_is_refused(self) -> None:
        """The faked-split signature from every live planning run: rather than
        call optimize, the model selected offerings by semesterName, so a course
        offered in both terms landed in both lists. A real placement puts each
        course in ONE slot, so a repeat proves optimize was skipped -- refuse it
        and point back at optimize."""
        winter = Collection(
            records=(Record(fields={"number": Scalar(I, "00940219")}, basis=Basis.SIMULATED),),
            completeness=Completeness(complete=True, total=1),
        )
        spring = Collection(
            records=(Record(fields={"number": Scalar(I, "00940219")}, basis=Basis.SIMULATED),),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer(
            "Winter\n{winter:detail}\nSpring\n{spring:detail}",
            {"winter": _held(winter, Basis.SIMULATED), "spring": _held(spring, Basis.SIMULATED)},
        )
        assert isinstance(result, Ungrounded)
        assert "00940219" in result.reason and "optimize" in result.reason

    def test_a_real_placement_each_course_once_is_accepted(self) -> None:
        """The other side: a genuine optimize plan places each course in exactly
        one semester, so no code repeats and the gate stays silent."""
        winter = Collection(
            records=(Record(fields={"number": Scalar(I, "00940219")}, basis=Basis.SIMULATED),),
            completeness=Completeness(complete=True, total=1),
        )
        spring = Collection(
            records=(Record(fields={"number": Scalar(I, "00960327")}, basis=Basis.SIMULATED),),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer(
            "Winter\n{winter:detail}\nSpring\n{spring:detail}",
            {"winter": _held(winter, Basis.SIMULATED), "spring": _held(spring, Basis.SIMULATED)},
        )
        assert isinstance(result, Answer)

    def test_an_empty_collection_renders_as_none(self) -> None:
        empty = Collection(records=(), completeness=Completeness(complete=True, total=0))
        # paired with a populated fact so the all-empty guard does not fire first
        result = resolve_answer(
            "Winter:\n{plan:detail}\nCount: {n}",
            {"plan": _held(empty, Basis.SIMULATED), "n": _held(Scalar(Q, 0.0), Basis.SIMULATED)},
        )
        assert isinstance(result, Answer)
        assert "(none)" in result.text
