"""The prose guards must hold in the language the answer is actually written in.

Nothing in `SYSTEM_PROMPT` says which language to reply in, and asked in Hebrew
the model replies in Hebrew -- nine of ten live runs did. Every prose check was
an English regex, so on those answers the guards did not fail, they went
SILENT, and a bare "נשארו לך 2 סמסטרים" shipped past the exact check that was
built to stop "it will take you 2 semesters".

That is the failure mode worth a test file of its own: a guard that is absent
looks identical to a guard that passed, and the run scores as clean either way.
The English cases are asserted alongside each Hebrew one so a future edit
cannot buy Hebrew coverage by loosening English.
"""

from __future__ import annotations

from app.agent_core.facts.answer import _tidy_affirmations
from app.agent_core.facts.postconditions import (
    check_count_states_its_basis,
    check_periods_are_whole,
)

# The live pair: the question asked, and the answer that shipped past the guard.
ASKED = "כמה סמסטרים נשארו לי עד סיום התואר?"
SHIPPED = "נשארו לך 2 סמסטרים עד סיום התואר."
WITH_BASIS = "נשארו לך 2 סמסטרים — נותרו לך 25.5 נקודות ואתה לוקח עד 18 בסמסטר."


def kinds(violations) -> list[str]:
    return [v.kind for v in violations]


class TestACountStillNeedsItsBasisInHebrew:
    def test_the_answer_that_shipped_is_refused(self) -> None:
        assert kinds(check_count_states_its_basis(SHIPPED, 25.5, ASKED)) == [
            "count_without_basis"
        ]

    def test_stating_the_credits_clears_it(self) -> None:
        assert check_count_states_its_basis(WITH_BASIS, 25.5, ASKED) == []

    def test_the_english_twin_is_still_refused(self) -> None:
        assert kinds(
            check_count_states_its_basis(
                "It will take you 2 semesters.",
                25.5,
                "How many semesters until I graduate?",
            )
        ) == ["count_without_basis"]

    def test_a_hebrew_number_word_counts_as_a_count(self) -> None:
        assert kinds(
            check_count_states_its_basis("נשארו לך שני סמסטרים.", 25.5, ASKED)
        ) == ["count_without_basis"]


class TestItDoesNotArmOnHebrewPolicyQuestions:
    """The English version began by refusing three correct policy answers.

    The regulations are full of semester counts -- "the two semesters
    immediately following", "by the end of the 4th semester" -- so a guard keyed
    on the ANSWER fires on any answer that quotes one. It is keyed on the
    QUESTION instead, and the Hebrew patterns have to be scoped just as tightly:
    "כמה זמן יש לי לערער" is "how long do I have to appeal", and the credits
    still needed have nothing to do with it.
    """

    QUESTIONS = [
        "כמה זמן יש לי לערער על ציון בבחינה?",
        "אפשר לחזור על קורס שכבר עברתי כדי לשפר ציון?",
        "יש דדליין לסיום דרישת האנגלית, ועמדתי בו?",
        "מה מספר הנקודות המקסימלי שמותר לי לקחת בסמסטר אחד?",
    ]

    def test_a_quoted_semester_count_is_not_refused(self) -> None:
        answer = "אתה רשאי להירשם מחדש בשני הסמסטרים העוקבים."
        for question in self.QUESTIONS:
            assert check_count_states_its_basis(answer, 25.5, question) == [], question


class TestAPeriodIsStillIndivisibleInHebrew:
    def test_a_fractional_hebrew_semester_is_caught(self) -> None:
        assert kinds(check_periods_are_whole("נשארו לך 1.42 סמסטרים")) == [
            "fractional_period"
        ]

    def test_the_english_twin_is_still_caught(self) -> None:
        assert kinds(check_periods_are_whole("You have 1.42 semesters left")) == [
            "fractional_period"
        ]

    def test_hebrew_credits_are_untouched(self) -> None:
        assert check_periods_are_whole("נותרו לך 25.5 נקודות") == []


class TestTheSeamBetweenTheTwoScripts:
    """`אתה זכאי לan additional exam date` -- a live answer, verbatim.

    The corpus is English, so a fact interpreted out of it is an English phrase,
    and Hebrew's one-letter prefixes are written closed up against the word they
    govern. The template was right and the value was right; only the join was
    wrong, and the model could not have fixed it because it never sees the value.
    """

    def test_a_prefix_before_a_latin_value_takes_a_hyphen(self) -> None:
        assert (
            _tidy_affirmations("אתה זכאי לan additional exam date.")
            == "אתה זכאי ל-an additional exam date."
        )

    def test_a_word_final_hebrew_letter_takes_a_space(self) -> None:
        assert (
            _tidy_affirmations("הקורס נקרא מבואIntroduction")
            == "הקורס נקרא מבוא Introduction"
        )

    def test_latin_running_into_hebrew_takes_a_space(self) -> None:
        assert _tidy_affirmations("eligible forקורס") == "eligible for קורס"

    def test_prose_in_one_script_is_untouched(self) -> None:
        for text in (
            "אתה במצב אקדמי תקין.",
            "You are eligible for the course.",
            "מותר לך עד 29 נקודות בסמסטר.",
        ):
            assert _tidy_affirmations(text) == text

    def test_it_only_ever_inserts(self) -> None:
        """The one thing a cosmetic repair must never do is change the claim."""
        text = "אתה זכאי לan additional exam date, ולא ל-2 מועדים."
        assert _tidy_affirmations(text).replace("-", "").replace(" ", "") == (
            text.replace("-", "").replace(" ", "")
        )


class TestThePartialIsWrittenInTheAskedLanguage:
    """The one path that cannot ask the model which language to use.

    A partial is assembled in code precisely because the run has no budget left
    for another call, so it said its three sentences in English regardless of
    who was reading. Live: a Hebrew question stalled at 118s and came back
    apologising in English over a list of Hebrew-derived facts.
    """

    def facts(self):
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.types import Basis, Scalar, ScalarKind

        return {
            "credits_needed": HeldFact(
                Scalar(ScalarKind.QUANTITY, 25.5), Basis.OFFICIAL_RECORD
            )
        }

    def partial(self, question: str, outcome: str = "stalled") -> str:
        from app.agent_core.facts.loop import LoopResult
        from app.agent_core.facts.service import _partial_from_facts

        return _partial_from_facts(
            LoopResult(outcome=outcome, facts=self.facts(), question=question)
        )

    def test_a_hebrew_question_gets_a_hebrew_partial(self) -> None:
        text = self.partial("כמה סמסטרים נשארו לי?")
        assert "נתקעתי" in text
        assert "I stopped making progress" not in text

    def test_an_english_question_is_unchanged(self) -> None:
        text = self.partial("How many semesters do I have left?")
        assert text.startswith("I stopped making progress")

    def test_every_outcome_has_a_hebrew_form(self) -> None:
        from app.agent_core.facts.service import (
            _PARTIAL_PREFIX_BY_OUTCOME,
            _PARTIAL_PREFIX_BY_OUTCOME_HE,
        )

        assert set(_PARTIAL_PREFIX_BY_OUTCOME_HE) == set(_PARTIAL_PREFIX_BY_OUTCOME)

    def test_the_facts_are_reported_either_way(self) -> None:
        """Translating the wrapper must not drop what the run established."""
        for question in ("כמה סמסטרים נשארו לי?", "How many semesters are left?"):
            assert "25.5" in self.partial(question)
            assert "credits needed" in self.partial(question)


class TestStoredEnumValuesReachTheReaderAsWords:
    """`סוג mandatory · דרישות met` -- a live Hebrew plan, verbatim.

    The planner writes `prereqStatus: "check_prerequisites"` and the catalog
    writes `category: "elective"`, and `:detail` printed them as stored. In a
    Hebrew answer that is English inside Hebrew prose; `check_prerequisites` is
    worse than untranslated, because it is an instruction to the system and the
    student has been handed a variable name.

    `_render_scalar` has printed bools as "yes"/"no" since the beginning, so a
    presentation decision in code is established here, not a new boundary.
    """

    def render(self, value: str, question: str) -> str:
        from app.agent_core.facts.answer import HeldFact, resolve_answer
        from app.agent_core.facts.types import (
            Basis, Collection, Completeness, Record, Scalar, ScalarKind,
        )

        plan = Collection(
            records=(
                Record(
                    fields={"סוג": Scalar(ScalarKind.TEXT, value)},
                    basis=Basis.SIMULATED,
                ),
            ),
            completeness=Completeness(complete=True, total=1),
        )
        result = resolve_answer(
            "{plan:detail}", {"plan": HeldFact(plan, Basis.SIMULATED)}, question
        )
        return result.text

    HEBREW_Q = "תכנן לי סמסטר"
    ENGLISH_Q = "plan my term"

    def test_a_hebrew_question_gets_hebrew_enum_values(self) -> None:
        assert self.render("mandatory", self.HEBREW_Q) == "סוג חובה"
        assert self.render("elective", self.HEBREW_Q) == "סוג בחירה"

    def test_an_english_question_keeps_english(self) -> None:
        assert self.render("mandatory", self.ENGLISH_Q) == "סוג mandatory"

    def test_an_internal_status_never_reaches_the_reader_raw(self) -> None:
        """`check_prerequisites` is a variable name, in either language."""
        for question in (self.HEBREW_Q, self.ENGLISH_Q):
            assert "check_prerequisites" not in self.render(
                "check_prerequisites", question
            )

    def test_an_unknown_value_passes_through_untouched(self) -> None:
        """A closed table must not mangle an enum nobody anticipated."""
        assert self.render("seminar", self.HEBREW_Q) == "סוג seminar"

    def test_free_text_is_never_translated(self) -> None:
        """Course titles are data. Guessing at their language corrupts them."""
        title = "Introduction to Human Factors Engineering"
        assert title in self.render(title, self.HEBREW_Q)
