"""A course code in an answer must say which course it is.

Reported from the GUI. The answer was correct and unreadable:

    "You need 00960324 first. It has 2 prerequisite options: 00940314,
     00980413. 0 of those are already on your passed list..."

Reading it means opening the catalog three times.

The cause was not the renderer. `course_display_name` returned None for all
2613 courses: `_name_index` reads an in-process graph engine that this
deployment does not have, and the catalog fallback is filled by
`load_catalog_names`, which existed since the port and which nothing ever
called. Both sources empty, no error anywhere -- a missing name is deliberately
not fatal, so it degraded silently on every answer the agent has ever given.
"""

from __future__ import annotations

import pytest

from app.agent_core.loop.course_names import (
    course_display_name,
    pair_codes_with_names,
    reset_course_name_index,
    set_catalog_names,
)

# The real wiki names, which is what production resolves. Asserting against
# seeded stand-ins would have hidden that the wiki source outranks the catalog.
WIKI = {
    "00960324": "Service Systems Engineering",
    "00940314": "Stochastic Models in Operations Research",
}
# No wiki page, so these fall back to the catalog's Hebrew title. A real answer
# mixes the two, and the reported one did: 00940224 was named and 00940226 was
# not, in the same sentence.
CATALOG_ONLY = {
    "00980413": "תהליכים סטוכסטיים",
    "03240305": "היסטוריה של המדע",
}


@pytest.fixture(autouse=True)
def _names():
    set_catalog_names(CATALOG_ONLY)
    yield
    reset_course_name_index()


class TestPairingTheCodeWithItsName:
    def test_the_reported_answer_becomes_readable(self) -> None:
        reported = (
            "You need 00960324 first. It has 2 prerequisite options: 00940314, "
            "00980413."
        )
        assert pair_codes_with_names(reported) == (
            f"You need 00960324 ({WIKI['00960324']}) first. It has 2 prerequisite "
            f"options: 00940314 ({WIKI['00940314']}), "
            f"00980413 ({CATALOG_ONLY['00980413']})."
        )

    def test_the_catalog_fallback_names_what_the_wiki_does_not(self) -> None:
        """The wiki covers 2601 courses; a student's record also carries general
        electives and humanities that were never wiki'd."""
        assert pair_codes_with_names("You took 03240305.") == (
            "You took 03240305 (היסטוריה של המדע)."
        )

    def test_only_the_first_mention_is_named(self) -> None:
        """Repeating it at every occurrence turns two lines into a paragraph."""
        out = pair_codes_with_names("00960324 blocks you; take 00960324 next term.")
        assert out.count(WIKI["00960324"]) == 1
        assert out.startswith(f"00960324 ({WIKI['00960324']})")

    def test_a_detail_row_already_carrying_the_title_is_left_alone(self) -> None:
        """A plan row projects `title` beside the number; pairing would double it."""
        row = f"- course 00960324 · title {WIKI['00960324']} · credits 3"
        assert pair_codes_with_names(row) == row

    def test_a_code_with_no_known_name_is_never_given_one(self) -> None:
        """The rule is that no name is INVENTED. What the code gets instead is
        the honest note that the catalog does not have it -- see
        `TestACourseTheCatalogDoesNotHave`."""
        out = pair_codes_with_names("00999999 is unknown.")
        assert "00999999" in out
        assert "not in the course catalog" in out

    def test_numbers_that_are_not_course_codes_are_untouched(self) -> None:
        """Credits, grades and GPAs must never be looked up as courses."""
        text = "You have 129.5 of 155 credits and a GPA of 74.45, needing 25.5 more."
        assert pair_codes_with_names(text) == text

    def test_it_never_changes_a_code(self) -> None:
        """The one thing a readability pass must not do is alter the claim."""
        text = "You need 00960324 before 00970135."
        out = pair_codes_with_names(text)
        for code in ("00960324", "00970135"):
            assert code in out

    def test_empty_and_none_are_safe(self) -> None:
        """None normalises to empty rather than propagating -- this runs on
        every outcome, including the ones that produced no text."""
        assert pair_codes_with_names("") == ""
        assert pair_codes_with_names(None) == ""


class TestTheNameSourceIsActuallyConnected:
    def test_the_wiki_source_resolves(self) -> None:
        """The regression that made every answer show bare codes: BOTH sources
        dead, and nothing raising to say so."""
        assert course_display_name("00960324") == WIKI["00960324"]

    def test_a_course_in_neither_source_has_no_name_to_give(self) -> None:
        """00940226 is named by 259 prerequisite edges and sits in no catalog row
        and no wiki page -- a gap in the DATA. An invented name would be worse
        than none, because nothing about it invites doubt; what it gets is the
        gap said out loud."""
        assert course_display_name("00940226") is None
        assert pair_codes_with_names("It requires 00940226.") == (
            "It requires 00940226 (not in the course catalog)."
        )

    def test_a_non_code_is_not_looked_up(self) -> None:
        for value in ("129.5", "abc", "", "0096032"):
            assert course_display_name(value) is None


class TestTheNameFollowsTheQuestionsLanguage:
    """A Hebrew question deserves the Hebrew course name.

    The wiki title is English and the catalog title is Hebrew, so the sources
    hold the two languages and the only question is which leads. It used to be
    a fixed preference for English, which meant a student asking in Hebrew was
    told they needed "Service Systems Engineering" and had to translate it back
    before they could find it on their registration page.

    Whichever is not preferred stays the fallback: 2601 courses have a wiki page
    and 2613 have a catalog row, so each covers gaps in the other, and a name in
    the wrong language still beats an 8-digit number.
    """

    BOTH = {"00960324": "הנדסת מערכות שירות"}  # also has an English wiki page

    def test_a_hebrew_answer_prefers_the_hebrew_name(self) -> None:
        set_catalog_names(self.BOTH)
        assert pair_codes_with_names("צריך 00960324.", hebrew=True) == (
            "צריך 00960324 (הנדסת מערכות שירות)."
        )

    def test_an_english_answer_prefers_the_english_name(self) -> None:
        set_catalog_names(self.BOTH)
        assert pair_codes_with_names("You need 00960324.") == (
            f"You need 00960324 ({WIKI['00960324']})."
        )

    def test_hebrew_falls_back_to_the_wiki_when_the_catalog_lacks_it(self) -> None:
        set_catalog_names({})
        assert course_display_name("00960324", hebrew=True) == WIKI["00960324"]

    def test_english_falls_back_to_the_catalog_when_the_wiki_lacks_it(self) -> None:
        set_catalog_names({"03240305": "היסטוריה של המדע"})
        assert course_display_name("03240305") == "היסטוריה של המדע"

    def test_the_answer_seam_reads_the_language_off_the_question(self) -> None:
        """`_answer_text` is where the two meet, and it is the only place that
        knows both the question and the finished prose."""
        from app.agent_core.facts.answer import Answer
        from app.agent_core.facts.loop import LoopResult
        from app.agent_core.facts.service import _answer_text
        from app.agent_core.facts.types import Basis

        set_catalog_names(self.BOTH)
        for question, expected in (
            ("צריך קורס?", "הנדסת מערכות שירות"),
            ("what do I need?", WIKI["00960324"]),
        ):
            result = LoopResult(
                outcome="answered",
                answer=Answer("00960324", Basis.OFFICIAL_RECORD, (), ()),
                question=question,
            )
            assert expected in _answer_text(result), question


class TestACourseTheCatalogDoesNotHave:
    """768 of 4766 prerequisite edges point at 259 codes with no catalog row.

    The edge is real -- the course genuinely IS named as a requirement -- and
    nothing about it can be looked up: no name, no credits, no offerings. Printed
    as a bare number beside fully-described courses it reads as if it were one of
    them, and a student told "you need 00940226" and nothing else cannot tell
    whether the agent failed or the data did.

    That is a gap in the seeded data rather than in this module, so it is
    SURFACED, not repaired. An invented name would be worse than none.
    """

    def test_an_uncatalogued_prerequisite_says_so(self) -> None:
        set_catalog_names(dict(CATALOG_ONLY))
        assert pair_codes_with_names("It requires 00940226.") == (
            "It requires 00940226 (not in the course catalog)."
        )

    def test_it_says_so_in_hebrew_too(self) -> None:
        set_catalog_names(dict(CATALOG_ONLY))
        assert pair_codes_with_names("נדרש 00940226.", hebrew=True) == (
            "נדרש 00940226 (לא נמצא בקטלוג)."
        )

    def test_a_known_course_is_unaffected(self) -> None:
        set_catalog_names(dict(CATALOG_ONLY))
        assert "not in the course catalog" not in pair_codes_with_names(
            "It requires 00960324."
        )

    def test_nothing_is_claimed_when_the_catalog_did_not_load(self, monkeypatch) -> None:
        """The failure mode that would matter: an index that failed to fill
        would otherwise make 2613 false statements about the curriculum."""
        import app.agent_core.loop.course_names as module

        set_catalog_names({})
        monkeypatch.setattr(module, "_name_index", lambda: {})
        assert module.pair_codes_with_names("It requires 00940226.") == (
            "It requires 00940226."
        )

    def test_only_the_first_mention_is_flagged(self) -> None:
        set_catalog_names(dict(CATALOG_ONLY))
        out = pair_codes_with_names("00940226 blocks you; take 00940226 later.")
        assert out.count("not in the course catalog") == 1
