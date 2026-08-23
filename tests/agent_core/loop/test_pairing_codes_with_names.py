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

    def test_a_code_with_no_known_name_is_untouched(self) -> None:
        assert pair_codes_with_names("00999999 is not in the catalog.") == (
            "00999999 is not in the catalog."
        )

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

    def test_a_course_in_neither_source_stays_a_bare_code(self) -> None:
        """00940226 is named by 259 prerequisite edges and sits in no catalog row
        and no wiki page -- a gap in the DATA. A bare code is the honest render;
        an invented name would be worse, because nothing about it invites doubt."""
        assert course_display_name("00940226") is None
        assert pair_codes_with_names("It requires 00940226.") == "It requires 00940226."

    def test_a_non_code_is_not_looked_up(self) -> None:
        for value in ("129.5", "abc", "", "0096032"):
            assert course_display_name(value) is None
