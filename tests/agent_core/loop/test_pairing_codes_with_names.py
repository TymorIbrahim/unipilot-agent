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

NAMES = {
    "00960324": "הנדסת מערכות שירות",
    "00940314": "מודלים סטוכסטיים בחקר בצועים",
    "00980413": "תהליכים סטוכסטיים",
    "00940704": "סדנת תכנות בשפת סי",
}


@pytest.fixture(autouse=True)
def _names():
    set_catalog_names(NAMES)
    yield
    reset_course_name_index()


class TestPairingTheCodeWithItsName:
    def test_the_reported_answer_becomes_readable(self) -> None:
        reported = (
            "You need 00960324 first. It has 2 prerequisite options: 00940314, "
            "00980413."
        )
        assert pair_codes_with_names(reported) == (
            "You need 00960324 (הנדסת מערכות שירות) first. It has 2 prerequisite "
            "options: 00940314 (מודלים סטוכסטיים בחקר בצועים), "
            "00980413 (תהליכים סטוכסטיים)."
        )

    def test_only_the_first_mention_is_named(self) -> None:
        """Repeating it at every occurrence turns two lines into a paragraph."""
        out = pair_codes_with_names("00960324 blocks you; take 00960324 next term.")
        assert out.count("הנדסת מערכות שירות") == 1
        assert out.startswith("00960324 (הנדסת מערכות שירות)")

    def test_a_detail_row_already_carrying_the_title_is_left_alone(self) -> None:
        """A plan row projects `title` beside the number; pairing would double it."""
        row = "- קורס 00940704 · שם סדנת תכנות בשפת סי · נקודות 1.5"
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
    def test_the_catalog_fallback_resolves(self) -> None:
        """The regression that made every answer show bare codes: both sources
        empty, and nothing raising to say so."""
        assert course_display_name("00960324") == "הנדסת מערכות שירות"

    def test_a_non_code_is_not_looked_up(self) -> None:
        for value in ("129.5", "abc", "", "0096032"):
            assert course_display_name(value) is None
