"""Tests for the course code -> display-name lookup used at the answer boundary.

The live evals shipped answers like "The courses on your record with a final
grade above 90 are: 00940704, 00940219, 03240033." -- correct, grounded, and
close to unreadable. 7 of 10 answers in the 2026-07-18 run carried bare codes.

The name CANNOT come from the model: the grounding backstop only checks numerals
(`_NUM.findall`), so a typed course name is never validated, and a plausible
fabricated name welded to a real code is worse than no name at all -- the user
has no reason to doubt it. So the name is looked up in code and rendered, never
composed.
"""

from __future__ import annotations

import pytest

from app.agent_core.loop.course_names import (
    canonical_course_code,
    course_codes_in,
    course_display_name,
    set_catalog_names,
)


@pytest.mark.parametrize(
    "wiki_label, canonical",
    [
        ("0960600", "00960600"),  # the real ISE elective, as the wiki renders it
        ("3240033", "03240033"),
        ("2160035", "02160035"),
    ],
)
def test_a_seven_digit_wiki_label_gets_its_leading_zero_back(wiki_label, canonical):
    assert canonical_course_code(wiki_label) == canonical


@pytest.mark.parametrize("already_canonical", ["00960600", "00940224"])
def test_an_eight_digit_code_is_left_alone(already_canonical):
    assert canonical_course_code(already_canonical) == already_canonical


@pytest.mark.parametrize("not_a_course_code", ["012345", "track-ise", "", "960600a", "123456789"])
def test_anything_that_is_not_a_bare_seven_digit_run_is_untouched(not_a_course_code):
    # 6-digit program codes, slugs, empties and 9-digit runs must pass through, so
    # this is safe to run over any extracted identifier, not only course codes.
    assert canonical_course_code(not_a_course_code) == not_a_course_code


@pytest.fixture(autouse=True)
def _clean_catalog_names():
    """`_catalog_names` is module-level process state, and `use_real_academic_engine`
    does not touch it. Without this, a test that seeds the fallback leaks it into
    every test that runs afterwards in the same session."""
    set_catalog_names({})
    yield
    set_catalog_names({})


def test_known_course_resolves_to_its_english_name(use_real_academic_engine):
    assert course_display_name("00940224") == "Data Structures and Algorithms"
    assert course_display_name("00960211") == "E-Commerce Models"


def test_a_qualifier_is_part_of_the_name_and_is_kept(use_real_academic_engine):
    """00940704 is "Introduction to Data Engineering (Advanced)". Stripping every
    parenthesised group -- the obvious way to drop the Hebrew gloss -- silently
    renames it to the non-Advanced course, which is a different course."""
    assert course_display_name("00940704") == "Introduction to Data Engineering (Advanced)"


def test_hebrew_gloss_is_dropped(use_real_academic_engine):
    """Titles read `<code> — <English> (<Hebrew>)`; the gloss would mix scripts
    into an English answer."""
    name = course_display_name("01040065")
    assert name == "Algebra 1M2"


def test_unknown_code_has_no_name(use_real_academic_engine):
    assert course_display_name("99999999") is None


def test_non_course_values_are_ignored(use_real_academic_engine):
    """Only 8-digit course codes are looked up -- a grade, a credit total or a
    semester must never be mistaken for one."""
    assert course_display_name("85") is None
    assert course_display_name("92.5") is None
    assert course_display_name("2025-1") is None
    assert course_display_name("") is None


# -- catalog fallback ---------------------------------------------------------


def test_catalog_title_names_a_course_the_wiki_does_not_cover(use_real_academic_engine):
    """The wiki holds 2601 courses, but a record also carries general electives and
    humanities that were never wiki'd. 03240305 shipped as a bare code in a live
    2026-07-19 answer, sitting among eight correctly-named courses."""
    assert course_display_name("03240305") is None

    set_catalog_names({"03240305": "היסטוריה של המדע"})
    assert course_display_name("03240305") == "היסטוריה של המדע"


def test_wiki_wins_over_the_catalog(use_real_academic_engine):
    """The wiki's names are English; the catalog's are usually Hebrew. A course in
    both must not lose its English name to the fallback."""
    set_catalog_names({"00940224": "מבני נתונים ואלגוריתמים"})
    assert course_display_name("00940224") == "Data Structures and Algorithms"


def test_catalog_fallback_still_refuses_non_course_values(use_real_academic_engine):
    """The fallback must not widen what counts as a course code -- a grade or a
    credit total that happens to be a catalog key stays unresolvable."""
    set_catalog_names({"85": "not a course", "2025-1": "also not a course"})
    assert course_display_name("85") is None
    assert course_display_name("2025-1") is None


# -- scanning prose for codes -------------------------------------------------


def test_course_codes_in_finds_codes_in_prose():
    assert course_codes_in("You passed 00940224 and 00960211.") == {"00940224", "00960211"}


def test_course_codes_in_ignores_numbers_that_are_not_course_codes():
    """Credits, grades and years share the digit alphabet; only the 8-digit shape
    is a course."""
    assert course_codes_in("You have 158.0 credits, a 92 average, and 2025 ahead.") == set()
    assert course_codes_in("") == set()
