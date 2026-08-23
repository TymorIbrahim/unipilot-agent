"""Course-code canonicalisation, and an honest note about the display names.

The last of `tests/pending_supabase/`. Only part of it ported, and the part that
did not is worth saying out loud.

WHAT IS LIVE. `canonical_course_code` restores the leading zero the wiki drops,
and it is load-bearing: `dispatch` runs every identifier through it before a
`plan_term` candidate list is built, so a seven-digit label from prose and an
eight-digit code from the catalog name the same course. Get it wrong in either
direction and a course silently fails to match -- or a six-digit program code
becomes an eight-digit course that does not exist.

WHAT WAS NOT, AND NOW IS. This suite's other half tested `course_display_name`
resolving codes to English names, and was rewritten to assert that it CANNOT --
"a dead path, not a broken one". That was wrong, and it made the suite guard the
defect: `_name_index` walked a graph engine absent from this deployment while
the wiki corpus loaded in the same process held 2601 course titles in exactly
the format it parses, and `load_catalog_names` had no call site. Both are wired
now, and every answer between the port and that fix carried bare 8-digit codes.

The lesson is in the old docstring, which had it right and drew the opposite
conclusion: "returns None" and "is broken" look identical from a call site. When
they do, find out which -- do not pin it.
"""

from __future__ import annotations

import pytest

from app.agent_core.loop.course_names import (
    canonical_course_code,
    course_codes_in,
    course_display_name,
    set_catalog_names,
)


@pytest.fixture(autouse=True)
def _clean_catalog_names():
    """`_catalog_names` is module-level process state. Without this, a test that
    seeds it leaks into every test that runs afterwards in the same session."""
    set_catalog_names({})
    yield
    set_catalog_names({})


class TestCanonicalisation:
    @pytest.mark.parametrize(
        "wiki_label, canonical",
        [
            ("0960600", "00960600"),  # the real ISE elective, as the wiki renders it
            ("3240033", "03240033"),
            ("2160035", "02160035"),
        ],
    )
    def test_a_seven_digit_wiki_label_gets_its_leading_zero_back(
        self, wiki_label: str, canonical: str
    ) -> None:
        assert canonical_course_code(wiki_label) == canonical

    @pytest.mark.parametrize("already_canonical", ["00960600", "00940224"])
    def test_an_eight_digit_code_is_left_alone(self, already_canonical: str) -> None:
        assert canonical_course_code(already_canonical) == already_canonical

    @pytest.mark.parametrize(
        "not_a_course_code", ["012345", "track-ise", "", "960600a", "123456789"]
    )
    def test_anything_that_is_not_a_bare_seven_digit_run_is_untouched(
        self, not_a_course_code: str
    ) -> None:
        """Six-digit program codes, slugs, empties and nine-digit runs must pass
        through, so this is safe to run over ANY extracted identifier -- which is
        what `dispatch` does, on every field of every record."""
        assert canonical_course_code(not_a_course_code) == not_a_course_code


class TestScanningProseForCodes:
    def test_it_finds_codes_in_prose(self) -> None:
        assert course_codes_in("You passed 00940224 and 00960211.") == {"00940224", "00960211"}

    def test_it_ignores_numbers_that_are_not_course_codes(self) -> None:
        """Credits, grades and years share the digit alphabet; only the 8-digit
        shape is a course."""
        assert course_codes_in("You have 158.0 credits, a 92 average, and 2025 ahead.") == set()
        assert course_codes_in("") == set()


class TestTheDisplayNamePathResolves:
    """This class used to be `TestTheDisplayNamePathIsInert`, and asserted the
    opposite of what it does now.

    Its docstring read: "Pinned as behaviour, not left as a comment, because
    'returns None' and 'is broken' look identical from a call site." That
    observation was exactly right and the conclusion was backwards -- the path
    was not inert by design, it was broken by the port, and pinning it made the
    suite guard the bug. Every answer the agent gave carried bare 8-digit codes
    for as long as this passed.

    `_name_index` walked a graph engine this deployment does not have while the
    wiki corpus in the same process held 2601 course titles; `load_catalog_names`
    was never called. Both are wired now, so the assertion is inverted.
    """

    def test_a_wiki_name_resolves(self) -> None:
        assert course_display_name("00940224") == "Data Structures and Algorithms"

    def test_an_unknown_code_has_no_name(self) -> None:
        assert course_display_name("99999999") is None

    def test_a_code_in_neither_source_stays_bare(self) -> None:
        """00940226 is named by 259 prerequisite edges and appears in no catalog
        row and no wiki page. That is a gap in the DATA, and the honest
        rendering is the bare code rather than an invented name."""
        assert course_display_name("00940226") is None


class TestTheFallbackIsReadyIfItIsEverWired:
    def test_a_seeded_catalog_name_resolves(self) -> None:
        set_catalog_names({"03240305": "היסטוריה של המדע"})
        assert course_display_name("03240305") == "היסטוריה של המדע"

    def test_it_still_refuses_non_course_values(self) -> None:
        """The fallback must not widen what counts as a course code -- a grade or
        a credit total that happens to be a catalog key stays unresolvable."""
        set_catalog_names({"85": "not a course", "2025-1": "also not a course"})
        assert course_display_name("85") is None
        assert course_display_name("2025-1") is None

    def test_non_course_values_are_ignored_with_no_catalog_at_all(self) -> None:
        for value in ("85", "92.5", "2025-1", ""):
            assert course_display_name(value) is None


class TestTheWikiNameMustBeAboutTheSameCourse:
    """The wiki's code -> page mapping is wrong for about one course in fifteen.

    Measured across the 1752 codes both sources carry: 1433 titles match, 200
    differ only in spelling (ניתוח / נתוח, עיקרי / עקרי), and 119 -- 6.8% --
    are different courses entirely. 02180006 is "Doctoral Dissertation in
    Education" in the wiki and שיח בכתת המתמטיקה והמדעים in the catalog.

    Preferring the wiki wholesale therefore attached a WRONG name to a real
    code, which is the one outcome this module's docstring says is worse than
    no name -- nothing about it invites doubt. A live plan rendered
    "00940704 (Introduction to Data Engineering (Advanced))" for a course the
    catalog, `plan_term` and the transcript all call סדנת תכנות בשפת סי.

    The catalog is the authority because it is what every join in the system
    agrees on; the wiki supplies English only where it corroborates.
    """

    def test_a_conflicting_wiki_page_is_discarded(self) -> None:
        set_catalog_names({"00940704": "סדנת תכנות בשפת סי"})
        assert course_display_name("00940704") == "סדנת תכנות בשפת סי"

    def test_an_agreeing_wiki_page_supplies_the_english_name(self) -> None:
        set_catalog_names({"00940224": "מבני נתונים ואלגוריתמים"})
        assert course_display_name("00940224") == "Data Structures and Algorithms"

    def test_a_spelling_variant_still_counts_as_the_same_course(self) -> None:
        """ניתוח / נתוח is one course, and 200 codes differ only like this."""
        set_catalog_names({"00140004": "נתוח מערכות"})
        assert course_display_name("00140004") == "Systems Analysis"

    def test_an_english_catalog_title_corroborates_too(self) -> None:
        """A catalog title is sometimes English, so the Hebrew half alone
        would score zero against it and reject a correct match."""
        set_catalog_names({"00960620": "Introduction to Human Factors Engineering"})
        assert course_display_name("00960620") == "Human Factors Engineering"

    def test_with_no_catalog_row_the_wiki_stands_uncontradicted(self) -> None:
        set_catalog_names({})
        assert course_display_name("00940224") == "Data Structures and Algorithms"
