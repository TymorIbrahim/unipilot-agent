"""Course-code canonicalisation, and an honest note about the display names.

The last of `tests/pending_supabase/`. Only part of it ported, and the part that
did not is worth saying out loud.

WHAT IS LIVE. `canonical_course_code` restores the leading zero the wiki drops,
and it is load-bearing: `dispatch` runs every identifier through it before a
`plan_term` candidate list is built, so a seven-digit label from prose and an
eight-digit code from the catalog name the same course. Get it wrong in either
direction and a course silently fails to match -- or a six-digit program code
becomes an eight-digit course that does not exist.

WHAT IS NOT. The suite's other half tested `course_display_name` resolving codes
to English names from the wiki. That path cannot fire in this deployment:
`_name_index` reads an in-process graph engine that does not exist here and
returns `{}` unconditionally, `load_catalog_names()` has no call site, and so
`course_display_name` returns None for every input. Everything downstream falls
back to the bare code, which is correct behaviour and exactly the readability
problem the module was written to solve.

That is not repaired here -- it is a dead path, not a broken one, and the names
a student actually sees in a plan come from `plan_term`'s own `courseTitle`.
What is pinned is the FALLBACK, so the day someone wires `load_catalog_names`
the tests say what it has to do, and so nobody reads the module and assumes it
is already doing it.
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


class TestTheDisplayNamePathIsInert:
    """Pinned as behaviour, not left as a comment, because "returns None" and
    "is broken" look identical from a call site."""

    def test_no_name_resolves_without_a_catalog(self) -> None:
        """`_name_index` needs a graph engine this deployment does not have, and
        nothing calls `load_catalog_names`, so every lookup falls back."""
        assert course_display_name("00940224") is None
        assert course_display_name("00960211") is None

    def test_an_unknown_code_has_no_name_either(self) -> None:
        assert course_display_name("99999999") is None


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
