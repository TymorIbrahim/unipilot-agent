"""The prerequisite warning has to reach the student in words.

`plan_term` flags a placed course whose prerequisites are unmet -- a flag that
was stamped "satisfied" on everything until it was fixed. A `:detail` slot
prints every field of a placed course, so the repaired flag arrived in a live
plan as:

    number 00940412 · name הסתברות מ · credits 4 · prereq check_prerequisites

which is the warning showing up in a vocabulary only the planner speaks. The
student is not eligible for that course and has no way to tell from that line.
"""

from __future__ import annotations

from app.agent_core.facts.dispatch import _placed_collection


def _plan(**overrides) -> dict:
    placed = {
        "courseNumber": "00940412",
        "credits": 4.0,
        "category": "mandatory",
        "prereqStatus": "satisfied",
        "coreqStatus": "none",
        "courseTitle": "הסתברות מ",
    }
    placed.update(overrides)
    return {"terms": [{"semesterCode": "winter", "placedCourses": [placed]}]}


def _status(field: str, **overrides) -> str:
    record = _placed_collection(_plan(**overrides)).records[0]
    return record.fields[field].value


class TestTheWarningIsReadable:
    def test_an_unmet_prerequisite_says_so_in_words(self) -> None:
        status = _status("prereqStatus", prereqStatus="check_prerequisites")
        assert "check_prerequisites" not in status
        assert "NOT met" in status

    def test_a_met_prerequisite_is_plain(self) -> None:
        assert _status("prereqStatus", prereqStatus="satisfied") == "met"

    def test_corequisite_status_is_not_shown_at_all(self) -> None:
        """It said "none" for every course in this deployment, because
        `corequisitesText` is not a column, and "none" reads to a student as
        "this one has no corequisites" rather than "never checked"."""
        record = _placed_collection(_plan()).records[0]
        assert "coreqStatus" not in record.fields

    def test_its_wording_survives_for_when_the_field_is_seeded(self) -> None:
        """Kept deliberately: the planner still computes the status, and the day
        `corequisitesText` is seeded this is what it has to say."""
        from app.agent_core.facts.dispatch import _READABLE_STATUS

        assert "NOT met" in _READABLE_STATUS["check_corequisites"]


class TestUnknownStatusesDegradeRatherThanVanish:
    def test_an_unmapped_status_survives_unchanged(self) -> None:
        """A status added to the planner later must degrade to its raw name, not
        disappear -- a blank where a warning belongs is the worse failure."""
        assert _status("prereqStatus", prereqStatus="some_new_status") == "some_new_status"

    def test_an_empty_status_stays_empty(self) -> None:
        assert _status("prereqStatus", prereqStatus="") == ""


class TestThePromptAndTheWordingCannotDrift:
    """They already did, once, in the direction that matters most.

    The prompt told the model to flag a course whose `prereqStatus` reads
    "check_prerequisites". Rewording the fact to "NOT met -- ..." left that
    instruction pointing at a string that no longer exists, so the model filtered
    for it, found none, and wrote "No prerequisite issues were flagged (0)" in a
    plan whose own detail lines said NOT met. A contradiction in one answer, and
    the safe-looking half was the wrong half.
    """

    def test_the_prompt_matches_what_the_fact_actually_says(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT
        from app.agent_core.facts.dispatch import _READABLE_STATUS

        unmet = _READABLE_STATUS["check_prerequisites"]
        assert unmet.startswith("NOT met")
        assert "NOT met" in SYSTEM_PROMPT, (
            "the prompt must name the text the fact really carries, or the model "
            "filters for a value that never appears and reports zero problems"
        )

    def test_the_prompt_no_longer_names_the_raw_enum(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert '"check_prerequisites"' not in SYSTEM_PROMPT


class TestThePlannerKeepsItsOwnVocabulary:
    def test_the_mapping_happens_at_the_presentation_boundary(self) -> None:
        """`prereqStatus` is the planner's internal enum and its own tests assert
        on it; only the fact admitted for the model is reworded."""
        from app.planning.term_plan import _prereq_status

        assert _prereq_status(
            {"courseNumber": "X"}, set(), {}, {"X": [["A"]]}, set()
        ) == "check_prerequisites"


class TestTheCatalogTellsTheTruthAboutTheShape:
    """`plan_term`'s catalog entry names the fields and says the plan must be
    projected before rendering, because a live run spent TWO of its eight turns
    discovering that by being refused. Guidance that drifts from the record is
    worse than none: it would send the model to project fields that are not
    there."""

    def test_the_advertised_fields_are_the_actual_fields(self) -> None:
        from app.agent_core.facts.catalog import COMPOSITES

        spec = next(s for s in COMPOSITES if s.name == "plan_term")
        actual = set(_placed_collection(_plan()).records[0].fields)
        for field in actual:
            assert field in spec.when, f"{field} is returned but the catalog never names it"

    def test_the_plan_is_still_too_wide_to_render_raw(self) -> None:
        """The premise of the advice. If a record ever narrows to within the cap
        the guidance becomes noise and should be deleted, not left to rot."""
        from app.agent_core.facts.answer import _MAX_DETAIL_FIELDS

        widest = len(_placed_collection(_plan()).records[0].fields)
        assert widest > _MAX_DETAIL_FIELDS, (
            f"a placed course now carries {widest} fields, within the {_MAX_DETAIL_FIELDS} "
            "cap -- `plan_term`'s 'PROJECT IT BEFORE YOU RENDER IT' note is now false"
        )
