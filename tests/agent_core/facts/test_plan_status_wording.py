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

    def test_corequisites_get_the_same_treatment(self) -> None:
        status = _status("coreqStatus", coreqStatus="check_corequisites")
        assert "check_corequisites" not in status
        assert "NOT met" in status


class TestUnknownStatusesDegradeRatherThanVanish:
    def test_an_unmapped_status_survives_unchanged(self) -> None:
        """A status added to the planner later must degrade to its raw name, not
        disappear -- a blank where a warning belongs is the worse failure."""
        assert _status("prereqStatus", prereqStatus="some_new_status") == "some_new_status"

    def test_an_empty_status_stays_empty(self) -> None:
        assert _status("prereqStatus", prereqStatus="") == ""


class TestThePlannerKeepsItsOwnVocabulary:
    def test_the_mapping_happens_at_the_presentation_boundary(self) -> None:
        """`prereqStatus` is the planner's internal enum and its own tests assert
        on it; only the fact admitted for the model is reworded."""
        from app.planning.term_plan import _prereq_status

        assert _prereq_status(
            {"courseNumber": "X"}, set(), {}, {"X": [["A"]]}, set()
        ) == "check_prerequisites"
