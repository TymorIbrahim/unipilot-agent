"""`no additional credit` is symmetric. It is not transitive.

The planner's only hard exclusion drops a course from a plan outright, so it has
to be read exactly as the catalog states it. Two failures, one in each direction,
both found on real data for the demo student:

  TOO LITTLE -- `courses.noAdditionalCreditText` was never added to the Supabase
  schema, so `build_catalog_overlap_groups` saw no text, built no groups, and
  the exclusion could not fire at all. A senior 129.5 credits into a 155-credit
  degree was scheduled 5 credits of 01040065, which the catalog says grants no
  additional credit alongside 01040016 -- a course they had already passed.

  TOO MUCH -- with the column restored, merging the groups into equivalence
  classes asserted the transitive closure and refused 00970317 to a student who
  had passed 00960570. Neither course names the other; they were fused because
  both happen to mention 01060173.

Both directions are pinned here. The second matters more than it looks: the
first failure is visible the moment a human reads a plan, and the second one
looks exactly like the planner being careful.
"""

from __future__ import annotations

from app.services.catalog_overlap_groups import (
    build_catalog_overlap_groups,
    overlap_group_for_course,
)


def course(number: str, overlap: str | None = None) -> dict:
    return {"courseNumber": number, "noAdditionalCreditText": overlap}


# The real rows, verbatim from the Technion catalog.
ALGEBRA = [
    course("01040065", "01040016 01040064"),
    course("01040016", "01040064 01040065"),
    course("01040166", "01040066 01040167"),
]
GAME_THEORY = [
    course("00970317", "01060173"),
    course("00960570", "00940204 00960575 01060173"),
]


class TestTheRelationIsRead:
    def test_a_declared_overlap_is_found(self) -> None:
        group = overlap_group_for_course("01040065", build_catalog_overlap_groups(ALGEBRA))
        assert group is not None and "01040016" in group

    def test_it_is_symmetric_even_when_stated_once(self) -> None:
        """Only 01040065's row names 01040064; nothing names 01040065 back from
        01040064's side. The relation still has to hold in both directions."""
        groups = build_catalog_overlap_groups([course("01040065", "01040064")])
        assert "01040065" in (overlap_group_for_course("01040064", groups) or set())

    def test_a_course_naming_nothing_has_no_group(self) -> None:
        assert overlap_group_for_course("00960211", build_catalog_overlap_groups(ALGEBRA)) is None

    def test_no_text_anywhere_means_no_groups(self) -> None:
        """The dead-rule state: this is what a missing column produces, and it
        has to stay distinguishable from 'nothing overlaps'."""
        assert build_catalog_overlap_groups([course("01040065"), course("01040016")]) == []


class TestItIsNotTransitive:
    def test_two_courses_sharing_a_partner_do_not_conflict(self) -> None:
        """The false exclusion. 00970317 and 00960570 both name 01060173 and
        neither names the other, so passing one must not block the other."""
        groups = build_catalog_overlap_groups(GAME_THEORY)
        group = overlap_group_for_course("00970317", groups)
        assert group is not None, "its own declared partner should still be found"
        assert "01060173" in group
        assert "00960570" not in group, "fused through a shared partner it never named"

    def test_the_chain_stops_at_one_hop(self) -> None:
        # Real-shaped course numbers: the text parser only recognises those, so
        # placeholders like "A1" silently yield no groups at all and the test
        # would pass against a completely broken implementation.
        groups = build_catalog_overlap_groups([
            course("09990001", "09990002"),
            course("09990002", "09990003"),
            course("09990003", "09990004"),
        ])
        group = overlap_group_for_course("09990001", groups)
        assert group == frozenset({"09990001", "09990002"}), f"followed the chain too far: {group}"

    def test_being_named_by_another_course_still_counts(self) -> None:
        """One hop, but in BOTH directions -- the union of every row mentioning
        it, not the first one found, which made the answer depend on row order."""
        groups = build_catalog_overlap_groups([
            course("09990010", "09990011"),
            course("09990012", "09990010"),
        ])
        assert overlap_group_for_course("09990010", groups) == frozenset(
            {"09990010", "09990011", "09990012"}
        )


class TestOrderDoesNotChangeTheAnswer:
    def test_the_same_catalog_shuffled_gives_the_same_group(self) -> None:
        rows = ALGEBRA + GAME_THEORY
        first = overlap_group_for_course("01040065", build_catalog_overlap_groups(rows))
        second = overlap_group_for_course("01040065", build_catalog_overlap_groups(rows[::-1]))
        assert first == second
