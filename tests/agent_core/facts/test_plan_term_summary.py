"""`plan_term` knew its per-term totals and made the model rebuild them.

Traced on three live planning runs, every one spent a turn after `plan_term` on:

    compute: distinct on term
    compute: select term == "winter"; select term == "spring"; sum credits

-- regrouping rows the planner had just grouped. At ~15s a turn against a 60s
ceiling that is a quarter of the budget, and it is also where the numbers can
DISAGREE with the plan: a live run selected `term == "winter"` against a plan
holding two winters and reported their sum as a single 23-credit term, over an
18-credit cap.

So the summary comes back as a second fact, named off the caller's own `as`.
"""

from __future__ import annotations

from app.agent_core.facts.dispatch import _term_totals

PLAN = {
    "terms": [
        {"semesterCode": "2026-1", "placedCourses": [
            {"courseNumber": "00940704", "credits": 1.5},
            {"courseNumber": "00960578", "credits": 2.5},
            {"courseNumber": "00960606", "credits": 3.0},
        ]},
        {"semesterCode": "2026-2", "placedCourses": [
            {"courseNumber": "00960211", "credits": 3.5},
            {"courseNumber": "00970800", "credits": 3.5},
        ]},
    ],
    "unscheduled": [{"courseNumber": "00960221"}],
}


def _rows(collection):
    return [{k: v.value for k, v in r.fields.items()} for r in collection.records]


class TestTheSummaryMatchesThePlan:
    def test_one_row_per_term(self) -> None:
        rows = _rows(_term_totals(PLAN))
        assert [r["term"] for r in rows] == ["2026-1", "2026-2"]

    def test_the_credit_totals_are_the_sums_of_the_placed_rows(self) -> None:
        rows = _rows(_term_totals(PLAN))
        assert rows[0]["credits"] == 7.0
        assert rows[1]["credits"] == 7.0

    def test_the_course_counts_are_right(self) -> None:
        rows = _rows(_term_totals(PLAN))
        assert [r["courses"] for r in rows] == [3.0, 2.0]

    def test_it_is_complete(self) -> None:
        """A count over it must not be refused for truncation -- it is the whole
        plan, and the plan is what the answer reports."""
        assert _term_totals(PLAN).completeness.complete is True

    def test_unscheduled_courses_are_not_a_term(self) -> None:
        """The `(unscheduled)` overflow is not a semester. Counting it as one is
        how a plan grew a phantom term."""
        assert len(_term_totals(PLAN).records) == 2


class TestWhenThereIsNothingToSummarise:
    def test_a_plan_that_placed_nothing_has_no_summary(self) -> None:
        """An empty summary reads as "no semesters", which is a claim; None
        leaves the fact absent, which is the truth."""
        assert _term_totals({"terms": [], "unscheduled": []}) is None

    def test_a_term_with_no_placed_courses_is_skipped(self) -> None:
        plan = {"terms": [
            {"semesterCode": "2026-1", "placedCourses": []},
            {"semesterCode": "2026-2", "placedCourses": [{"courseNumber": "x", "credits": 3.0}]},
        ]}
        rows = _rows(_term_totals(plan))
        assert [r["term"] for r in rows] == ["2026-2"]

    def test_all_terms_empty_is_none(self) -> None:
        assert _term_totals({"terms": [{"semesterCode": "2026-1", "placedCourses": []}]}) is None


class TestTheRecipeSaysToUseIt:
    def test_it_is_documented(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "_by_term" in SYSTEM_PROMPT

    def test_the_recipe_warns_against_rebuilding_it(self) -> None:
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "do NOT rebuild it" in SYSTEM_PROMPT
