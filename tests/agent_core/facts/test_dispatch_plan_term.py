"""Dispatch of `plan_term` -- the one tool that reaches the api planner over HTTP.

The heavy planning is api-side (covered by the api service's term_plan tests). Here
we test the agent-side seam: the student id is taken from the `me` fact, the call is
forwarded to the internal client, and the PLACED courses come back as a SIMULATED
collection whose `credits` the model reads off -- or a loud DEFECT when the id or
settings are missing, or the service refuses. `fetch_term_plan` is always mocked;
no HTTP happens.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext, dispatch
from app.agent_core.facts.types import Basis, Collection, Record, Scalar, ScalarKind
from app.clients.internal_api_client import InternalApiClientError

_FETCH = "app.agent_core.facts.dispatch.fetch_term_plan"


def _context(*, me: str | None = "student-1", settings: object | None = object()) -> DispatchContext:
    facts = {}
    if me is not None:
        facts["me"] = HeldFact(value=Scalar(ScalarKind.IDENTIFIER, me), basis=Basis.OFFICIAL_RECORD)
    return DispatchContext(facts=facts, settings=settings)


def _held_courses(*rows: dict[str, str]) -> HeldFact:
    """A held collection of courses, the way the recipe hands candidates over."""
    records = tuple(
        Record(
            fields={key: Scalar(ScalarKind.TEXT, value) for key, value in row.items()},
            basis=Basis.OFFICIAL_RECORD,
        )
        for row in rows
    )
    return HeldFact(value=Collection(records=records), basis=Basis.OFFICIAL_RECORD)


def _call(**args) -> dict:
    return {
        "tool": "plan_term",
        "as": "winter",
        "args": {
            "terms": ["winter"],
            "candidates": [{"courseNumber": "00940412", "category": "mandatory"}],
            **args,
        },
    }


_RESULT = {
    "status": "ok",
    "maxCredits": 20.0,
    "terms": [
        {
            "semesterCode": "winter",
            "credits": 4.0,
            "placedCourses": [
                {
                    "courseNumber": "00940412",
                    "courseTitle": "A",
                    "credits": 4.0,
                    "category": "mandatory",
                    "prereqStatus": "satisfied",
                    "coreqStatus": "none",
                    "selectedLessonEvents": [],
                }
            ],
            "weeklySchedule": {"status": "valid", "conflicts": []},
            "examSummary": {"exams": [], "warnings": []},
        }
    ],
    "unscheduled": [{"courseNumber": "00940999", "reason": "Schedule conflict with a higher-priority course"}],
}


async def test_placed_courses_come_back_as_a_simulated_collection_with_credits():
    with patch(_FETCH, new_callable=AsyncMock, return_value=_RESULT) as fetch:
        result = await dispatch(_call(), _context())

    fetch.assert_awaited_once()
    assert result.defects == {}
    held = result.facts["winter"]
    assert held.basis == Basis.SIMULATED
    assert isinstance(held.value, Collection)
    (row,) = held.value.records
    assert row.fields["courseNumber"].value == "00940412"
    assert row.fields["credits"].value == 4.0
    assert row.fields["term"].value == "winter"


async def test_a_placed_course_without_a_title_falls_back_to_its_number():
    """courseTitle is always emitted so the answer step can `project` it safely."""
    result = {
        "status": "ok",
        "maxCredits": 20.0,
        "terms": [
            {
                "semesterCode": "winter",
                "credits": 3.0,
                "placedCourses": [
                    {"courseNumber": "00940412", "credits": 3.0, "category": "elective"}
                ],
            }
        ],
        "unscheduled": [],
    }
    with patch(_FETCH, new_callable=AsyncMock, return_value=result):
        out = await dispatch(_call(), _context())

    (row,) = out.facts["winter"].value.records
    assert row.fields["courseTitle"].value == "00940412"


async def test_the_student_id_is_taken_from_the_me_fact():
    with patch(_FETCH, new_callable=AsyncMock, return_value=_RESULT) as fetch:
        await dispatch(_call(), _context(me="stu-42"))

    assert fetch.await_args.kwargs["user_id"] == "stu-42"


async def test_a_bare_code_candidate_is_accepted_and_canonicalised():
    with patch(_FETCH, new_callable=AsyncMock, return_value=_RESULT) as fetch:
        await dispatch(
            {"tool": "plan_term", "as": "winter", "args": {"terms": ["winter"], "candidates": ["0940412"]}},
            _context(),
        )

    assert fetch.await_args.kwargs["candidates"] == [{"courseNumber": "00940412", "category": "elective"}]


async def test_candidates_can_name_a_held_collection_mapping_type_to_category():
    """The usual path: the model passes the NAME of its typed remaining-courses
    fact, and each record's `type` maps to a planning category."""
    context = _context()
    context.facts["remaining"] = _held_courses(
        {"courseNumber": "00940412", "type": "required"},
        {"courseNumber": "00940500", "type": "elective"},
    )
    with patch(_FETCH, new_callable=AsyncMock, return_value=_RESULT) as fetch:
        await dispatch(
            {"tool": "plan_term", "as": "winter", "args": {"terms": ["winter"], "candidates": "remaining"}},
            context,
        )

    assert fetch.await_args.kwargs["candidates"] == [
        {"courseNumber": "00940412", "category": "mandatory"},
        {"courseNumber": "00940500", "category": "elective"},
    ]


async def test_candidates_naming_an_unknown_collection_is_a_defect():
    with patch(_FETCH, new_callable=AsyncMock) as fetch:
        result = await dispatch(
            {"tool": "plan_term", "as": "winter", "args": {"terms": ["winter"], "candidates": "nope"}},
            _context(),
        )

    fetch.assert_not_awaited()
    assert "winter" in result.defects


async def test_a_missing_me_fact_is_a_defect_not_a_call():
    with patch(_FETCH, new_callable=AsyncMock) as fetch:
        result = await dispatch(_call(), _context(me=None))

    fetch.assert_not_awaited()
    assert "me" in result.defects["winter"].message


async def test_unconfigured_settings_is_a_defect_not_a_call():
    with patch(_FETCH, new_callable=AsyncMock) as fetch:
        result = await dispatch(_call(), _context(settings=None))

    fetch.assert_not_awaited()
    assert "winter" in result.defects


async def test_a_service_refusal_comes_back_as_a_defect():
    error = InternalApiClientError(status_code=400, detail="Invalid semesterCode: 2025-9")
    with patch(_FETCH, new_callable=AsyncMock, side_effect=error):
        result = await dispatch(_call(terms=["2025-9"]), _context())

    assert "Invalid semesterCode" in result.defects["winter"].message


async def test_empty_terms_is_a_defect_before_any_call():
    with patch(_FETCH, new_callable=AsyncMock) as fetch:
        result = await dispatch(
            {"tool": "plan_term", "as": "winter", "args": {"terms": [], "candidates": ["00940412"]}},
            _context(),
        )

    fetch.assert_not_awaited()
    assert "winter" in result.defects
