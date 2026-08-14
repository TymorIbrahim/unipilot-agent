"""The source registry -- phase 11 of docs/agent/tools_implementation_plan.md.

`find` refuses to guess whether "3.5" is a quantity and "00940224" is an
identifier, so something has to declare it. This is that declaration.

**Derived from the stored documents, not from the API's input models.** The
first version of this file read `services/api/app/schemas/*.py` and was wrong in
four places, because an input model describes what an endpoint ACCEPTS, not what
Mongo holds. Checked against the live collections (2026-07-19):

  - `completed_courses.courseId` is an **ObjectId** referencing `courses._id`,
    not a course code. Declaring it a string made every transcript fetch fail at
    admission, since the key itself is one.
  - `completed_courses` carries **no `courseNumber` at all** (0 of 93). A course
    code is only reachable by joining to `courses` on that ObjectId -- which the
    old tool layer did in Python, and which is now an ordinary `join`.
  - `semester_plans` stores `semesters[]`, each with its own
    `plannedCourses[]`. There is no top-level `semesterCode` (0 of 247).
  - Plans DENORMALISE `courseNumber` into `plannedCourses[]` while transcripts
    do not, so the two need different routes to the same fact.

Verify with `tests/agent_core/facts/test_sources.py`, which checks these claims
against real documents rather than against this file.
"""

from __future__ import annotations

from app.agent_core.facts.find import ArrayOf, SourceSchema, Sub
from app.agent_core.facts.types import Basis, ScalarKind

_Q = ScalarKind.QUANTITY
_I = ScalarKind.IDENTIFIER
_T = ScalarKind.TEXT

COMPLETED_COURSES = SourceSchema(
    collection="completed_courses",
    # An ObjectId, and the only identity always present.
    key="courseId",
    fields={
        "courseId": _I,
        "userId": _I,
        "semesterCode": _I,
        "grade": _Q,
        "gradePoints": _Q,
        "creditsEarned": _Q,
        "attempt": _Q,
        "source": _I,
        # NO courseNumber: it is not stored here. Reaching a course code means
        # joining to `courses` on courseId = _id.
    },
    basis=Basis.OFFICIAL_RECORD,
    # The one route from a transcript row to a course code.
    joins=(("courseId", "courses._id"),),
    object_id_fields=frozenset({"courseId", "userId"}),
)

SEMESTER_PLANS = SourceSchema(
    collection="semester_plans",
    key="_id",
    fields={
        "_id": _I,
        "userId": _I,
        "name": _T,
        "plannerType": _I,
        "status": _I,
        "version": _Q,
        # `semesters[]` is an array of semesters, each holding its own
        # `plannedCourses[]`. Per-course questions need `unnest` twice --
        # which is exactly the nesting that operator exists for.
        #
        # Declared here at last. The comment above described this shape for a
        # while WITHOUT declaring it, so `find` skipped the field entirely and
        # the nesting `unnest` exists for could not be reached from any source.
        # A comment is not a declaration.
        "semesters": ArrayOf(
            Sub(
                {
                    # `order` is the slot index and `goalCredits` the slot
                    # capacity -- what `optimize` needs to place courses into
                    # semesters.
                    "semesterCode": _I,
                    "order": _Q,
                    "goalCredits": _Q,
                    "notes": _T,
                    "plannedCourses": ArrayOf(
                        Sub(
                            {
                                # Plans DENORMALISE the course code; transcripts
                                # do not. This is the one route to a course code
                                # that needs no join.
                                "courseId": _I,
                                "courseNumber": _I,
                                "courseTitle": _T,
                                "credits": _Q,
                                "category": _I,
                                "isActive": ScalarKind.BOOL,
                            }
                        )
                    ),
                }
            )
        ),
    },
    basis=Basis.OFFICIAL_RECORD,
    # `semesters[]` unnests into exactly what `optimize` calls slots: an ordered
    # sequence with a per-slot capacity. The route is find -> unnest, and
    # `test_reachability.py` walks it rather than trusting this declaration.
    yields=frozenset({"slots"}),
    object_id_fields=frozenset({"_id", "userId"}),
)

COURSES = SourceSchema(
    collection="courses",
    key="courseNumber",
    fields={
        # `_id` is the join target for a transcript's courseId.
        "_id": _I,
        "courseNumber": _I,
        "title": _T,
        "titleHebrew": _T,
        "credits": _Q,
        "faculty": _I,
        "studyFramework": _I,
        "catalogYear": _Q,
        "academicYear": _Q,
        "status": _I,
    },
    basis=Basis.OFFICIAL_RECORD,
    object_id_fields=frozenset({"_id"}),
)

STUDENT_PROFILES = SourceSchema(
    collection="student_profiles",
    key="institutionId",
    fields={
        "institutionId": _I,
        "userId": _I,
        "facultyId": _I,
        "programType": _I,
        "degreeId": _I,
        "programSlug": _I,
        "catalogYear": _Q,
        "currentSemesterCode": _I,
        "maxCreditsPerSemester": _Q,
    },
    basis=Basis.OFFICIAL_RECORD,
    joins=(("degreeId", "degree_programs._id"),),
    object_id_fields=frozenset({"userId", "degreeId"}),
)

DEGREE_PROGRAMS = SourceSchema(
    collection="degree_programs",
    key="_id",
    fields={"_id": _I, "name": _T, "totalCredits": _Q},
    basis=Basis.OFFICIAL_RECORD,
    object_id_fields=frozenset({"_id"}),
)

COURSE_OFFERINGS = SourceSchema(
    collection="course_offerings",
    key="_id",
    fields={
        "_id": _I,
        "courseNumber": _I,
        # `spring` / `summer` / `winter` -- the period vocabulary `forecast`
        # keys on, and the reason offering questions are answerable at all.
        "semesterName": _I,
        "semesterCode": _Q,
        "academicYear": _Q,
        "catalogVersion": _I,
        "status": _I,
    },
    basis=Basis.OFFICIAL_RECORD,
    joins=(("courseNumber", "courses.courseNumber"),),
    object_id_fields=frozenset({"_id"}),
)

# The first source that is not the registrar's OFFICIAL_RECORD. `services/operator` sweeps the
# student's live Moodle and writes one `moodle_grades` document per submitted ASSIGNMENT (not per
# course -- Moodle's course "total" is a gradebook aggregate, not the real course grade) in this exact
# shape and vocabulary (camelCase, `courseNumber` a string) precisely so it joins `courses` the same
# way `course_offerings` does. `basis=LIVE_MOODLE` sits just below OFFICIAL_RECORD, so where a live
# grade and a registrar grade meet, `weakest()` correctly keeps the weaker Moodle basis. `observedAt`
# (datetime) and `outOf` are deliberately UNDECLARED: an undeclared field is simply not read, and the
# fact layer has no datetime scalar -- declaring `observedAt` would fail the coerce check.
MOODLE_GRADES = SourceSchema(
    collection="moodle_grades",
    # `_id` is always present and unique; the course code is reached by joining `courses`, exactly as
    # `course_offerings` does. A course has many assignments, so the course code alone is not a key --
    # the business identity is (userId, courseNumber, assignment).
    key="_id",
    fields={
        "_id": _I,
        "userId": _I,
        "courseNumber": _I,
        "courseTitle": _T,
        "assignment": _T,  # the assignment name, e.g. "תרגיל בית 1" -- free text like courseTitle
        "term": _I,  # winter / spring / summer -- a category, like course_offerings.semesterName
        "termIndex": _Q,
        "grade": _Q,  # 0-100; an ungraded assignment stores null and is simply absent from a grade query
    },
    basis=Basis.LIVE_MOODLE,
    joins=(("courseNumber", "courses.courseNumber"),),
    object_id_fields=frozenset({"_id", "userId"}),
)

REGISTRY: dict[str, SourceSchema] = {
    "course_offerings": COURSE_OFFERINGS,
    "completed_courses": COMPLETED_COURSES,
    "semester_plans": SEMESTER_PLANS,
    "courses": COURSES,
    "student_profiles": STUDENT_PROFILES,
    "degree_programs": DEGREE_PROGRAMS,
    "moodle_grades": MOODLE_GRADES,
}


__all__ = [
    "COMPLETED_COURSES",
    "COURSE_OFFERINGS",
    "COURSES",
    "DEGREE_PROGRAMS",
    "MOODLE_GRADES",
    "REGISTRY",
    "SEMESTER_PLANS",
    "STUDENT_PROFILES",
]
