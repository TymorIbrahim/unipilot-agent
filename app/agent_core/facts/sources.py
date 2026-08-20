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
_B = ScalarKind.BOOL

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
        # Generated in Postgres, not stored upstream -- see db/schema.sql. A
        # failed course earns nothing, and `creditsEarned` does not always say
        # so: one live row is graded 30 and still carries its full 5.5.
        "creditsCounted": _Q,
        "passed": _B,
        "attempt": _Q,
        "source": _I,
        # NO courseNumber: it is not stored here. Reaching a course code means
        # joining to `courses` on courseId = _id.
    },
    field_notes={
        "creditsCounted": (
            "credits that COUNT toward the degree -- `creditsEarned` when the course was "
            "passed, else 0. SUM THIS for 'how many credits have I completed'. Summing "
            "`creditsEarned` instead counts courses the student FAILED and overstates the total. "
            "WEIGHT A GPA BY THIS TOO: gpa = sum(grade * creditsCounted) / sum(creditsCounted). "
            "Weighting by `creditsEarned` drags a failed grade into the average -- it reported "
            "72.64 where the real figure is 74.45."
        ),
        # Imperative, like `creditsCounted` above, and for the same measured
        # reason. This note used to say only what the column MEANS, and the
        # model read it, understood it, and still built "courses I have
        # completed" from every transcript row: 2 of 3 live runs told a student
        # they were eligible for 01040174 because they had ATTEMPTED 01040166,
        # which they failed with a 30. `creditsCounted`'s note tells the model
        # what to DO with it and is used correctly every time.
        "passed": (
            "false when the grade is below 55, the pass mark. FILTER ON THIS before treating a "
            "row as a course the student HAS. A transcript row is an ATTEMPT: a failed one "
            "counts toward nothing, satisfies no prerequisite, and must be re-taken. Deriving "
            "'the courses I have completed' without it reports a student ELIGIBLE for a course "
            "they cannot take."
        ),
        "creditsEarned": "as recorded by the registrar, INCLUDING some failed courses. Rarely what a question means.",
    },
    basis=Basis.OFFICIAL_RECORD,
    # The one route from a transcript row to a course code.
    joins=(("courseId", "courses._id"),),
    object_id_fields=frozenset({"courseId", "userId"}),
    # `courseId` repeats -- 70 duplicate groups live, from retakes and from one
    # course appearing on several students' transcripts. `_id` is undeclared, so
    # `find` never reads it as a fact; it exists purely to make the sort a total
    # order, without which a truncated fetch returns a different PAGE each run.
    order_tiebreak=("_id",),
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
        # NO academicYear: it was declared here and exists on 0 of 2,613 real
        # course documents (measured 2026-08-14). A declaration nothing can
        # satisfy is the same lie as a missing one -- the field would simply be
        # absent on every record, and any aggregate over it fails closed for a
        # reason that has nothing to do with the data. `course_offerings` is
        # where an academic year actually lives.
        "status": _I,
    },
    basis=Basis.OFFICIAL_RECORD,
    object_id_fields=frozenset({"_id"}),
)

STUDENT_PROFILES = SourceSchema(
    collection="student_profiles",
    # `userId`, not `institutionId`. The original key was a TENANT NAME: all 951
    # live profiles share one of two values ('technion', 'uni-main'), so every
    # profile had the same identity. `find` sorts by the key and `difference` /
    # `join` treat it as identity, which made a multi-profile fetch both
    # unordered and unable to tell two students apart. `userId` is unique per
    # profile and is what every query filters on anyway.
    key="userId",
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
    field_notes={
        "totalCredits": (
            "the credits the DEGREE requires. Credits still needed = totalCredits - "
            "sum(completed_courses.creditsCounted), and that subtraction is the ONLY way to get "
            "it. Summing the credits of the courses left in `track_courses` answers a different "
            "question -- a track lists more courses than the degree requires, because its "
            "electives are choices -- and gives 50.0 where the student needs 25.5."
        ),
    },
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
    field_notes={
        "academicYear": (
            "these are RECORDED offerings and the record ENDS at the present year. Filtering for a "
            "future year matches 0 rows because it has not happened yet -- that is silence, not "
            "evidence the course will not run. NEVER answer 'will it be offered' from a count of "
            "future rows: fetch the WHOLE history and pass it to `forecast`."
        ),
        "semesterName": (
            "spring / summer / winter -- the period `forecast` keys on. Give it as `period_path` "
            "with `cycle_path: academicYear`, so the rate is 'in how many YEARS did this term "
            "occur' and not 'what share of all offerings were this term'."
        ),
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

PREREQUISITE_EDGES = SourceSchema(
    collection="prerequisite_edges",
    key="edge",
    fields={
        "edge": _I,
        "course": _I,
        "requires": _I,
        "group": _I,
    },
    # Parsed out of catalog prose, not read from a registrar's record. Weaker
    # than OFFICIAL_RECORD, and the basis ordering carries that into any answer
    # built on it.
    basis=Basis.WIKI_DERIVED,
    # `course` and `requires` are course NUMBERS (00960211), not ObjectIds. A
    # live eval filtered `course` by a course's `_id` and got nothing back --
    # `completed_courses` keys courses by ObjectId while this keys them by
    # number, and the model had no way to know the two differ.
    joins=(("course", "courses.courseNumber"), ("requires", "courses.courseNumber")),
    field_notes={
        "requires": (
            "a course number that satisfies this group only if the student PASSED it. Compare "
            "against completed courses filtered on `passed`, never against every transcript "
            "row -- a failed attempt is not a satisfied prerequisite."
        ),
        "group": (
            "edges sharing a group are ALTERNATIVES: any ONE satisfies it. Different groups are "
            "each required. Count DISTINCT groups, not rows."
        ),
    },
    yields=frozenset({"edges"}),
    # 566 of 4,766 edges repeat an `edge` id across OR-branches: one course can
    # require another in two different alternative groups.
    order_tiebreak=("group",),
)

TRACK_COURSES = SourceSchema(
    collection="track_courses",
    key="edge",
    fields={
        "edge": _I,
        "track": _I,
        "course": _I,
        "category": _I,
    },
    field_notes={
        "category": (
            "\"mandatory\" or \"elective\", from the section of the track page that lists the "
            "course. USE THIS to type candidates for `plan_term` -- it is the same split you "
            "would otherwise reconstruct with search_corpus + two extract_list calls against the "
            "track page, which costs three or four turns and returns TRUNCATED collections. "
            "ABSENT where the page's headings do not say; fall back to the wiki only for those."
        ),
        "course": (
            "a course number this track LISTS -- candidates to choose among, not a set the "
            "student must complete in full. Their credits do NOT total the degree: use them to "
            "pick what to take, and `degree_programs.totalCredits` minus completed credits to say "
            "how much is LEFT."
        ),
    },
    basis=Basis.WIKI_DERIVED,
    # `course` is a course NUMBER; join to the catalog for credits and faculty.
    joins=(("course", "courses.courseNumber"),),
)

# Both of the above were `DerivedSchema`s computed at runtime from a networkx
# graph, and both are now ordinary tables -- materialised by `scripts/seed.py`.
# Nothing about how the model reaches them changed: it is still
# `{"tool": "find", "args": {"source": "prerequisite_edges"}}`. What changed is
# that a cold start no longer builds a graph to answer it.
#
# They are in the REGISTRY rather than added by `wiring.build_wiring`, which is
# the point: registration used to depend on a graph engine that this repo does
# not contain, so the `try` around it silently succeeded at registering nothing
# and `traverse` was advertised with no reachable input.
PASSED_COURSES = SourceSchema(
    collection="passed_courses",
    key="courseId",
    fields={
        "courseId": _I,
        "userId": _I,
        "courseNumber": _I,
        "title": _T,
        "grade": _Q,
        "creditsEarned": _Q,
        "creditsCounted": _Q,
        "semesterCode": _I,
        "attempt": _Q,
    },
    field_notes={
        "courseNumber": (
            "the course code, ALREADY joined from `courses`. Absent on a row whose catalog "
            "entry is missing -- which is unrecoverable, not a bug to route around: a "
            "comparison against it simply does not match, so eligibility comes back denied "
            "rather than granted."
        ),
        "grade": "always 55 or above here; failing attempts are not in this view.",
        "creditsCounted": (
            "credits that COUNT toward the degree. Every row here is ALSO a row in "
            "`completed_courses` -- this view is that table filtered, not a second set of "
            "results. Sum ONE of them, never both: adding the two together double-counts "
            "every passed course and reports a student past a requirement they have not met."
        ),
    },
    basis=Basis.OFFICIAL_RECORD,
    joins=(("courseNumber", "courses.courseNumber"), ("courseId", "courses._id")),
)
"""The transcript filtered to what COUNTS, with course numbers already joined.

A view over `completed_courses`, and the answer to a defect that appeared three
times in three consumers: credits (135 instead of 129.5), the term planner's
prerequisite flag, and finally the agent's own eligibility reasoning, which told
a student they met 1 of 1 prerequisite groups because they had ATTEMPTED the
course that satisfies it and been graded 30.

Every one of those was the same mistake -- reading a transcript row as a course
the student HAS -- and the first two were fixed by filtering at the point of
use. This is the third fix and the only structural one: a source that is already
filtered cannot be misused that way. Same reasoning as `creditsCounted`, which
exists so that "sum the credits" is right by default rather than right if you
remember a rule.

It also collapses a three-step dance the model performed on every eligibility
question -- find the transcript, find the catalog rows for it, project distinct
course numbers -- into one `find`.

`completed_courses` stays registered and is still the right source for anything
about ATTEMPTS: a failed course that must be re-taken, a grade history, how many
times something was sat. This view cannot answer those, by construction.
"""

REGISTRY: dict[str, SourceSchema] = {
    "passed_courses": PASSED_COURSES,
    "prerequisite_edges": PREREQUISITE_EDGES,
    "track_courses": TRACK_COURSES,
    "course_offerings": COURSE_OFFERINGS,
    "completed_courses": COMPLETED_COURSES,
    "semester_plans": SEMESTER_PLANS,
    "courses": COURSES,
    "student_profiles": STUDENT_PROFILES,
    "degree_programs": DEGREE_PROGRAMS,
    # NOT `moodle_grades`. It is operator-fed -- `services/operator` sweeps a
    # student's live Moodle to write it -- and nothing in this deployment runs
    # that sweep, so the table would be empty for every demo student. An empty
    # source advertised to the model is worse than an absent one: "what did I
    # score on assignment 2" comes back as a confident "no records" instead of
    # "I cannot see that". It declares no `yields`, so dropping it costs no
    # capability. The schema below is kept because it documents the one
    # non-registrar basis the layer models.
}


__all__ = [
    "COMPLETED_COURSES",
    "COURSE_OFFERINGS",
    "COURSES",
    "DEGREE_PROGRAMS",
    "MOODLE_GRADES",
    "PASSED_COURSES",
    "PREREQUISITE_EDGES",
    "REGISTRY",
    "SEMESTER_PLANS",
    "STUDENT_PROFILES",
    "TRACK_COURSES",
]
