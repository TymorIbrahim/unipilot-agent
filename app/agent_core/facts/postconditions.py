"""Deterministic post-conditions over a finished plan answer -- the oracle the
grounding invariant is missing.

`resolve_answer` proves every number in an answer is a real derived fact: it
checks PROVENANCE. It cannot check that the number is SANE (a grade of -82), nor
that it answers the question actually asked rather than a subtly different one (a
threshold that holds the floor for one course in isolation, when the courses are
taken together). Those are the two ways a genuine, correctly-sourced fact still
makes a wrong answer, and nothing downstream catches either today -- a live
winter run shipped six per-course minimums, two of them negative, that would in
fact drop the student's GPA to 65 against an 80 floor if earned together.

These checks close that gap for the one shape we have seen fail: per-course
minimum grades that hold a GPA floor. They are PURE ARITHMETIC on numbers already
derived -- no model call, no I/O -- so they are cheap enough to run on every
answer and exact enough to trust as a gate. They are written to serve two callers
without change: the loop, as an independent verify step that hands a failure back
as a loud reason to retry; and the eval harness, as a pass/fail oracle over a
saved run. A caught violation names WHAT is wrong and WHY, in the same voice the
loop already feeds back, because a reason a model cannot act on wastes the retry.
"""

from __future__ import annotations

import math
import re

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

MIN_GRADE = 0.0
MAX_GRADE = 100.0

_PERIOD_NOUN = r"semesters?|terms?|years?|סמסטרים|סמסטר|שנים|שנה"
"""The unit every period check turns on, in both languages the agent answers in.

Nothing said which language to answer in, and asked in Hebrew the model answers
in Hebrew -- nine of ten did. Every prose check here was an English-only regex,
so they did not fail on those answers, they went SILENT on them, which is the
worse failure: `check_count_states_its_basis` was built because three runs
shipped "2 semesters" with no basis, and the Hebrew twin shipped
"נשארו לך 2 סמסטרים" past a guard that was looking for the word "semesters".

Kept in ONE place rather than spelled out in each regex below, because the cost
of the bug was three checks going dark at once and the fix should not be three
edits that can drift apart.
"""

MAX_TERM_CREDITS = 40.0
"""A sanity ceiling on ONE term's planned credits -- roughly twice a full load.
A term far above this is not a heavy schedule; it is the `optimize` output's
"(unscheduled)" overflow swept into the plan because the placed rows were never
selected out. A live winter plan put 27 courses / 83 credits in one term this
way. Like the 0-100 grade bound, this is a range check, not a policy: it catches
a number no real semester reaches."""

_FLOOR_EPSILON = 1e-6
"""Float slack when replaying a threshold. A minimum computed to land EXACTLY on
the floor must not be flagged for landing there -- only for landing below it."""


@dataclass(frozen=True)
class Standing:
    """The student's record BEFORE the planned term(s).

    `total_points` is the sum of grade*credits over completed courses; `gpa` is
    its ratio to `total_credits`. The floor is measured against the standing
    AFTER the plan, never against this -- this is only the starting point.
    """

    total_points: float
    total_credits: float

    @property
    def gpa(self) -> float:
        return self.total_points / self.total_credits


@dataclass(frozen=True)
class GradedCourse:
    """A planned course, and the minimum grade the answer claims holds the floor.

    `min_grade` is None on an ORDINARY plan row -- a term plan carries a course,
    its credits and its category, and no minimum at all. It used to be required,
    which meant every check here was reachable only from the min-grade planner
    and an ordinary plan was never verified: a live "how many semesters" answer
    put 23 credits in one term against an 18 cap, unexamined. The load checks
    need only `credits`; the grade checks skip a row with nothing to judge.
    """

    code: str
    credits: float
    min_grade: float | None = None
    term: str | None = None
    """Which term this row belongs to, when the collection spans several.

    A multi-semester answer slots a SUMMARY -- one row per term with that term's
    total -- and summing those against a per-semester cap compares the whole
    plan to one semester's limit."""


@dataclass(frozen=True)
class Violation:
    """One way a plan answer is unsound, phrased loudly enough to hand back to the
    loop as the reason to try again. `kind` is a stable tag for the harness to
    count by; `message` is student-of-the-model prose, not a code."""

    kind: str
    message: str


_GROUP_ID = re.compile(r"\b\d{6,8}\.\d+\b")
_EDGE_ID = re.compile(r"\b\d{6,8}\s*->\s*\d{6,8}\b")
_OBJECT_ID = re.compile(r"\b[0-9a-f]{24}\b", re.IGNORECASE)
"""A Mongo-style `_id`, carried over into Postgres as the join key.

Deliberately exact at 24 hex characters. Nothing a student-facing answer
legitimately contains has this shape: course numbers are 6-8 digits, grades and
credits are small decimals, semester codes are "YYYY-N"."""


def check_no_group_identifiers(text: str) -> list[Violation]:
    """A prerequisite GROUP id must never be shown as if it were a course.

    Groups are labelled `<course>.<n>` -- `00970800.0`, `00970800.1` -- which is
    bookkeeping, not something a student can register for. A live answer read
    "the alternatives I derived are 00970800.0, 00970800.1", naming two things
    that do not exist instead of the four course codes behind them.

    Nothing else could catch it. The tokens were slotted from a real fact, so the
    answer boundary passed them, and they LOOK like course codes to a reader --
    which is precisely what makes them worse than a visible error.

    The prompt already says to name alternatives by their `requires` codes and
    mostly does; this is for when it does not.
    """
    found = _GROUP_ID.findall(text)
    if not found:
        return []
    return [
        Violation(
            kind="group_identifier_shown",
            message=(
                f"the answer shows {', '.join(sorted(set(found))[:3])}, which are prerequisite "
                "GROUP labels, not courses. A student cannot register for those. Project the "
                "`requires` field of the edges and name the actual course codes, grouping them "
                "as the choices they are."
            ),
        )
    ]


_CLAIMED_PASS = re.compile(
    r"\byou\s+(?:already\s+|have\s+|had\s+|previously\s+)*passed\s+(?:course\s+)?(\d{6,8})\b",
    re.IGNORECASE,
)


_CLAIMED_PASS_PRONOUN = re.compile(
    r"\byou\s+(?:already\s+|have\s+|had\s+|previously\s+)*passed\s+it\b"
    r"|\b(?:כבר\s+)?עברת\s+(?:אותו|את\s+הקורס)\b",
    re.IGNORECASE,
)
"""The same claim with a pronoun where the code should be.

`_CLAIMED_PASS` binds to a code the answer states, and the model does not always
state one. Live, asked "am I eligible to take 00960211?":

    "No -- the course exists in the catalog, and you already passed it in 41
     recorded attempts, so you are not eligible to take it again."

The student has NOT passed 00960211; 41 is their total passed-course count. The
identical defect the named form was written for, wearing a pronoun, and it sailed
through because the guard was looking for digits next to the verb."""


def check_claimed_pass_is_on_the_transcript(
    text: str, passed_codes: "Sequence[str]", question: str = ""
) -> list[Violation]:
    """Telling a student they passed a course they have not is the worst answer
    this system can give, and every other gate passes it.

    Live: "If I fail 00970800, how does that change my graduation timeline?" ->

        You already passed 00970800 in 41 transcript rows, so failing it on a
        re-take would not change your graduation timeline. A passed course does
        not add any further credit when repeated.

    00970800 is one of this student's six REMAINING mandatory courses. The
    `find` carried no course filter, so the fact -- which the model named
    `passed_00970800` -- held all 41 of their passed courses, and 41 is a real,
    correctly derived number. The grounding invariant is satisfied: the digit
    came from a fact. What is false is the sentence around it.

    This is the failure `HeldFact.derivation` was added for, one step further
    on. Showing "41 (read from passed_courses)" beside the answer lets a
    DEVELOPER see the mistake; the student reading "you already passed 00970800"
    cannot. So the claim is checked against the codes actually on the transcript.

    Skipped when no passed-courses fact is held -- an unverifiable claim is not
    blocked, only a contradicted one, which is the rule every check here follows.
    """
    if not passed_codes:
        return []
    known = {str(code) for code in passed_codes}

    # "you already passed it" -- the pronoun's referent is the course the
    # QUESTION named, and only when it named exactly one, so an answer
    # comparing two courses is never guessed at.
    asked = set(_CODE.findall(question or ""))
    if len(asked) == 1 and _CLAIMED_PASS_PRONOUN.search(text or ""):
        code = asked.pop()
        if code not in known:
            return [
                Violation(
                    "unearned_pass",
                    f"the answer says the student already passed {code}, and {code} is not among "
                    f"the {len(known)} courses on their transcript. A `find` on `passed_courses` "
                    "filtered only by userId returns EVERY course they passed, so its COUNT is "
                    "their transcript length, not evidence about this course. Filter by the "
                    "course number too, and if no row comes back say plainly they have not "
                    "taken it.",
                )
            ]

    for match in _CLAIMED_PASS.finditer(text or ""):
        code = match.group(1)
        if code not in known:
            return [
                Violation(
                    "unearned_pass",
                    f"the answer says the student passed {code}, and {code} is not among the "
                    f"{len(known)} courses on their transcript. Check the fact you are citing "
                    "actually concerns that course: a `find` on `passed_courses` filtered only by "
                    "userId returns EVERY course they passed, whatever you named it. Filter by "
                    "the course number too, or say plainly that they have not taken it.",
                )
            ]
    return []


_JOIN_SIDE_LABEL = re.compile(r"\b(?:left|right)\.\w+")


def check_no_join_side_labels(text: str) -> list[Violation]:
    """A joined record's internal field names must not reach the student.

    `join` prefixes its inputs to keep them apart -- `left.requires`,
    `right.title` -- and `:detail` prints whatever a record carries. Live, asked
    which remaining course unlocks the most others:

        - left.requires 01040017 · right.title הסתברות מ · unlocked 1

    Three labels from the query engine and none from the domain. The prompt has
    said "ALWAYS project BEFORE :detail" for a while; this is the same rule
    where it can be enforced rather than requested.

    Same family as the edge ids and the ObjectId: an internal artifact that
    reads as plausible because a real value sits next to it.
    """
    found = _JOIN_SIDE_LABEL.findall(text or "")
    if not found:
        return []
    return [
        Violation(
            "join_side_label",
            f"the answer shows '{found[0]}', which is a field name from the join rather than "
            "anything a student would recognise. `project` the joined rows to the two to four "
            "columns the answer needs, giving each a plain name, and slot THAT.",
        )
    ]


def check_no_object_identifiers(text: str) -> list[Violation]:
    """A database `_id` must never stand in for the thing it identifies.

    Live, on the deployed agent, "Register me for 00960211 right now." came back:

        I've prepared a request to register 6a3db0e382df7b7cb04552e8.

    That is `courses._id`. The student asked about 00960211, and the proposal
    they are being invited to confirm names a token they cannot look up, cannot
    check, and cannot match to the course they asked for.

    Intermittent -- the same prompt run locally named 00960211 correctly -- and
    that is the argument for catching it in code rather than in the prompt: a
    fault that appears in one run of two is one no amount of instruction reliably
    removes, and the answer layer is the last place it can be seen at all.

    24 hex characters is unambiguous here. Course numbers are 6-8 digits, grades
    and credits are small decimals, and semester codes are "YYYY-N", so nothing
    a student answer legitimately contains has this shape.
    """
    found = _OBJECT_ID.findall(text or "")
    if not found:
        return []
    return [
        Violation(
            "object_identifier",
            f"the answer shows the internal database id {found[0]} where a name or code belongs. "
            "A student cannot look that up or match it to what they asked about. Join to the "
            "catalog and slot the `courseNumber` (or the title) instead -- the `_id` is a key "
            "for joining, never something to show.",
        )
    ]


def check_no_edge_identifiers(text: str) -> list[Violation]:
    """An EDGE id must never stand in for the course it points at.

    `prerequisite_edges` rows are identified as `<course>-><requires>`. A live
    answer read "any one of the course codes in 00960211->00940224,
    00960211->00940226" -- the right two prerequisites, named as internal edge
    keys a student cannot look up, let alone register for.

    The prompt has warned against rendering edge rows into a sentence since
    before today, and it still happens. It is worth catching in code because it
    reads as PLAUSIBLE: the real course code is sitting inside the token, so the
    sentence looks specific and technical rather than broken -- and a substring
    check for the code even passes.
    """
    found = _EDGE_ID.findall(text)
    if not found:
        return []
    return [
        Violation(
            kind="edge_identifier_shown",
            message=(
                f"the answer shows {', '.join(sorted(set(found))[:3])}, which are prerequisite "
                "EDGE ids, not courses. Name the course on the RIGHT of the arrow: `project` the "
                "edges' `requires` field and slot that, grouped as the choices they are."
            ),
        )
    ]


_ALTERNATIVES = re.compile(
    r"(?:any one of|any of|one of|either|alternatives?|options?|choose from|choice of)"
    r"[:\s]+((?:\d{6,8}\s*(?:,|or|and)\s*)+\d{6,8})",
    re.IGNORECASE,
)
_CODE = re.compile(r"\b\d{6,8}\b")

_REPEATED_IN_A_LIST = re.compile(r"\b(\d{6,8})\b(?:\s*(?:,|or|and)\s*\1\b)+")
"""The same course code listed against itself, whatever words introduce it.

The lead-in vocabulary above was `any one of|one of|either`, and a live answer
said "with alternatives 00960211, 00960211" -- degenerate, and missed, because
`alternatives` was not in the list. That is the `prereqStatus` drift again: a
check keyed on prose and prose that got reworded.

This needs no lead-in. A list of one code repeated is never right in any
sentence: not as alternatives, not as prerequisites, not as courses to take."""


def check_alternatives_are_distinct(text: str, question: str = "") -> list[Violation]:
    """A choice between prerequisites must be a choice between DIFFERENT courses.

    A live answer read "you meet 1 of 1 prerequisite groups. The course requires
    1 requirement: any one of 00960211, 00960211" -- the course being asked
    about, listed twice as its own prerequisite. The edges were projected on
    `course` instead of `requires`, so every alternative collapsed onto the
    target.

    Two things make it checkable without knowing the curriculum: a choice whose
    options are all the same course is not a choice, and no course is its own
    prerequisite. Both hold for every course in every catalog, so this needs no
    data -- which is what makes it safe to run on every answer.
    """
    asked = set(_CODE.findall(question or ""))
    violations: list[Violation] = []

    # Phrasing-independent first, so a reworded lead-in cannot hide it.
    for repeated in _REPEATED_IN_A_LIST.findall(text or ""):
        violations.append(
            Violation(
                "degenerate_alternatives",
                f"the answer offers a choice between {repeated} and itself. Alternatives "
                "come from the edges' `requires` field -- projecting `course` collapses every "
                "option onto the course being asked about.",
            )
        )
    if violations:
        return violations

    for listing in _ALTERNATIVES.findall(text or ""):
        options = _CODE.findall(listing)
        if len(options) > 1 and len(set(options)) == 1:
            violations.append(
                Violation(
                    "degenerate_alternatives",
                    f"the answer offers a choice between {options[0]} and itself. Alternatives "
                    "come from the edges' `requires` field -- projecting `course` collapses every "
                    "option onto the course being asked about.",
                )
            )
        elif asked & set(options):
            violations.append(
                Violation(
                    "self_prerequisite",
                    f"the answer lists {sorted(asked & set(options))[0]} among its own "
                    "prerequisites. No course requires itself; that is the `course` side of the "
                    "edge, not the `requires` side.",
                )
            )
    return violations


_FRACTIONAL_PERIOD = re.compile(
    rf"(\d+\.\d+)\s+({_PERIOD_NOUN})\b", re.IGNORECASE
)


_MET_NONE = re.compile(
    r"\bmet?(?:s|ting)?\s+0\b|\b0\s+of\s+(\d+)\s+prerequisite\s+groups?\b|\bmeet 0 of\b",
    re.IGNORECASE,
)
_CLAIMS_ELIGIBLE = re.compile(
    r"\b(?:you\s+are|are)\s+eligible\b|\beligible:\s*yes\b|\byou\s+(?:can|may)\s+take\b",
    re.IGNORECASE,
)
_CLAIMS_INELIGIBLE = re.compile(
    r"\bnot\s+eligible\b|\bineligible\b|\beligible:\s*no\b", re.IGNORECASE
)


def check_eligibility_is_not_self_contradictory(
    text: str, question: str = ""
) -> list[Violation]:
    """An answer must not count zero met groups and then declare eligibility.

    Live answer, shipped:

        "No. You checked 1 prerequisite group and met 0, so you are eligible to
         take 01040174."

    Both halves are in one sentence and they are opposites. Worse than a plainly
    wrong answer: a reader who skims the front takes "No" and a reader who skims
    the end takes "eligible", and the run that produced it scored as CORRECT
    because the checker read only the leading word.

    Scoped by the QUESTION, not by the answer. The first version skipped any
    answer naming more than one course code, to spare "you are eligible for
    00960211 but you meet 0 of 1 groups for 01040174" -- two courses, two
    verdicts, no contradiction. That exemption disabled the check entirely: a
    good eligibility answer ALWAYS names the target course and the prerequisites
    that would satisfy it, so every real case has three or more codes in it. The
    guard went in, looked correct, and never fired once. It let this through:

        "You are eligible for 01040174, because you meet 0 of 1 prerequisite
         groups. To make it yes, pass any one of 01040066, 01040166."

    Counting what the QUESTION asks about instead gets both right. One course
    asked -> every verdict in the answer is about it, so "eligible" beside "met
    0" is a contradiction. Two or more asked -> which clause owns which verdict
    needs a parser this does not have, and blocking a correct answer is the
    worse error, so it stands aside.
    """
    body = text or ""
    if len(set(_CODE.findall(question or ""))) != 1:
        return []
    if not _MET_NONE.search(body):
        return []
    if _CLAIMS_INELIGIBLE.search(body):
        return []  # "you met 0 ... so you are NOT eligible" is the coherent form
    if not _CLAIMS_ELIGIBLE.search(body):
        return []
    return [
        Violation(
            "contradictory_eligibility",
            "the answer says 0 prerequisite groups are met AND that the student is eligible. "
            "Those are opposites. Meeting 0 of the required groups means NOT eligible -- say "
            "that, and name the prerequisite that is missing.",
        )
    ]


def check_periods_are_whole(text: str) -> list[Violation]:
    """You cannot take 0.42 of a semester.

    Asked how many semesters remained, the agent answered "you have 1.42
    semesters remaining at your current max load" -- 25.5 credits over an
    18-credit cap, reported as the raw quotient. The arithmetic is right and the
    sentence is not: a semester is indivisible, so the honest reading of 1.42 is
    "at least 2".

    Wrong in the optimistic direction, which is why it matters: a student reading
    1.42 hears "nearly done in one more term" when they need two. Both runs of
    that question answered this way, so it is the shape of the answer rather than
    a slip.

    Credits, grades and GPAs are untouched -- those are genuinely continuous.
    Only a count of PERIODS is flagged, because only periods cannot be part-taken.
    """
    found = _FRACTIONAL_PERIOD.findall(text or "")
    if not found:
        return []
    value, unit = found[0]
    whole = math.ceil(float(value))
    return [
        Violation(
            "fractional_period",
            f"the answer reports {value} {unit}, and a {unit.rstrip('s')} cannot be part-taken. "
            f"Round UP -- {whole} -- and say 'at least {whole}': the remainder is a term the "
            "student still has to attend, not a fraction they can skip.",
        )
    ]


def check_gpa_in_range(gpa: float) -> list[Violation]:
    """A GPA is a ratio in (0, 100]. Above 100 is the classic slotting slip --
    total_points landed where the ratio belonged."""
    if gpa <= 0.0 or gpa > MAX_GRADE:
        return [
            Violation(
                "gpa_range",
                f"GPA is {gpa:.3f}, outside the possible 0-100 range. A GPA above 100 usually "
                "means a total (total_points) was slotted where the ratio (total_points/total_credits) "
                "belonged.",
            )
        ]
    return []


def check_grades_in_range(courses: Sequence[GradedCourse]) -> list[Violation]:
    """Every minimum grade must be a grade a student could actually earn: 0-100.

    A NEGATIVE minimum is not wrong arithmetic -- it is a real signal, reported
    as the wrong thing: it means even a 0 holds the floor, so the honest report
    is 0 ("any passing grade"), never the negative number itself. An ABOVE-100
    minimum means the floor cannot be held by that course alone.
    """
    out: list[Violation] = []
    for course in courses:
        if course.min_grade is None:
            continue  # an ordinary plan row claims no minimum, so none can be wrong
        if course.min_grade < MIN_GRADE:
            out.append(
                Violation(
                    "min_grade_range",
                    f"{course.code}: min_grade {course.min_grade:g} is below 0 -- no grade is "
                    "negative. It means even a 0 in this course would hold the floor, so report 0 "
                    "(or 'any passing grade'), not a negative number.",
                )
            )
        elif course.min_grade > MAX_GRADE:
            out.append(
                Violation(
                    "min_grade_range",
                    f"{course.code}: min_grade {course.min_grade:g} exceeds 100 -- no grade is above "
                    "100, so the floor cannot be held by this course alone.",
                )
            )
    return out


def check_term_load(term_credits: float, term_label: str = "a term") -> list[Violation]:
    """One planned term must not exceed a plausible semester load. Over the ceiling
    means the raw `optimize` output (placed rows PLUS its "(unscheduled)" overflow)
    was scored instead of just the placed rows -- so the whole remaining catalog
    landed in one term."""
    if term_credits > MAX_TERM_CREDITS:
        return [
            Violation(
                "term_load",
                f"{term_label} totals {term_credits:g} credits -- no single semester is that large. "
                "The plan is built from the raw optimize output including its '(unscheduled)' overflow; "
                "select slot == the term FIRST and derive credits, min_grade and the listing from those "
                "placed rows only.",
            )
        ]
    return []


def check_term_within_cap(
    term_credits: float, cap: float, term_label: str = "a term"
) -> list[Violation]:
    """A planned term must not exceed the student's OWN per-semester limit.

    Distinct from `check_term_load`, which is a sanity ceiling at 40 credits --
    "a number no real semester reaches", written to catch an 83-credit term made
    of `optimize` overflow. It is a range check by design, and 23 credits sails
    through it.

    This is the POLICY check, and it had no equivalent until now: nothing
    compared a plan to `student_profiles.maxCreditsPerSemester`. A live answer
    to "how many semesters will it take me to graduate" reported "Winter -- 23
    credits" against a cap of 18, and no guard looked, because the value was not
    a fact the answer layer could reach. It is one now -- the profile is seeded
    at the start of every run -- so the check is finally possible.

    The usual cause is not an over-full term but two terms collapsed into one:
    `plan_term` tags placed courses with the term NAME, so asking for
    ["winter", "spring", "winter"] returns two winters that a later
    `select term == "winter"` merges. The message says so, because a model told
    only "too many credits" drops a course instead of splitting the term.
    """
    if cap <= 0 or term_credits <= cap + _FLOOR_EPSILON:
        return []
    return [
        Violation(
            "term_over_cap",
            f"{term_label} totals {term_credits:g} credits, over this student's limit of "
            f"{cap:g} per semester. If you asked `plan_term` for the same term name twice, "
            "the two came back under one label and a `select` on it merged them -- give each "
            "term a distinct name, and pass `max_credits` so the planner enforces the cap "
            "rather than leaving it to be noticed here.",
        )
    ]


_NARRATES_CATALOG = re.compile(
    r"\b(?:it|the course|this course|\d{6,8})\s+"
    r"(?:exists?|is\s+(?:present|listed|found))\s+(?:in|on)\s+the\s+catalog"
    r"|\bexists?\s+in\s+the\s+catalog\b"
    r"|\b(?:is|are)\s+in\s+the\s+catalog\b"
    r"|\bfound\s+(?:it\s+)?in\s+the\s+catalog\b",
    re.IGNORECASE,
)
_NARRATES_CATALOG_HE = re.compile(
    r"\bהקורס\s+(?:קיים|נמצא|מופיע)\s+בקטלוג"
    r"|\b(?:קיים|נמצא|מופיע)\s+בקטלוג\b",
)
_DENIES_CATALOG = re.compile(
    r"\b(?:not|n[o']t)\s+(?:be\s+)?(?:in|on|exists?|listed|found|appears?)\b"
    r"[^.;!?]{0,25}\bcatalog\b"
    r"|\b(?:could|can)\s+n[o']t\s+find\b[^.;!?]{0,40}\bcatalog\b"
    r"|\bלא\s+(?:קיים|נמצא|מופיע)\b[^.;!?]{0,20}בקטלוג",
    re.IGNORECASE,
)
"""A denial of EXISTENCE, which is the one case where the catalog is the answer.

Twice too loose before this. First it asked whether "not" appeared anywhere
within 30 characters of the word "catalog"; then whether a negator sat within 12
characters before the phrase. A leading VERDICT satisfies both --

    "No -- the course exists in the catalog, and you already passed it in 41
     recorded attempts, so you are not eligible to take it again."

-- so the check exempted itself on precisely the answer it was written for, and
that answer also told the student they had passed a course they had not.

The negation has to belong to the existence PREDICATE ("is not in", "does not
exist", "could not find"), not merely appear near it. A verdict elsewhere in the
sentence is not a claim about the catalog."""


def check_answer_does_not_narrate_the_catalog_lookup(text: str) -> list[Violation]:
    """A successful existence check is not news, and it is crowding out the answer.

    The prompt tells the model to confirm a named course exists before reasoning
    about it, and the model reports having done so. Two live answers to "will
    00940412 be offered next spring?":

        "00940412 exists in the catalog, and yes."
        "Yes -- it exists in the catalog, and yes."

    The student typed the course number, so they know it exists; the verdict is
    the last word of both, and the second is also why `_tidy_affirmations` could
    not repair the doubled yes -- that repair only spans separators, and here a
    whole clause sits between the two.

    Only the AFFIRMATIVE narration is refused. "00999999 is not in the catalog"
    is the case where existence IS the answer, and it must survive: reasoning on
    past an empty lookup is how a course that does not exist got a confident
    eligibility verdict.

    Asking in the prompt did not work -- it was added in the same voice as the
    rules that do hold and the next runs narrated anyway -- which is the whole
    reason this file exists.
    """
    body = text or ""
    match = _NARRATES_CATALOG.search(body) or _NARRATES_CATALOG_HE.search(body)
    if match is None:
        return []
    if _DENIES_CATALOG.search(body):
        return []  # "00999999 is NOT in the catalog" -- there, existence IS the answer
    return [
        Violation(
            "narrates_the_catalog_lookup",
            "the answer reports that the course exists in the catalog. The student named it, "
            "so that is not news -- and it is standing where the answer should be. Confirming "
            "existence is a precondition for your REASONING, not a sentence in the reply. Lead "
            "with the verdict and follow it with the basis you derived; mention the catalog "
            "only when the lookup FAILED, because then it is the answer.",
        )
    ]


_PREREQ_UNMET_CLAIM = re.compile(
    r"(?:prerequisites?|prereqs?)\b[^.;]{0,40}?\b(?:are|is|were|remain)?\s*"
    r"(?:still\s+)?(?:not|un)[- ]?(?:yet\s+)?(?:satisfied|met|fulfilled|complete)"
    r"|(?:not|n[o']t)\s+(?:yet\s+)?(?:satisfied|met)\b[^.;]{0,30}?"
    r"\b(?:prerequisites?|prereqs?)\b",
    re.IGNORECASE,
)


def check_prereq_verdict_matches_the_edges(
    text: str, satisfied_courses: Mapping[str, str]
) -> list[Violation]:
    """An answer must not declare prerequisites unmet for a course that meets them.

    Live, and wrong in the direction that costs a student a semester:

        "Before you can take 00970135, you need to complete 00960324 first. I
         also checked 00960324 itself, and its prerequisites are not yet
         satisfied by your passed courses, so the chain stops there for now."

    00960324 requires ANY ONE of 00940314 or 00980413, and the student passed
    00940314 with a 57. They were eligible to register that day and were told
    the chain stopped.

    The run had already fetched both halves -- 00960324's edges and the passed
    courses -- and then asserted the verdict in prose instead of computing it.
    That is the hole this closes: the grounding invariant refuses a typed
    DIGIT, so "you need 25.5 credits" cannot be invented, while "its
    prerequisites are not satisfied" is words and costs nothing to make up. The
    strongest guarantee in the system does not reach the claims that carry no
    number, and an eligibility verdict is exactly such a claim.

    `satisfied_courses` maps a course code to the passed course that satisfies
    it, computed from the typed edges the run holds -- so this replays the
    derivation the model skipped rather than trusting either side's prose.

    Scoped to the code NEAREST BEFORE the claim, because an answer legitimately
    naming two courses ("00970135 is blocked; 00960324 is not") would otherwise
    be refused for the one it got right.
    """
    body = text or ""
    claim = _PREREQ_UNMET_CLAIM.search(body)
    if claim is None or not satisfied_courses:
        return []
    codes = _CODE.findall(body[: claim.start()])
    if not codes:
        return []
    subject = codes[-1]
    satisfied_by = satisfied_courses.get(subject)
    if satisfied_by is None:
        return []
    return [
        Violation(
            "prereq_verdict_contradicts_the_edges",
            f"the answer says {subject}'s prerequisites are not satisfied, and the edges you "
            f"hold say they ARE -- {subject} needs any one of its group, and {satisfied_by} is "
            "on the transcript. Do not assert an eligibility verdict you did not compute: "
            "`distinct` the edges on `group`, `select` the ones whose `requires` is in "
            f"passed_courses, `distinct` those on `group`, and compare the counts. Then say "
            f"{subject} IS takeable and name {satisfied_by} as the reason.",
        )
    ]


_NEGATED_ZERO = re.compile(
    # `\w*n['\u2019]t` rather than `n[o']t`: in "aren't" and "doesn't" the n is
    # mid-word, so a leading \b never matches and the contraction slips past.
    r"(?:\b(?:no|not|never|אין|לא)\b|\w*n['\u2019]t\b)"
    r"[^.;!?\n]{0,14}?(?<![\d.])0(?![\d.])",
    re.IGNORECASE,
)


_ASKS_GPA_RANKING = re.compile(
    r"(?:which|what)\b[^.?]{0,60}?\b(?:raise|rais|improve|boost|increase|help|lift)\w*\b"
    r"[^.?]{0,25}?\b(?:gpa|average|grade point)\b"
    r"|\b(?:gpa|average)\b[^.?]{0,30}?\b(?:the most|most|best|highest)\b"
    r"|\b(?:best|better)\b[^.?]{0,25}?\bfor my (?:gpa|average)\b"
    r"|\bאיזה\b[^.?]{0,40}?\b(?:יעלה|ישפר|ירים)\b[^.?]{0,25}?\bממוצע"
    r"|\bממוצע\b[^.?]{0,30}?\bהכי\b",
    re.IGNORECASE,
)
_DISCLAIMS_THE_FUTURE = re.compile(
    r"\bcan ?n[o']?t\b|\bcannot\b|\bunable\b|\bno way to\b|\bnot (?:possible|derivable|"
    r"something I can|able)\b|\bdepends on\b[^.]{0,40}\bgrade\b|\bwithout knowing\b"
    r"|\bfuture grade\b|\bnot a record\b|\bhave not been earned\b|\bhaven[’']?t been earned\b"
    r"|\bאי אפשר\b|\bלא ניתן\b|\bתלוי ב\b[^.]{0,30}\bציון\b|\bאיני יכול\b",
    re.IGNORECASE,
)


def check_no_ranking_by_an_unearned_grade(text: str, question: str = "") -> list[Violation]:
    """Courses cannot be ordered by a grade the student has not earned yet.

    Live, asked "which courses next semester would raise my GPA the most?":

        "These are the courses next semester that would raise your GPA the most:
         - course 00960620 · credits 3.5
         - course 00960606 · credits 3
         - course 00970325 · credits 3
         ..."

    That is the course list sorted by CREDITS, presented as a GPA-impact
    ranking. GPA impact is grade times credits, and the grade is not a record --
    it does not exist yet for any of them. The agent took the one field it had,
    ordered by it, and labelled the result something else.

    Every other gate passes it, which is why it needs its own. The credits are
    real derived facts. No digit was typed. No course was invented. Only the
    CLAIM ABOUT WHAT THE ORDERING MEANS is fabricated, and nothing else here
    examines a claim that carries no number of its own.

    Worse than a wrong number, because it is actionable: a student reads the top
    of that list and registers, believing it was computed.

    An answer that says plainly it cannot be derived passes -- that is the
    correct response, and offering what CAN be done (the credit weighting, a
    plan) alongside it is encouraged rather than penalised.
    """
    if not _ASKS_GPA_RANKING.search(question or ""):
        return []
    body = text or ""
    if _DISCLAIMS_THE_FUTURE.search(body):
        return []
    if len(set(_CODE.findall(body))) < 2:
        return []  # not presenting an ordering of courses
    return [
        Violation(
            "ranked_by_an_unearned_grade",
            "the answer orders courses by how much they would raise the GPA, and no fact you "
            "hold can support that order: GPA impact is grade x credits, and the grade does "
            "not exist yet for a course not taken. Sorting by credits and calling it GPA "
            "impact is the ranking invented. Say plainly that it cannot be derived from the "
            "record, then offer what CAN be: that credits weight a grade's effect, or a plan "
            "for the term.",
        )
    ]


def check_a_zero_count_is_not_also_negated(text: str) -> list[Violation]:
    """A count of zero IS the negative. Negating it says the opposite.

    Live, asked in Hebrew whether a course had been passed:

        "לא — אין לך 0 רשומות מעבר לקורס המבוקש."
        ("No -- you do NOT have 0 passing records for that course.")

    Every part is derived and the sentence means the reverse of what it is for:
    read literally, not having zero records is having some. The model wrote a
    negation around a `{count}` slot without knowing the slot would render 0 --
    it cannot know, it never sees the value -- so the phrasing works for every
    count except the one that actually came back.

    Nothing else catches it. The digit came from a real fact, so the grounding
    invariant is satisfied; the verdict "לא" is correct, so the stance checks
    pass; and it reads as fluent prose. Only the arithmetic of the sentence is
    wrong.

    Scoped tight: the negation must sit within a few characters BEFORE the zero.
    "you meet 0 of 1 groups, so you are not eligible" puts the negation after,
    and is the coherent form this must never touch.
    """
    found = _NEGATED_ZERO.search(text or "")
    if not found:
        return []
    return [
        Violation(
            "negated_zero",
            f"'{found.group(0).strip()}' negates a count that is already zero, so the sentence "
            "claims the opposite of what you mean -- not having 0 records is having some. A "
            "zero count needs no negation: say it plainly ('you have not taken it', 'no "
            "matching records'), or state the count without the negative around it.",
        )
    ]


def check_term_within_requested_cap(
    term_credits: float, requested: float, term_label: str = "a term"
) -> list[Violation]:
    """A limit the STUDENT named in the request outranks the one on their profile.

    Asked "I'm working part-time next semester, so keep it under 10 credits",
    the agent returned a 16-credit term. The profile cap is 18, so
    `check_term_within_cap` passed it, `check_term_load`'s 40-credit ceiling
    passed it, and the answer presented the plan as the recommendation. Nothing
    in the system had any idea 10 had been said.

    `plan_term` can enforce this -- it takes `max_credits` -- and the prompt
    asks the model to pass it "if the request names a different limit". That is
    a judgement call made once per run, and on the Hebrew phrasing of the same
    question it was made wrong three times out of three. So it is checked, for
    the same reason every other check in this file exists.

    Distinct from `check_term_within_cap` in WHOSE limit it holds. The profile
    cap is what the student can normally carry; this is what they have just said
    they can carry THIS term, and it is the only one of the two the student is
    actively watching. Advice that ignores it is worse than no advice: a student
    who asked for a part-time load and got a full one either notices and stops
    trusting the tool, or does not and over-commits.
    """
    if requested <= 0 or term_credits <= requested + _FLOOR_EPSILON:
        return []
    return [
        Violation(
            "term_over_requested_cap",
            f"{term_label} totals {term_credits:g} credits, and the question asked for no more "
            f"than {requested:g}. The limit in the REQUEST wins over the student's profile cap -- "
            f"they have just told you what they can carry this term. Pass max_credits={requested:g} "
            "to `plan_term` and let it seat a plan that fits, rather than reporting one that does "
            "not. Do not simply drop courses from this list by hand: the planner is what checks "
            "offerings, conflicts and prerequisites.",
        )
    ]


_PERIOD_COUNT = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight"
    r"|שני|שניים|שתי|שלושה|שלוש|ארבעה|ארבע|חמישה|חמש|שישה|שש)\s+"
    r"(?:more\s+|additional\s+|further\s+|extra\s+|עוד\s+)?"
    r"(?:semesters?|terms?|סמסטרים|סמסטר)\b",
    re.IGNORECASE,
)


def _states(text: str, number: float) -> bool:
    """Whether this number really appears, not merely its digits inside another.

    Bounded the same way the eval scorer's `states_number` is, and for the same
    reason: 155 must not satisfy a check for 15, while a number ENDING A
    SENTENCE still counts."""
    needle = f"{number:g}"
    return re.search(rf"(?<![\d.]){re.escape(needle)}(?!\d)(?!\.\d)", text or "") is not None


_ASKS_HOW_LONG = re.compile(
    r"how (?:many|long)\b.{0,40}\b(?:semester|term|year)s?\b"
    r"|\bhow long\b.{0,40}\b(?:graduat|finish|degree|left)"
    r"|\b(?:when|by when)\b.{0,30}\bgraduat"
    r"|\b(?:semesters?|terms?)\b.{0,20}\b(?:to|until|left|remain).{0,20}\bgraduat"
    r"|\bgraduat\w*\b.{0,40}\bhow (?:many|long)\b"
    # Hebrew, scoped exactly as tightly as the English above. "כמה" alone is
    # NOT enough: "כמה זמן יש לי לערער על ציון" is an appeal-window question,
    # and arming this on it is how the English version started refusing three
    # correct policy answers for not citing a credit gap they never needed.
    r"|\bכמה\b.{0,20}\b(?:סמסטרים|סמסטר|שנים)\b"
    r"|\bכמה זמן\b.{0,30}\b(?:נשאר|לסיים|לסיום|התואר|תואר)"
    r"|\bמתי\b.{0,30}\b(?:אסיים|לסיים|סיום|מסיים)\b"
    r"|\b(?:נשארו|נשאר)\b.{0,20}\b(?:סמסטרים|סמסטר)\b",
    re.IGNORECASE | re.DOTALL,
)


def check_count_states_its_basis(
    text: str, remaining_required: float, question: str = ""
) -> list[Violation]:
    """A count of semesters must carry the credits it was derived from.

    Three live runs answered "It will take you 2 semesters to graduate", then
    listed the terms. The count is right and the student cannot check it: the
    same sentence would be produced by a correct derivation, by counting however
    many terms the planner happened to fill, and by a guess. All three runs held
    25.5, 155 and 129.5 throughout and slotted none of them.

    Asking for it in the prompt did not work -- it was added to the system
    prompt in the same voice as the rules that do hold, and the next three runs
    answered exactly as before. So it is checked, which is what this file is for.

    Fires only when the requirement is actually HELD, so an answer that could
    not have cited it is not punished for the omission. This is not a style
    rule: a number a reader cannot verify is the shape every wrong answer in
    this project has taken, and the credits are what separate a derivation from
    a coincidence.
    """
    # Only when the QUESTION asks how long the student has left. Without that
    # gate this fired on any answer quoting a rule that mentions a number of
    # semesters -- and the regulations are full of them: "the two semesters
    # immediately following", "by the end of the 4th semester", "from semester
    # 13". Three policy questions were refused for not stating a 25.5-credit gap
    # that had nothing to do with what they asked.
    #
    # It became always-armed when the credit standing started being seeded:
    # `remaining_required` used to be absent on a question that never fetched
    # credits, which had been doing the gating by accident.
    if not _ASKS_HOW_LONG.search(question or ""):
        return []
    if remaining_required <= 0 or not _PERIOD_COUNT.search(text or ""):
        return []
    if _states(text, remaining_required):
        return []
    return [
        Violation(
            "count_without_basis",
            f"the answer says how many semesters but never states the {remaining_required:g} "
            "credits it follows from, so the student cannot check it. Add a slot for the fact "
            "holding the credits still needed, next to the count -- name YOUR fact, whatever "
            "you called it. Do not invent a fact name to slot: if you did not derive the gap "
            "under a name you hold, derive it first.",
        )
    ]


def check_plan_within_requirement(
    planned_credits: float, remaining_required: float, cap: float
) -> list[Violation]:
    """A plan must not schedule so far past the requirement that it costs a term.

    The gap this closes: every other check here is PER TERM. Each of them passed
    on a plan that placed 38.5 credits toward a 25.5-credit requirement, because
    each term inside it was legal (16, 12, 7, 3.5 against an 18 cap). Nothing
    looked at the plan whole, so scheduling thirteen credits the student does not
    need was invisible -- and it is not a cosmetic excess, it is what turned the
    answer "2 semesters" into "4 semesters" on the same data, in the same hour.

    The cause is the older "remaining means two different things" defect, one
    layer further in. This student has 21 unfinished track courses worth 50.0
    credits but needs only 25.5 more to graduate: 17.5 of mandatory, and then
    ANY 8 credits of the 32.5 on offer. Electives are a quota, not a work list.
    Handing the whole remaining set to the planner schedules all of it.

    Deliberately tolerant, because overshoot on its own is not an error: courses
    are indivisible, so the last one taken almost always carries the total past
    the requirement, and a plan of 28 credits against 25.5 is simply what it
    costs to finish. Only an overshoot big enough to CHANGE THE TERM COUNT is
    flagged -- ceil(planned/cap) > ceil(required/cap) -- which is precisely the
    overshoot that reaches the student as a wrong answer. A guard that fired on
    every 2.5-credit excess would refuse correct plans, and this file has done
    that before.
    """
    if cap <= 0 or remaining_required <= 0 or planned_credits <= remaining_required:
        return []
    needed_terms = math.ceil((remaining_required - _FLOOR_EPSILON) / cap)
    planned_terms = math.ceil((planned_credits - _FLOOR_EPSILON) / cap)
    if planned_terms <= needed_terms:
        return []
    excess = planned_credits - remaining_required
    return [
        Violation(
            "plan_exceeds_requirement",
            f"the plan schedules {planned_credits:g} credits, but only {remaining_required:g} "
            f"remain to graduate -- {excess:g} more than the degree needs, which stretches it "
            f"from {needed_terms} semester(s) to {planned_terms}. You are placing every "
            "unfinished course in the track. Take all the MANDATORY ones, then add electives "
            f"only until the total reaches {remaining_required:g}; the rest are choices this "
            "student never has to make. Then the term count follows from the credits that "
            "actually have to be earned.",
        )
    ]


def check_joint_floor(
    standing: Standing,
    courses: Sequence[GradedCourse],
    floor: float,
    *,
    epsilon: float = _FLOOR_EPSILON,
) -> list[Violation]:
    """Earning EXACTLY each reported minimum in ALL the planned courses at once
    must still hold the floor.

    This is the check a per-course threshold cannot pass by luck: each minimum is
    computed so its course ALONE holds the floor, but the student takes them
    TOGETHER. When the current GPA already clears the floor, the isolated minimums
    run low -- even negative -- and earning all of them at once can drop the GPA
    well below the floor the answer promised to hold. Replaying the whole plan is
    the only way to see that.
    """
    added_credits = sum(course.credits for course in courses)
    total_credits = standing.total_credits + added_credits
    if total_credits <= 0.0:
        return []  # No credits to average over -- no GPA exists to check.

    added_points = sum(course.min_grade * course.credits for course in courses)
    joint_gpa = (standing.total_points + added_points) / total_credits
    if joint_gpa < floor - epsilon:
        return [
            Violation(
                "joint_floor",
                f"Earning each course's stated minimum at once gives GPA {joint_gpa:.2f}, below the "
                f"{floor:g} floor the plan promised to hold. The minimums were computed per course in "
                "isolation -- each holds the floor alone, but the courses are taken together, so the "
                "thresholds must be solved jointly.",
            )
        ]
    return []


def check_plan(
    standing: Standing,
    courses: Sequence[GradedCourse],
    floor: float,
) -> list[Violation]:
    """All post-conditions for a min-grade plan answer, in one pass.

    Order is by how directly a violation points at its cause: a bad GPA poisons
    every threshold below it, an out-of-range grade is a local slip, and the
    joint floor is the whole plan judged together.
    """
    return [
        *check_gpa_in_range(standing.gpa),
        *check_grades_in_range(courses),
        *check_joint_floor(standing, courses, floor),
    ]


__all__ = [
    "MAX_GRADE",
    "MAX_TERM_CREDITS",
    "MIN_GRADE",
    "GradedCourse",
    "Standing",
    "Violation",
    "check_gpa_in_range",
    "check_grades_in_range",
    "check_joint_floor",
    "check_plan",
    "check_term_load",
]
