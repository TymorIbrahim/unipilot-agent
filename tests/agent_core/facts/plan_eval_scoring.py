"""Score a saved plan-eval run against the deterministic post-conditions.

The live planning tests are diagnostic -- they save a transcript and answer to
`agent_planning_eval/` but assert almost nothing, so a correct plan and the
negative-min-grade plan pass identically. This turns those saved runs into
PASS/FAIL by replaying the post-conditions (`facts.postconditions`) over what the
answer actually claimed.

The extraction here is the brittle part, and it is kept OUT of production on
purpose: the checks are exact arithmetic that the loop will one day run on live
facts, but reading numbers back out of a finished answer is prose-parsing, which
only the harness needs. When an input cannot be recovered, the affected check is
reported SKIPPED with a reason -- never silently passed. A run that could not be
scored is not a run that passed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.agent_core.facts.postconditions import (
    GradedCourse,
    Standing,
    Violation,
    check_gpa_in_range,
    check_grades_in_range,
    check_joint_floor,
    check_term_load,
)

# A number that will not swallow a trailing '.' from prose like "GPA is 83.888."
_NUM = r"\d+(?:\.\d+)?"
# A rendered term header, "Winter -- 19.5 credits", used to count the plan's terms.
_TERM_HEADER = re.compile(rf"--\s*{_NUM}\s*credits", re.IGNORECASE)
_FLOOR = re.compile(rf"above\s+({_NUM})")
# One rendered plan line: "... number 00940314 ... credits 3.5 ... min_grade 10.57".
_COURSE = re.compile(rf"number\s+(\d+).*?credits\s+({_NUM})\D+?min_grade\s+(-?{_NUM})")
# Facts print as `name=value`, so require `=`/`:` -- NOT a bare space, which would
# also match `credits 3.5` in a rendered plan line and grab a course's credits as
# the standing. `\b` so bare `credits=` matches but `winter_credits=` does not (no
# word boundary inside `winter_credits`). Several spellings; gpa cross-checks them.
_SEP = r"\s*[=:]\s*"
_TOTAL_CREDITS = re.compile(rf"\b(?:total_credits|completed_credits|earned_credits|credits){_SEP}({_NUM})")
_POINTS = re.compile(rf"\b(?:total_points|completed_points|quality_points|earned_points|points){_SEP}({_NUM})")
_GPA = re.compile(rf"(?:\bgpa{_SEP}|GPA is\s+)({_NUM})")
_GPA_TOLERANCE = 0.5


@dataclass(frozen=True)
class ScoredRun:
    """The verdict for one saved run: what failed, and what could not be judged."""

    violations: list[Violation] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Green only when every applicable check ran AND none was violated. A
        skipped check is 'not verified', which is not the same as 'passed'."""
        return not self.violations and not self.skipped

    def summary(self) -> str:
        if self.passed:
            return "PASS (all checks ran clean)"
        parts = [f"{v.kind}: {v.message}" for v in self.violations]
        parts += [f"SKIPPED -- {reason}" for reason in self.skipped]
        head = "FAIL" if self.violations else "UNSCORED"
        return f"{head} ({len(self.violations)} violation(s), {len(self.skipped)} skipped)\n  " + "\n  ".join(parts)


def extract_floor(question: str) -> float | None:
    match = _FLOOR.search(question)
    return float(match.group(1)) if match else None


def extract_courses(answer: str) -> list[GradedCourse]:
    """One GradedCourse per rendered plan line. Empty when the answer is not a
    per-course plan (e.g. a refusal), which the caller reads as 'nothing to score'."""
    return [
        GradedCourse(code=code, credits=float(credits), min_grade=float(min_grade))
        for line in answer.splitlines()
        for match in [_COURSE.search(line)]
        if match
        for code, credits, min_grade in [match.groups()]
    ]


def extract_standing(record: Mapping[str, object]) -> Standing | None:
    """Recover total_points/total_credits from the answer and transcript details.

    Searches every detail string, not just the answer, because the standing is a
    derived scalar the model prints mid-run, not something it always restates in
    the final prose. Returns None when either half is missing -- the joint-floor
    check is then skipped, not guessed.
    """
    haystacks = [str(record.get("answer", ""))]
    transcript = record.get("transcript")
    if isinstance(transcript, Sequence):
        haystacks += [str(turn.get("detail", "")) for turn in transcript if isinstance(turn, Mapping)]

    points = credits = None
    for text in haystacks:
        if points is None and (m := _POINTS.search(text)):
            points = float(m.group(1))
        if credits is None and (m := _TOTAL_CREDITS.search(text)):
            credits = float(m.group(1))

    # Cross-fill from gpa = points / credits, so holding gpa and either total is
    # enough; then use gpa to reject a mis-named fact (a confident wrong verdict
    # is worse than an honest skip). Mirrors answer_verify._standing.
    gpa = extract_reported_gpa(record)
    if credits is None and points is not None and gpa:
        credits = points / gpa
    if points is None and credits is not None and gpa is not None:
        points = gpa * credits
    if points is None or credits is None or credits <= 0:
        return None
    if gpa is not None and abs(points / credits - gpa) > _GPA_TOLERANCE:
        return None
    return Standing(total_points=points, total_credits=credits)


def extract_reported_gpa(record: Mapping[str, object]) -> float | None:
    haystacks = [str(record.get("answer", ""))]
    transcript = record.get("transcript")
    if isinstance(transcript, Sequence):
        haystacks += [str(turn.get("detail", "")) for turn in transcript if isinstance(turn, Mapping)]
    for text in haystacks:
        if m := _GPA.search(text):
            return float(m.group(1))
    return None


def score_run(record: Mapping[str, object]) -> ScoredRun:
    """Judge one saved run. Runs each check for which the inputs are recoverable,
    and records a reason for each one that is not."""
    violations: list[Violation] = []
    skipped: list[str] = []

    courses = extract_courses(str(record.get("answer", "")))
    if not courses:
        return ScoredRun(skipped=["no per-course plan lines found in the answer -- nothing to score"])

    violations += check_grades_in_range(courses)

    # Term load: a single rendered term far over a full semester is the overflow
    # bug. For one term the whole listing IS that term; for a multi-term plan the
    # prose can't be split cleanly here, so defer to the loop verifier (which reads
    # each term's typed facts) rather than sum across terms.
    if len(_TERM_HEADER.findall(str(record.get("answer", "")))) <= 1:
        violations += check_term_load(sum(c.credits for c in courses), "the plan")
    else:
        skipped.append("term-load: multi-term prose not split here; the loop verifier checks per term")

    gpa = extract_reported_gpa(record)
    if gpa is None:
        skipped.append("gpa-range: no GPA value found in the answer or transcript")
    else:
        violations += check_gpa_in_range(gpa)

    floor = extract_floor(str(record.get("question", "")))
    standing = extract_standing(record)
    if floor is None:
        skipped.append("joint-floor: no GPA floor ('above N') found in the question")
    elif standing is None:
        skipped.append("joint-floor: could not recover total_points/total_credits from the run")
    else:
        violations += check_joint_floor(standing, courses, floor)

    return ScoredRun(violations=violations, skipped=skipped)


__all__ = ["ScoredRun", "extract_courses", "extract_floor", "extract_standing", "score_run"]
