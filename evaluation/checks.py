"""The vocabulary for judging an answer, in one tested place.

Written after five wrong verdicts in a single session, every one of them a
hand-rolled regex in a throwaway script, and every one of them scoring a CORRECT
answer as a failure:

  - "requires one of 00940224, 00940226."   -> "missing 00940226" (trailing full stop)
  - "you still need 25.5."                  -> "missing 25.5"     (trailing full stop)
  - "Your current GPA is 72.64."            -> "missing 72.64"    (trailing full stop)
  - "I couldn't find a recorded grade"      -> "did not deny knowledge"
  - "could not confirm that course"         -> "did not deny knowledge"

A scorer that is wrong in the pessimistic direction is not the safe kind of
wrong. It hides real regressions in a crowd of false ones, and it costs a live
run every time it lies. So the predicates live here, they are tested, and the
probe scripts import them instead of inventing another regex.

The prose predicates are deliberately generous. English has many ways to deny
knowledge and only a few to assert a number, so `denies_knowledge` casts wide
while `states_number` stays exact.
"""

from __future__ import annotations

import re

# An edge id -- `00960211->00940224` -- contains a real course code. Matching the
# code inside it credits an answer that named an internal key rather than a
# course, which is how a debugging dump once scored as correct.
_EDGE_ID = re.compile(r"\b\d{6,8}\s*->\s*\d{6,8}\b")

_DENIALS = (
    r"could ?n[o']?t (find|confirm|determine|locate|verify)",
    r"can ?n[o']?t (find|confirm|determine|assess|tell|say)",
    r"\bunable to\b",
    r"\bno (such|record|grade|result|entry|match|data)\b",
    r"\bnot (in|on|found|listed|recorded|present)\b",
    r"\bdoes ?n[o']?t exist\b",
    r"\bhave ?n[o']?t taken\b",
    r"\bwas ?n[o']?t able\b",
    r"\bis ?n[o']?t in\b",
    r"\b0 (rows|records|results|matches|attempts)\b",
    r"\bno .{0,20}(found|matched)\b",
)
_DENIES = re.compile("|".join(_DENIALS), re.IGNORECASE)

_AFFIRMS = re.compile(
    # `you meet` used to be bare, which made "you meet 0 of 1 prerequisite
    # groups" -- a REFUSAL -- read as an affirmation. Harmless until the
    # contradiction check started asking whether both fired at once, and then it
    # scored every correct denial as self-contradictory. The count is what makes
    # it an affirmation, so the count is in the pattern.
    # A verdict "yes" opens the answer or follows punctuation -- "Yes.",
    # "Eligible: yes", "offered next spring: yes", "yes -- you meet 1 of 1".
    # A BARE \byes\b also matched the hypothetical the prompt now explicitly
    # asks for: "No -- you meet 0 of 1. To make it yes, pass 01040066." That is
    # a correct DENIAL, and it scored as affirming and denying at once.
    r"(?:^|[:.\-—–]\s*)yes\b"
    r"|\byou are eligible\b|\byou can take\b"
    r"|\b(?:you\s+)?(?:have\s+)?(?:meet|meets|met)\s+[1-9]\d*\s+of\b"
    # A projection affirms in the passive too: "next spring is forecast to
    # offer it" states the same thing as "yes, it will run".
    r"|\bforecast(?:ed)?\s+to\b|\bis\s+scheduled\s+to\b"
    # A projection affirms without ever saying "yes": "it has run every spring,
    # so it is expected again". Scoring that as a non-answer is the pessimistic
    # failure this file exists to prevent.
    r"|\b(will|should) (be )?(offered|run|available)\b"
    r"|\bis (expected|likely|on track)\b|\bexpect(ed)? to\b|\blikely to\b"
    r"|\bevery (spring|winter|summer|semester|year)\b",
    re.IGNORECASE,
)

_NEGATIVE_CLAIMS = (
    # Not a denial of KNOWLEDGE -- a confident claim that the thing is false.
    # "I could not determine whether it runs" is honest; "it will not run" is
    # the inverted forecast, and only the second one is a wrong answer.
    r"\bwill not\b",
    r"\bwo ?n[o']?t\b",
    r"\bnot be (offered|available|running)\b",
    r"\bis ?n[o']?t (offered|available|expected)\b",
    r"\bunlikely\b",
    r"\bnot likely\b",
    r"\bnot expected\b",
    r"\bnot eligible\b",
    r"\bdoes ?n[o']?t meet\b",
    r"\bdo ?n[o']?t meet\b",
    r"\bcannot (take|register)\b",
    r"\bcan ?n[o']?t (take|register)\b",
    r"^\s*no[,.\s—-]",
    # A leading "No." was not enough. Live answer, scored FAIL while correct:
    #   "For 01040174, no -- you meet 0 of 1 prerequisite groups."
    # The denial is real and mid-sentence, because the model led with the course.
    r"[,;:]\s*no\b\s*[—–-]",
    r"\bso,?\s*no\b",
    # Domain-specific and unambiguous: this phrasing IS the refusal, and the
    # count is what makes it one. It survives every rewording of the prose
    # around it, which "no" does not.
    r"\bmeet(s|ing)? 0 of\b",
    r"\b0 of \d+ prerequisite groups?\b",
    # A target that cannot be hit is a denial, and none of the above saw it.
    # The GPA question scored 0/3 "never states the negative" on three answers
    # that all said so: "not reachable with this load", "which is not
    # achievable", "above the maximum possible grade".
    r"\bnot\s+(reachable|achievable|attainable|possible|feasible|viable)\b",
    r"\bcan ?n[o']?t be (met|reached|achieved|hit|done)\b",
    r"\bcannot be (met|reached|achieved|hit|done)\b",
    r"\babove the maximum\b|\bexceeds the maximum\b",
    r"\bimpossible\b",
)
_CLAIMS_NO = re.compile("|".join(_NEGATIVE_CLAIMS), re.IGNORECASE | re.MULTILINE)


def states_number(text: str, number: str | float) -> bool:
    """Whether the answer really states this number.

    Bounded so that 155 does not satisfy a check for 15 and 129.5 does not
    satisfy 29.5 -- while a number ENDING A SENTENCE still counts, which a
    blanket `(?![\\d.])` gets wrong and which was the single most common way
    these checks lied.
    """
    needle = f"{number:g}" if isinstance(number, float) else str(number)
    body = _EDGE_ID.sub(" ", text or "")
    return re.search(rf"(?<![\d.]){re.escape(needle)}(?!\d)(?!\.\d)", body) is not None


def mentions_code(text: str, code: str) -> bool:
    """Whether a course CODE is named as a course, not buried in an edge id."""
    return states_number(text, code)


def denies_knowledge(text: str) -> bool:
    """Whether the answer says it could not establish something.

    Generous by design: this is the shape of a CORRECT answer to an unanswerable
    question, so a narrow pattern here turns good behaviour into a red mark.
    """
    return bool(_DENIES.search(text or ""))


def claims_yes(text: str) -> bool:
    """Whether the answer affirms -- eligible, allowed, will run."""
    return bool(_AFFIRMS.search(text or ""))


def claims_no(text: str) -> bool:
    """Whether the answer asserts the negative -- will NOT run, NOT eligible.

    Distinct from `denies_knowledge` on purpose. "I could not determine whether
    it runs next spring" is a correct answer to an unanswerable question; "it
    will not run next spring" is a claim, and when the data says it has run
    every spring on record, it is a WRONG claim. Only the second is scored as a
    failure, so the two must not share a predicate.
    """
    return bool(_CLAIMS_NO.search(text or ""))


def scores(
    text: str | None,
    *,
    must: tuple = (),
    must_not: tuple = (),
    stance: str | None = None,
) -> tuple[str, str]:
    """(verdict, why), where verdict is "correct", "incomplete" or "wrong".

    Three states, not two, because two conflated the only distinction that
    matters. `eligibility_01040174` answered "yes, you meet 1 of 1 prerequisite
    groups" for a student who meets none -- a student told to register for
    something they cannot take. Fixed, it answered "No. You meet 0 of 1", which
    is right but does not say WHICH prerequisite is missing, so it still failed
    `must_contain` and the score stayed 0/3. A measurement that cannot see a
    dangerous answer become a correct one is not measuring.

    So the STANCE is correctness and is checked first: getting it wrong is
    "wrong". Missing numbers or codes with the stance right is "incomplete" --
    a real failure, counted separately, never confused with the other.

    `stance` is "affirm" or "deny". Some questions have no distinguishing
    number: a forecast's failure mode is INVERSION -- "00940412 will not be
    offered next spring" when it has run every spring on record -- and the wrong
    answer names exactly the same course code as the right one. `must_contain`
    cannot separate those two; only the stance can.
    """
    if not text:
        return "wrong", "no answer at all"
    for value in must_not:
        if states_number(text, value):
            return "wrong", f"states {value}, which is the known-wrong value"
    if stance == "deny" and claims_no(text) and claims_yes(text):
        # Both halves present under a stance is not a pass, it is a
        # contradiction. Measured: "You are eligible for 01040174, because you
        # meet 0 of 1 prerequisite groups" scored CORRECT, because `claims_no`
        # matched "meet 0 of" and nothing looked further. An answer a reader can
        # take either way is worse than a plainly wrong one.
        return "wrong", "affirms and denies in the same answer"
    if stance == "affirm":
        # Order matters: "will not be offered" contains "be offered", so the
        # negative claim has to be tested first or an inversion scores as a pass.
        if claims_no(text):
            return "wrong", "answers no where the data says yes"
        if not claims_yes(text):
            return "wrong", "never affirms, and the data says yes"
    elif stance == "deny":
        if not claims_no(text):
            return "wrong", "never states the negative, and the data says no"
    # A give-up is not a thin answer, and it is not a correct one either. Every
    # question in the ground truth is answerable from the data by construction,
    # so declining one is a failure whatever else the text does or does not say.
    #
    # Checked BEFORE the numbers, not inside the `missing` branch where it
    # started: a question with an empty `must_contain` -- which
    # `semesters_to_graduate` needs, since 2 is a floor and 3 is also right --
    # never reached it, and "I wasn't able to work that out" scored CORRECT.
    if denies_knowledge(text):
        return "wrong", "declined to answer a question the data supports"

    missing = [v for v in must if not states_number(text, v)]
    if missing:
        return "incomplete", f"right answer, but never states {', '.join(str(m) for m in missing)}"
    return "correct", "matches ground truth"


__all__ = [
    "claims_no",
    "claims_yes",
    "denies_knowledge",
    "mentions_code",
    "scores",
    "states_number",
]
