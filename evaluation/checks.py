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

_AFFIRMS = re.compile(r"\byes\b|\byou are eligible\b|\byou meet\b|\byou can take\b", re.IGNORECASE)


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


def scores(text: str | None, *, must: tuple = (), must_not: tuple = ()) -> tuple[bool, str]:
    """(passed, why) for numbers/codes an answer must and must not state."""
    if not text:
        return False, "no answer at all"
    for value in must_not:
        if states_number(text, value):
            return False, f"states {value}, which is the known-wrong value"
    missing = [v for v in must if not states_number(text, v)]
    if missing:
        return False, f"never states {', '.join(str(m) for m in missing)}"
    return True, "matches ground truth"


__all__ = [
    "claims_yes",
    "denies_knowledge",
    "mentions_code",
    "scores",
    "states_number",
]
