"""The answer boundary -- phase 9d of docs/agent/tools_implementation_plan.md.

The last check before a person reads anything: every number in the answer must
come from a fact, and the answer's confidence must be the weakest thing it
stands on.

**Why this is simpler than the version it replaces.** The old boundary sorted
answer content into "slot-filled token" or "free prose", and interpretation --
a cited paragraph -- was neither. It was a value the model had to state
verbatim, so it could not be a slot, and it carried a claim, so it could not be
free prose. That gap produced a rejection with no legal move, and the loop
burned its remaining turns discovering there wasn't one.

Typed facts remove the third category rather than accommodating it. `interpret`
returns a typed SCALAR plus a separate CITATION, so the value is slottable like
any other and the prose it came from travels alongside the answer instead of
inside it.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Union

from app.agent_core.facts.prose import Citation
from app.agent_core.facts.types import Basis, Collection, Scalar, weakest

_SLOT = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(?::([a-z]+))?\}")
_NUMERAL = re.compile(r"\d")
_BROKEN_SLOT = re.compile(r"\{[a-zA-Z_][^}]*[.:][^}]*\}")
"""A `{...}` that looks like an intended slot but is not one -- it carries a `.`
or a second `:`, as in `{fact.field}`. Checked AFTER valid slots are
substituted, so anything it matches is genuinely malformed, not a false hit on a
real slot."""
_COURSE_CODE = re.compile(r"\b\d{7,8}\b")
"""A course code as it renders in an answer: a 7- or 8-digit run. Distinct from
credits (a decimal or a 1-2 digit count) and a grade (a decimal), so matching it
in a finished plan finds course identities without catching the numbers beside
them. Used only to catch the same course placed twice -- see the `:detail` gate
in `resolve_answer`."""
_OBJECT_ID = re.compile(r"\b[0-9a-f]{24}\b")
"""A Mongo ObjectId: exactly 24 hex characters. Nothing else in this domain is
that shape -- a course code is 8 digits, a grade two -- so matching it in a
finished answer catches an internal key leaking to the reader without any risk
of a false positive on real content."""
_NUMERIC_TOKEN = re.compile(r"\d(?:[\d.,]*\d)?")
"""A whole number, not a single digit, and never ending on punctuation.

Two traps this shape avoids, both found in live runs:
  - Matching ONE digit would gut the echo check below: "0" is in almost every
    question with a course code, so every digit would be waved through.
  - A trailing `[.,]` would capture "00960211," (with the comma) in "...course
    00960211, and...", which is NOT equal to the "00960211" in the question --
    so a course code the user typed, echoed back with a comma after it, was
    refused as an ungrounded number. Requiring the token to END in a digit
    keeps decimals ("92.5") and thousands ("1,000") whole while dropping
    trailing punctuation."""


@dataclass(frozen=True)
class HeldFact:
    """A fact in the working set: its value, how it is known, where it was read
    (for prose), and HOW IT WAS DERIVED.

    `derivation` exists because of a live failure that no check could catch. The
    model named a fact `remaining_credits`, filled it with the degree's TOTAL,
    and answered "you still need 155 credits". Every gate passed and should
    have: the number came from a real fact, official, non-empty, not typed.

    The invariant guarantees a number came from a fact. It cannot guarantee the
    fact means what the sentence says it means, because the NAME is prose the
    model wrote. So the derivation travels with the answer and a reader sees
    "155 (degree_programs.totalCredits)" -- which makes the mistake obvious
    without anything having to understand it.
    """

    value: Union[Collection, Scalar]
    basis: Basis
    citation: Citation | None = None
    derivation: str | None = None


@dataclass(frozen=True)
class Answer:
    text: str
    basis: Basis
    used: tuple[str, ...]
    citations: tuple[Citation, ...]
    derivations: tuple[tuple[str, str], ...] = ()
    """(fact name, how it was derived) for every slot the answer used."""

    @property
    def speculative(self) -> bool:
        return self.basis is Basis.SIMULATED


@dataclass(frozen=True)
class Ungrounded:
    """The answer was refused. `reason` says what to change."""

    reason: str


def resolve_answer(
    template: str, facts: Mapping[str, HeldFact], question: str = ""
) -> Union[Answer, Ungrounded]:
    """Fill a templated answer from held facts, or refuse it.

    The template carries prose and `{fact_name}` slots. Values are substituted
    from facts in code -- the model never types a number, which is what makes
    the grounding structural rather than a request the model may decline.

    `question` exists for one reason: course codes are numerals. Without it the
    rule "no typed digits" also forbade the model from NAMING the course the
    user had just asked about -- a live run wrote "course 00960211 is not
    offered in summer", had it refused as an ungrounded number, and never
    recovered. A token echoed verbatim from the question is not a laundered
    computation; it is the user's own reference, and refusing it makes the agent
    unable to say what it is talking about.
    """
    used: list[str] = []
    unknown: list[str] = []
    detail_renders: list[str] = []
    detailed: list[tuple[str, Any]] = []

    def substitute(match: re.Match[str]) -> str:
        name, modifier = match.group(1), match.group(2)
        held = facts.get(name)
        if held is None:
            unknown.append(name)
            return match.group(0)
        used.append(name)
        rendered = _render(held.value, modifier)
        if modifier == "detail":
            # Kept per SLOT, not merged, so the duplicate rule can tell a course
            # listed in two semesters from a course named twice inside one list.
            detail_renders.append(rendered)
            detailed.append((name, held.value))
        return rendered

    filled = _SLOT.sub(substitute, template)

    if unknown:
        return Ungrounded(
            f"the answer refers to {', '.join(repr(u) for u in unknown)}, which "
            f"{'is' if len(unknown) == 1 else 'are'} not a held fact."
            f"{_did_you_mean(unknown, facts)} "
            f"Available: {sorted(facts)}."
        )

    # A `{...}` that survived substitution looked like a slot and was not one --
    # `{above_90_joined.course_code}`, a dotted field projection the grammar does
    # not allow. It matched no slot, so it was neither rendered nor flagged as
    # unknown, and the raw braces SHIPPED in an accepted answer. Catch it: a slot
    # is `{fact}` or `{fact:count}`, never `{fact.field}`.
    leftover = _BROKEN_SLOT.search(filled)
    if leftover:
        return Ungrounded(
            f"'{leftover.group(0)}' is not a valid slot. A slot is {{fact_name}} or "
            "{fact_name:count} -- never {fact.field}. To show one field of a collection, the bare "
            "{fact_name} already lists its readable field; to count it, {fact_name:count}."
        )

    # Any digit in the LITERAL prose -- the parts that were not substituted --
    # is a number the model typed rather than derived. This is the whole
    # invariant, and it is checked against the template rather than the filled
    # text so that legitimate substituted values are never mistaken for it.
    literal = _SLOT.sub("", template)
    stray = next(
        (m for m in _NUMERIC_TOKEN.finditer(literal) if m.group(0) not in question),
        None,
    )
    if stray:
        context = literal[max(0, stray.start() - 30):stray.start() + 30].strip()
        return Ungrounded(
            f"the answer states a number that came from no fact: ...{context}... "
            "Every figure must be a {slot} filled from a fact, so that what the reader sees "
            "is what was computed."
        )

    # A held fact's NAME written as prose is a slot the model forgot to brace.
    # Live: "I've prepared a request to register target_course_number." -- the
    # student is shown a variable where a course code belongs, and every other
    # check passes because the name is not a number and the real slots resolved.
    # Only names carrying an underscore are considered: a fact called `credits`
    # legitimately appears in "you have 129.5 credits", where `target_course_number`
    # is not something anyone writes by accident.
    leaked = sorted(
        name for name in facts
        if "_" in name and re.search(rf"(?<![{{\w]){re.escape(name)}(?!\w)", filled)
    )
    if leaked:
        return Ungrounded(
            f"the answer writes the fact NAME {', '.join(repr(n) for n in leaked)} as prose. "
            "A reader sees the variable, not its value -- put it in braces to slot it, "
            "or say the value in your own words."
        )

    if not used:
        return Ungrounded(
            "the answer stands on no facts at all. Even a qualitative answer must cite what it "
            "read, or there is nothing distinguishing it from a guess."
        )

    # An answer whose every slot renders empty is not grounded in anything.
    # Observed on the first live run: the model wrote "I can't determine ...
    # (none), (none), (none) are all empty" and it PASSED, because slots were
    # present and no digit was typed. Citing empty facts is not citing facts,
    # and shipping it would let a non-answer wear the shape of a verified one.
    # Kept as it was, after a live run tempted a change. "No summer offerings"
    # cites an empty COMPLETE collection, and allowing that would also re-admit
    # the non-answer this rule exists for -- "I can't determine: (none), (none)"
    # cites empty complete collections too. Completeness does not separate the
    # two, so the boundary stays strict and the PROMPT tells the model how to
    # phrase a negative: cite the collection you searched, not the empty result
    # of searching it. A mix of empty and populated facts is already allowed.
    empty = [name for name in used if _is_empty(facts[name].value)]
    if len(empty) == len(set(used)):
        return Ungrounded(
            f"every fact the answer cites is empty ({', '.join(sorted(set(empty)))}). "
            "Fetch the data before writing the answer -- an answer built on empty collections "
            "says nothing, however correctly it is phrased."
        )

    # A raw ObjectId in the FINISHED answer -- checked on `filled`, not the
    # template, because these arrive THROUGH a slot: the model held a transcript
    # keyed by `courseId` and slotted it, rendering two dozen 24-hex ids into
    # prose meant for a person. An ObjectId is an internal join key; it is never
    # the right thing to show a user, and a course it identifies always has a
    # readable number reachable by joining to `courses`. So this is a rejection
    # with a legal move, not a dead end.
    object_id = _OBJECT_ID.search(filled)
    if object_id:
        return Ungrounded(
            f"the answer shows a raw internal id ({object_id.group(0)}). That is a database key, "
            "meaningless to the reader. Resolve it to the course NUMBER -- join the fact holding "
            "the id to `courses` on `_id` and slot `courseNumber` -- and cite that instead."
        )

    # A plan (rendered with `:detail`) that names the same course twice was not
    # placed -- it was faked. The signature is unmistakable and it came up in
    # every live planning run: rather than call `optimize`, the model built the
    # semesters by selecting `course_offerings` on `semesterName`, and a course
    # offered in BOTH winter and spring then appears in both lists (its per-
    # semester credit totals balloon past any real load). A real placement
    # assigns each course to exactly one slot, so a repeat is proof `optimize`
    # was skipped. Refuse with the fix, so the loop pushes the model back to it.
    # A source row slotted straight into prose. `:detail` renders EVERY field a
    # record carries, which is right for a projected table and wrong for a row
    # that came out of `find` untouched -- a live answer listed 16 remaining
    # courses as `courseNumber … · title … · titleHebrew … · credits … ·
    # faculty … · studyFramework … · catalogYear 2025 · status published`,
    # showing a student the catalog's bookkeeping and the title twice.
    #
    # Checked on WIDTH rather than on field names, because this boundary stays
    # domain-blind: it cannot know that `status` is uninteresting, but it can
    # know that a table a person is meant to read has a handful of columns and
    # that nine means nobody chose them. The fix is one `project` away, so the
    # refusal names it.
    for name, value in detailed:
        widest = max((len(record.fields) for record in getattr(value, "records", ())), default=0)
        if widest > _MAX_DETAIL_FIELDS:
            fields = sorted({f for r in value.records for f in r.fields})
            return Ungrounded(
                f"'{name}' is rendered with :detail but its records carry {widest} fields "
                f"({', '.join(fields)}) -- that is a source row, not an answer. `project` the "
                "few a reader needs (a course number, a title, a credit count) and slot THAT. "
                "Rendering every column shows internal bookkeeping and repeats itself."
            )

    # ACROSS detail slots, not within one. The signature this rule exists for is
    # a course appearing in TWO SEMESTERS, and the semesters are two separately
    # rendered collections -- that is what the faked split produces, because the
    # model selected `course_offerings` twice by `semesterName`.
    #
    # Counting repeats in the merged text instead was a false positive with
    # teeth: "am I eligible for 00960211" renders that course's prerequisite
    # EDGES, and every edge spells the course code (`00960211->00940224`,
    # `00960211->00940226`). A correct eligibility answer therefore looked
    # exactly like a faked plan and was refused -- with advice to call `optimize`,
    # on a question containing no plan at all. Gating on placement FIELDS would
    # not do either: a faked plan has no `slot`, which is precisely what makes it
    # fake, so that gate would switch the rule off exactly when it is needed.
    if len(detail_renders) > 1:
        appearances: Counter[str] = Counter()
        for rendered in detail_renders:
            # `set` per rendering: a course named twice inside ONE list is a list
            # that mentions it twice, not a course placed twice.
            appearances.update(set(_COURSE_CODE.findall(rendered)))
        repeated = [code for code, n in appearances.items() if n > 1]
        if repeated:
            return Ungrounded(
                f"the plan lists course {repeated[0]} in more than one semester. A real placement "
                "assigns each course to exactly ONE slot -- this happens when the semesters are built "
                "by selecting `course_offerings` on `semesterName` (a course offered in winter AND "
                "spring then lands in both) instead of by `optimize`. Call `optimize` with the "
                "remaining courses as items and the two terms as slots, and read ITS placed rows: "
                "each course appears once, under one `slot`."
            )

    consumed = [facts[name] for name in used]
    return Answer(
        text=_tidy_affirmations(filled),
        basis=weakest([held.basis for held in consumed]),
        used=tuple(dict.fromkeys(used)),
        citations=tuple(held.citation for held in consumed if held.citation is not None),
        derivations=tuple(
            (name, facts[name].derivation)
            for name in dict.fromkeys(used)
            if facts[name].derivation
        ),
    )


def _did_you_mean(unknown: tuple, facts: Mapping[str, object]) -> str:
    """Name the held fact an unknown slot was probably reaching for.

    A live run derived `english_requirement`, then wrote
    `{english_requirement_phrase}` in the answer and was refused. The refusal
    already listed every held fact, and the model still did not recover -- a
    list of twelve names does not point at the one that is a character-swap
    away, and the run ended with no answer for what was a typo.

    The cutoff is deliberately high. This only ever SUGGESTS -- nothing is
    renamed or auto-corrected, because a slot quietly resolved to a fact the
    model did not mean is the grounding failure this whole layer exists to
    prevent. A near-miss below the cutoff simply gets the list, as before.
    """
    from difflib import get_close_matches

    pairs = [
        (name, close[0])
        for name in unknown
        for close in [get_close_matches(name, list(facts), n=1, cutoff=0.75)]
        if close
    ]
    if not pairs:
        return ""
    if len(pairs) == 1:
        return f" Did you mean {pairs[0][1]!r}?"
    return " Did you mean " + ", ".join(f"{bad!r} -> {good!r}" for bad, good in pairs) + "?"


_DOUBLED_YES = re.compile(r"\b(yes|no)\b([\s,;:.—–-]+)\b\1\b", re.IGNORECASE)
_STRANDED_YES = re.compile(r"\b(are|is|am|was|were)\s+(yes|no)\s+(?=[a-z])", re.IGNORECASE)


def _tidy_affirmations(text: str) -> str:
    """Repair the two ways a true/false slot lands badly in a sentence.

    A BOOL fact renders as the bare word "yes" or "no", which reads correctly on
    its own ("Eligible: yes.") and badly when the model also writes the word --
    "Yes -- {eligible}." becomes "Yes -- yes.", and "You are {eligible} eligible"
    becomes "You are yes eligible". Measured on the eligibility question, roughly
    a third of otherwise-correct answers came out one of those two ways.

    The prompt asks for neither, and mostly gets its way; this catches the rest.
    It is deliberately narrow -- an adjacent duplicate, or a yes/no stranded
    between a copula and an adjective -- because it edits an answer that already
    passed grounding, and the one thing it must never do is change what the
    answer CLAIMS. Both rewrites drop a redundant word and touch nothing else.
    """
    tidied = _DOUBLED_YES.sub(lambda m: m.group(1), text)
    return _STRANDED_YES.sub(lambda m: f"{m.group(1)} ", tidied)


def _is_empty(value: Union[Collection, Scalar]) -> bool:
    return isinstance(value, Collection) and not value.records


def _render(value: Union[Collection, Scalar], modifier: str | None) -> str:
    if isinstance(value, Scalar):
        return _render_scalar(value)

    if modifier == "count":
        return str(len(value.records))
    if modifier == "detail":
        return _render_detail(value)
    if modifier == "list" or modifier is None:
        rendered = [_readable_field(record) for record in value.records]
        rendered = [text for text in rendered if text]
        if not rendered:
            return "(none)"
        # A slot holding a large collection dumped every record into the prose --
        # a live partial answer listed 117 prerequisite edges inline, which is
        # noise, not an answer. Cap the inline list and say how many more there
        # are; a caller who wants the number uses `{name:count}`.
        if len(rendered) > _LIST_CAP:
            shown = ", ".join(rendered[:_LIST_CAP])
            return f"{shown}, and {len(rendered) - _LIST_CAP} more"
        return ", ".join(rendered)
    return str(len(value.records))


def _render_detail(value: Collection) -> str:
    """One record per line, each showing ALL its readable fields as `name value`.

    The bare `{fact}` list shows one field per record -- enough to say "these
    courses", not enough for a plan the reader must act on, where each row needs
    its number AND title AND credits AND the grade just computed for it. A plan
    with only course numbers was the exact gap that made a two-semester schedule
    unreadable however correctly it was derived.

    This stays domain-blind on purpose: it renders whatever fields the record
    carries, in order, under the names the caller `project`ed them to. So the
    labels a reader sees ("min_grade 87") are the caller's own field names, not
    anything this module knows about courses -- the same separation that keeps
    the rest of the boundary general. `_id` and any ObjectId-shaped value are
    dropped, because an internal key is never the thing to show a person (and the
    finished-answer ObjectId guard would reject it anyway).
    """
    lines = [line for line in (_detail_line(record) for record in value.records) if line]
    if not lines:
        return "(none)"
    if len(lines) > _DETAIL_CAP:
        shown = "\n".join(lines[:_DETAIL_CAP])
        return f"{shown}\n...and {len(lines) - _DETAIL_CAP} more"
    return "\n".join(lines)


def _detail_line(record: Collection) -> str:
    parts = [
        f"{name} {_render_scalar(v)}"
        for name, v in record.fields.items()
        if isinstance(v, Scalar) and name != "_id" and not _OBJECT_ID.fullmatch(str(v.value))
    ]
    return "- " + " · ".join(parts) if parts else ""


_MAX_DETAIL_FIELDS = 5
"""How many fields a `:detail` record may carry before it reads as a data dump.

Calibrated against both ends: every legitimate rendering in the suite projects 1
or 2 fields, and a realistic reader-facing row (number, title, credits, grade)
is 4 -- while the narrowest source in the registry that caused trouble is 7 and
`courses` is 9. Five leaves room for a genuinely wide answer table and still
catches a row nobody narrowed.
"""

_DETAIL_CAP = 60
"""How many rows `:detail` prints before it summarises the rest. Generous: a
full two-semester plan plus its unscheduled overflow is well under this, and a
plan that truncated its own courses would be worse than one that ran long."""


_LIST_CAP = 15
"""How many records a bare collection slot lists inline before it summarises the
rest. Enough to show a real plan's courses; short of dumping a whole catalog."""


def _readable_field(record: Collection) -> str:
    """The first field of a record worth showing a person.

    Skips `_id` and any ObjectId-shaped value, because the old render took the
    FIRST field regardless -- and for an offerings or courses record that is the
    ObjectId `_id`, so `{offerings}` dumped two dozen internal keys into prose
    and the ObjectId guard then refused the whole answer. A readable field
    exists on every domain record (a course number, a name); prefer it.
    """
    scalars = [(name, v) for name, v in record.fields.items() if isinstance(v, Scalar)]
    readable = [
        v for name, v in scalars
        if name != "_id" and not _OBJECT_ID.fullmatch(str(v.value))
    ]
    chosen = readable[0] if readable else (scalars[0][1] if scalars else None)
    return _render_scalar(chosen) if chosen is not None else ""


_DECIMALS = 2
"""How many decimals a derived number shows a reader.

Two, because every quantity in this domain is a grade, a GPA or a credit count,
and none of them are meaningful past hundredths. The value is NOT rounded in the
fact -- only in the rendering -- so arithmetic downstream still uses the full
precision it was computed with.
"""


def _render_scalar(value: Scalar) -> str:
    if isinstance(value.value, bool):
        return "yes" if value.value else "no"
    if isinstance(value.value, float):
        if value.value.is_integer():
            # 16.0 credits reads as a rounding artefact; 16 reads as an answer.
            return str(int(value.value))
        # A live answer said "Your GPA is 72.64074074074074". Every digit of that
        # is real -- it is a mean over 44 courses -- and printing all of them
        # still makes a correct number look like a bug, and buries the two
        # figures a student actually compares against a threshold.
        trimmed = f"{value.value:.{_DECIMALS}f}".rstrip("0").rstrip(".")
        return trimmed or "0"
    return str(value.value)


__all__ = ["Answer", "HeldFact", "Ungrounded", "resolve_answer"]
