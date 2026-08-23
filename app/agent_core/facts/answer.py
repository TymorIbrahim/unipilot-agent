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
_HEBREW_CHAR = re.compile(r"[֐-׿]")
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
    # The reader's language, decided once. Enum VALUES the planner stores
    # ("elective", "check_prerequisites") are printed to the student, and in a
    # Hebrew answer they arrived as English inside Hebrew prose -- see
    # `_ENUM_WORDS`. Read off the QUESTION rather than the model's prose: the
    # question is what the student wrote, and a template can be almost all
    # slots with too little prose to judge.
    hebrew = bool(_HEBREW_CHAR.search(question or ""))

    used: list[str] = []
    counted: list[str] = []
    """Facts cited as `{name:count}` rather than listed.

    Tracked because HOW a fact is cited is what separates a real negative answer
    from a non-answer, where completeness could not -- see the empty-facts rule
    below."""
    unknown: list[str] = []
    detail_renders: list[str] = []
    detailed: list[tuple[str, Any]] = []
    detail_slots: list[str] = []
    """The `{name:detail}` slot text itself, so where it LANDED can be checked.

    A `:detail` render is a table -- `label value · label value · ...` -- and a
    table inside a clause is not a sentence."""

    def substitute(match: re.Match[str]) -> str:
        name, modifier = match.group(1), match.group(2)
        held = facts.get(name)
        if held is None:
            unknown.append(name)
            return match.group(0)
        used.append(name)
        if modifier == "count":
            counted.append(name)
        rendered = _render(held.value, modifier, hebrew)
        if modifier == "detail":
            detail_slots.append(match.group(0))
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

    embedded = _detail_inside_a_clause(template, detail_slots)
    if embedded:
        return Ungrounded(
            f"'{embedded}' is a TABLE and it is written inside a sentence, which reads as "
            "\"your plan is term winter · courses 6 · credits 16\". A `:detail` slot renders "
            "one labelled row per record, so it belongs on its own line, or after a label "
            "ending in a colon. If what you wanted was ONE number out of that row, `project` "
            "the field you mean and slot THAT, or `compute` it -- a slot cannot take a field "
            "off a collection."
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
    #
    # HOW the fact is cited does separate them, though, where completeness
    # could not. `{name}` over an empty collection renders "(none)" and says
    # nothing -- that is the non-answer this rule exists for. `{name:count}`
    # renders "0", which is a real derived number about a real query, and it is
    # the only way to state a true negative about a thing that does not exist.
    #
    # Without this, the system contradicted itself. The prompt orders: "ASKED
    # ABOUT A NAMED COURSE, CONFIRM IT EXISTS FIRST ... Nothing back means the
    # code is not in the catalog ... say that." Then every phrasing of it was
    # refused here. Live, asked "am I eligible for course 00999999?", the agent
    # searched the catalog twice, wrote three correct answers -- "I found
    # {catalog:count} records for that course" -- had all three rejected, and
    # returned the give-up sentence. A guard that forbids the behaviour the
    # prompt demands cannot be satisfied by any model.
    empty = [name for name in used if _is_empty(facts[name].value)]
    if len(empty) == len(set(used)) and not set(counted) & set(empty):
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


_LABEL_LEAD = re.compile(r"(?:^|\n)\s*(?:[-*\u2022]\s*)?[^\n]{0,60}?[:\u2014\u2013-]\s*$")


def _detail_inside_a_clause(template: str, slots: "list[str]") -> str:
    """The first `:detail` slot written into the middle of a sentence, if any.

    A one-record `:detail` used to carry a list bullet, which forced it onto its
    own line. Dropping the bullet was right -- a stranded "- " mid-sentence
    looked broken -- but it also made the row INLINE-ABLE, and the model started
    writing it into prose:

        "Your winter semester plan is term winter · courses 6 · credits 16."

    Correct, and not a sentence. The renderer cannot fix this, because the error
    is where the slot was PLACED, and that is visible only here.

    Deliberately permissive about what may precede it: a label ending in a
    colon or a dash ("Summary: {plan:detail}") is a legitimate introduction and
    reads fine. Only a slot with an ordinary clause running into it is refused.
    """
    for slot in slots:
        index = template.find(slot)
        if index <= 0:
            continue
        before = template[:index]
        if before.endswith("\n") or not before.strip():
            continue
        if _LABEL_LEAD.search(before):
            continue
        return slot
    return ""


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
_STRANDED_YES = re.compile(
    # An adverb may sit between the copula and the stranded word: a live answer
    # read "You are ALREADY yes eligible for 00960324", which the adjacent-only
    # pattern missed. At most one word, so this stays a local repair and cannot
    # reach across a clause.
    #
    # The lookahead is what keeps it a repair rather than a mangling. "The
    # answer is yes and the course is open" has a yes that IS the predicate, and
    # deleting it leaves "The answer is and the course is open" -- which the
    # original adjacent-only pattern did too, so this is an old bug the widening
    # would have made louder. A stranded yes is followed by the word it wrongly
    # qualifies; a real one is followed by a conjunction or a clause.
    # WHITELISTED by the word that follows, not blacklisted by conjunctions.
    # The blacklist deleted a real negation: "there are 0 transcript attempts,
    # so there is no grade on record" shipped as "so there is grade on record".
    # "no" before a NOUN is a determiner and carries the whole meaning of the
    # sentence, where "no" before a predicate adjective is the stranded bool
    # this repair is for -- and no blacklist of conjunctions can tell those
    # apart, because neither "grade" nor "record" is a conjunction.
    #
    # A rule that edits an answer which already passed grounding must fail
    # CLOSED. Listing the adjectives this domain actually produces means an
    # unfamiliar sentence is left exactly as written, which is the safe
    # direction; the old rule's unfamiliar sentence lost a word.
    r"\b(are|is|am|was|were)(\s+\w+)?\s+(yes|no)\s+"
    r"(?=(?:eligible|offered|available|expected|required|reachable|achievable"
    r"|possible|feasible|met|complete|completed|scheduled|open|full|passed"
    r"|allowed|permitted|exempt|valid)\b)",
    re.IGNORECASE,
)


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
    tidied = _STRANDED_YES.sub(_repair_stranded, tidied)
    tidied = _STUTTERED_PHRASE.sub(lambda m: m.group(1), tidied)
    return _SCRIPT_SEAM.sub(_repair_seam, tidied)


_HEBREW_PREFIX = set("בלכמשהו")
"""The single letters Hebrew writes closed up against the next word.

Only these take the hyphen form before a foreign word. A Hebrew letter that ends
a whole word ("תקיןOK") is a plain run-together and wants an ordinary space."""


def _repair_seam(match: "re.Match[str]") -> str:
    """Separate the two scripts, the way Hebrew actually writes the seam.

    A lone prefix letter takes a hyphen -- ל-an, ב-Winter -- which is the
    standard form for attaching one to a foreign word. Anything else takes a
    space. Both are only ever an INSERTION: no character on either side is
    changed, so this cannot alter what the answer claims.
    """
    if match.group(1):
        hebrew, latin = match.group(1), match.group(2)
        joiner = "-" if _is_lone_prefix(match.string, match.start(1), hebrew) else " "
        return f"{hebrew}{joiner}{latin}"
    return f"{match.group(3)} {match.group(4)}"


def _is_lone_prefix(text: str, index: int, letter: str) -> bool:
    """A prefix letter standing alone -- start of text, or preceded by a space."""
    if letter not in _HEBREW_PREFIX:
        return False
    return index == 0 or not text[index - 1].isalpha()


_SCRIPT_SEAM = re.compile(r"([֐-׿])([A-Za-z])|([A-Za-z])([֐-׿])")
"""A Hebrew letter and a Latin letter with nothing between them.

Asked in Hebrew, the model answers in Hebrew, and the facts it slots are often
English -- the regulations corpus is English, so `interpret` returns English
phrases. That is fine until a slot lands after one of Hebrew's single-letter
prefixes, which attach directly to the word they govern with no space. A live
answer shipped:

    אתה זכאי לan additional exam date

The template was "אתה זכאי ל{entitlement}" and it is CORRECT Hebrew -- ל is
written closed up against a Hebrew word. Against a Latin one it needs the
space, and the model cannot know which it will get because it never sees the
value. So the seam is repaired here, where both sides are known.

Digits are deliberately excluded. "2 סמסטרים" already spaces itself, and Hebrew
attaches numerals with a hyphen ("ב-10 נקודות") which is a separator this would
not see anyway -- adding a rule for them would only put spaces inside forms that
are already right."""


def _repair_stranded(match: "re.Match[str]") -> str:
    """Drop a stranded "yes"; turn a stranded "no" into "not".

    Deleting either was a meaning INVERSION in the negative case, and it was
    here from the beginning: "You are no eligible for 01040174" -- the bool
    rendering of an ineligible verdict -- came out as "You are eligible for
    01040174", telling a student they may register for something they cannot.
    The same edit that merely tidies an affirmative reverses a denial, because
    "no" is doing the negating and "yes" is not doing anything.
    """
    copula, adverb, verdict = match.group(1), match.group(2) or "", match.group(3)
    if verdict.lower() == "no":
        return f"{copula}{adverb} not "
    return f"{copula}{adverb} "


_STUTTERED_PHRASE = re.compile(r"\b(\w+(?:\s+\w+){2,})\s+\1\b", re.IGNORECASE)
"""A phrase written twice, back to back, because a TEXT slot already contained it.

The bool case above wearing a longer coat. `interpret` returned the English
requirement as the sentence "Cannot graduate without completing 2 English
courses", the model wrote a sentence around it, and the answer shipped as:

    "You cannot graduate without completing Cannot graduate without completing
     2 English courses."

Correct, grounded, and unreadable -- and it scored as a PASS, because the number
it had to state was in there.

Three words minimum and immediately adjacent, so it only ever collapses a
stutter. Two-word repeats occur in ordinary prose ("had had", "that that"), and
a repeat separated by other words is usually deliberate. Like the rewrites
above, this drops a redundancy and changes nothing the answer CLAIMS."""


def _is_empty(value: Union[Collection, Scalar]) -> bool:
    return isinstance(value, Collection) and not value.records


def _render(
    value: Union[Collection, Scalar], modifier: str | None, hebrew: bool = False
) -> str:
    if isinstance(value, Scalar):
        return _render_scalar(value, hebrew)

    if modifier == "count":
        return str(len(value.records))
    if modifier == "detail":
        return _render_detail(value, hebrew)
    if modifier == "list" or modifier is None:
        rendered = [_readable_field(record, hebrew) for record in value.records]
        rendered = [text for text in rendered if text]
        # DEDUPED, first occurrence wins. A search hit list holds one record per
        # PASSAGE, and several passages come from one page, so the readable
        # field repeats: a live Hebrew refusal listed "Undergraduate Study
        # Regulations (Technion)" twelve times in one sentence. Repeating a name
        # says nothing the first mention did not, and the honest count is still
        # available as `{name:count}`.
        rendered = list(dict.fromkeys(rendered))
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


def _render_detail(value: Collection, hebrew: bool = False) -> str:
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
    # A ONE-record detail is a phrase, not a list, and the model writes it into
    # the middle of a sentence -- a single-term summary came out as
    # "סה״כ התכנון הוא - term winter · courses 6 · credits 16", with a list
    # bullet stranded mid-clause. Nothing is being enumerated, so nothing needs
    # a bullet; two or more rows still get one, because then it IS a list.
    bullet = len(value.records) > 1
    lines = [
        line
        for line in (_detail_line(record, bullet, hebrew) for record in value.records)
        if line
    ]
    if not lines:
        return "(none)"
    if len(lines) > _DETAIL_CAP:
        shown = "\n".join(lines[:_DETAIL_CAP])
        return f"{shown}\n...and {len(lines) - _DETAIL_CAP} more"
    return "\n".join(lines)


def _detail_line(record: Collection, bullet: bool = True, hebrew: bool = False) -> str:
    parts = [
        f"{_readable_label(name)} {_render_scalar(v, hebrew)}"
        for name, v in record.fields.items()
        if isinstance(v, Scalar) and name != "_id" and not _OBJECT_ID.fullmatch(str(v.value))
    ]
    if not parts:
        return ""
    return ("- " if bullet else "") + " · ".join(parts)


_LABEL_SEAM = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _readable_label(name: str) -> str:
    """A projected field name, as a person would read it.

    The model names the fields, and it names them the way code names
    identifiers, because a fact name IS a code identifier everywhere else in
    this system -- `prereqStatus`, `min_grade`, `courseNumber`. The prompt asks
    for reader-facing names and gets schema-shaped ones, measured across every
    row in the evaluation traces.

    So the ask stays in the prompt, where the real fix is, and this is the net
    under it: purely typographic, no vocabulary and no domain knowledge, so a
    field this has never seen still comes out better than it went in.
    """
    spaced = _LABEL_SEAM.sub(" ", name.replace("_", " ")).strip()
    if spaced.isascii() and not spaced.isupper():
        return spaced.lower()
    # An all-caps label is an acronym the student knows -- GPA, not "gpa" --
    # and a non-ASCII one is already in the reader's own script.
    return spaced


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


def _readable_field(record: Collection, hebrew: bool = False) -> str:
    """The field of a record worth showing a person.

    PREFERRED BY NAME first, then by elimination. The old rule took the first
    field that was not an `_id` or ObjectId-shaped, and "first" is an accident of
    the schema: a `prerequisite_edges` record begins with `edge`, so `{prereqs}`
    rendered `00960211->00940224, 00960211->00940226` and
    `check_no_edge_identifiers` then refused the whole answer.

    That was the single most common wasted turn across 38 measured runs -- 12 of
    them -- and the loop cannot repair it, because the rendering is not something
    the model chose. It slotted a fact and the renderer picked the internal key.

    The preferred names are the ones a student recognises, in the order the
    domain wants them: a prerequisite edge is interesting for what it REQUIRES,
    everything else for its course number or title. Falling through to the old
    elimination keeps any record shape working, now also skipping edge and group
    ids -- the two other tokens the post-conditions refuse, and for the same
    reason: they look like courses and are not.
    """
    scalars = [(name, v) for name, v in record.fields.items() if isinstance(v, Scalar)]
    by_name = {name: v for name, v in scalars}
    for preferred in _PREFERRED_READABLE:
        value = by_name.get(preferred)
        if value is not None and str(value.value).strip():
            return _render_scalar(value, hebrew)

    readable = [
        v for name, v in scalars
        if name != "_id"
        and not _OBJECT_ID.fullmatch(str(v.value))
        and not _EDGE_ID_VALUE.fullmatch(str(v.value))
        and not _GROUP_ID_VALUE.fullmatch(str(v.value))
    ]
    chosen = readable[0] if readable else (scalars[0][1] if scalars else None)
    return _render_scalar(chosen, hebrew) if chosen is not None else ""


_PREFERRED_READABLE = (
    # A prerequisite edge is interesting for what it REQUIRES -- which is also
    # exactly what `check_no_edge_identifiers` tells the model to name.
    "requires",
    "courseNumber",
    "course",
    "number",
    "code",
    "title",
    "name",
)

_EDGE_ID_VALUE = re.compile(r"\d{6,8}\s*->\s*\d{6,8}")
_GROUP_ID_VALUE = re.compile(r"\d{6,8}\.\d+")


_DECIMALS = 2
"""How many decimals a derived number shows a reader.

Two, because every quantity in this domain is a grade, a GPA or a credit count,
and none of them are meaningful past hundredths. The value is NOT rounded in the
fact -- only in the rendering -- so arithmetic downstream still uses the full
precision it was computed with.
"""


_ENUM_WORDS: dict[str, tuple[str, str]] = {
    # stored value      ->  (English, Hebrew)
    "mandatory":            ("mandatory", "חובה"),
    "elective":             ("elective", "בחירה"),
    "satisfied":            ("met", "מולאו"),
    "met":                  ("met", "מולאו"),
    "check_prerequisites":  ("check first", "יש לבדוק"),
    "check_corequisites":   ("check first", "יש לבדוק"),
    "unmet":                ("NOT met", "לא מולאו"),
    "none":                 ("none", "אין"),
}
"""Stored values that reach the reader, in the words a reader uses.

The planner writes `prereqStatus: "check_prerequisites"` and the catalog writes
`category: "elective"`, and `:detail` printed them verbatim, so a live Hebrew
plan shipped rows reading `סוג mandatory · דרישות met` -- English enum values
inside Hebrew prose. `check_prerequisites` is worse than untranslated: it is an
instruction to the system, and a student reading it has been handed a variable
name.

This is a presentation decision in code, which `_render_scalar` already makes:
a bool has been printed as "yes"/"no" since the beginning, for exactly this
reason. So the boundary is not new here, only wider. It stays narrow in the
ways that matter -- an explicit table, a closed vocabulary read off the two
modules that emit it (`planning/term_plan.py`, the track catalog), and ANY
value not listed passes through untouched, so an enum nobody anticipated is
rendered as-is rather than mangled.

Kept SHORT on purpose. The label beside them is the model's to name, so the
value has to compose with whatever it picks: "prerequisites met" reads well
until the label is already `prerequisites`, and then the row says "prerequisites
meets-the-prerequisites". A short value degrades gracefully in both directions.

What it deliberately does NOT do is translate free text. Course titles, faculty
names and interpreted regulation phrases are data, and guessing at their
language is how a correct answer becomes a wrong one."""


def _render_scalar(value: Scalar, hebrew: bool = False) -> str:
    if isinstance(value.value, bool):
        return ("כן" if value.value else "לא") if hebrew else ("yes" if value.value else "no")
    if isinstance(value.value, str):
        words = _ENUM_WORDS.get(value.value.strip().lower())
        if words is not None:
            return words[1] if hebrew else words[0]
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
