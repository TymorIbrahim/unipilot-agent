"""The facts layer's production entry point.

`/advise` calls this. It is the one place the fact/tool loop is assembled for a
real request: a `DispatchContext` wired from the running settings, the asking
student's identity seeded as the `me` fact, the chat adapter built, and the
loop run under a turn budget.

Everything the route needs to answer is DERIVED here from the loop's own result
-- the answer text, its confidence, the course codes it grounded, the outcome
status -- so the HTTP layer stays a thin shape-translation and never reaches
into the working set itself.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agent_core.facts.adapter import ChatModelAdapter, build_system_prompt
from app.agent_core.reasoning.llm_client import build_chat_llm
from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.conversation import SupabaseConversations
from app.agent_core.facts.loop import MAX_TURNS, LoopResult, run_loop
from app.agent_core.facts.types import Basis, Scalar, ScalarKind
from app.agent_core.facts.wiring import ModelExtractor, build_context
from app.agent_core.loop.course_names import course_codes_in, course_display_name

logger = logging.getLogger(__name__)

# Outcome (facts loop) -> the frontend's retrieval_agent.status vocabulary.
# A declined or proposed question IS a completed, valid response: the system
# answered by declining an out-of-scope ask or by preparing a change for
# approval. A refusal or a spent budget did NOT answer, and says so.
_STATUS_BY_OUTCOME = {
    "answered": "succeeded",
    "declined": "succeeded",
    "proposed": "succeeded",
    "refused": "incomplete",
    "stalled": "incomplete",
    "exhausted": "incomplete",
}

# When the loop could not answer, the reason is diagnostic prose meant for a
# developer, not a student. This is what the student sees instead.
_COULD_NOT_ANSWER = (
    "I wasn't able to work that out from your records with confidence. Could you rephrase it, "
    "or ask about something more specific?"
)


@dataclass(frozen=True)
class Advice:
    """What the route needs, all derived from the loop result."""

    answer: str
    confidence: str
    course_ids: list[str]
    status: str
    sources: list[str]
    outcome: str


async def run_advice(
    question: str,
    user_id: str,
    *,
    settings: Any | None = None,
    on_progress: Callable[[str], None] | None = None,
    max_turns: int = MAX_TURNS,
    time_budget_s: float | None = None,
    conversation_id: str | None = None,
    chat: Any | None = None,
    started_at: float | None = None,
) -> LoopResult:
    """Run the fact loop for one student's question.

    The student's identity is the one fact GIVEN rather than derived -- the loop
    cannot ask who is asking, and every "my records" filter resolves through it.

    `time_budget_s`, when set, bounds the whole request by the wall clock and
    lets the turn count run free -- so a hard question can take as many steps as
    it needs inside the window rather than being cut off at a fixed turn count.

    `conversation_id` threads a follow-up to its predecessors: the prior
    exchanges are loaded so a message like "continue" resolves, and this run's
    answer is appended when it concludes. Only the TEXT is carried -- facts are
    re-derived fresh every run, so a follow-up is grounded in live records, not
    in a snapshot from a previous turn.

    `chat` REPLACES the client this would otherwise build. It exists so
    `/api/execute` can hand in a traced client and have every model call --
    the reasoning turns and the extractor's, which are built here rather than
    by the caller -- land in one ordered `steps` log. Passing the client rather
    than a finished adapter is what makes that possible: a finished adapter would
    cover the reasoning turns and leave the extractor untraced, which is exactly
    the kind of partial record the spec's `steps` must not be.
    """
    from app.db.postgres import get_database

    if chat is None and build_chat_llm(settings=settings) is None:
        # No credentials. A loop with no model cannot run; surface it as an
        # honest non-answer rather than crashing the route.
        return LoopResult(
            outcome="exhausted",
            reason="no language model is configured",
            question=question,
        )

    database = await get_database()

    profile = await _profile_of(database, user_id)
    if profile is None:
        # The failure this whole layer exists to prevent, arriving through the
        # one door nothing guarded. An unknown student has no transcript, so
        # every `find` returns an empty collection, `sum` over empty is 0, and
        # the run answers "You have completed 0 credits" -- confident, grounded
        # in a real (empty) fetch, and about nobody. Measured in production with
        # student_id=nonexistent-student: status ok, six steps, that answer.
        #
        # Checked once here rather than defended against downstream, because
        # every tool would have to know, and any that forgot would produce the
        # same confident zero.
        return LoopResult(
            outcome="refused",
            reason=f"{UNKNOWN_STUDENT}: {user_id!r} has no record, so there is nothing to answer from",
            question=question,
        )

    context = build_context(
        database, settings, _audience_of_profile(profile), **_extractor_override(chat)
    )

    # AFTER the context, because the system prompt now carries the tool catalog
    # and the source list, and both are read off the context. Building the
    # adapter first would have to guess at them.
    adapter = ChatModelAdapter(
        chat if chat is not None else build_chat_llm(settings=settings),
        build_system_prompt(context),
    )
    context.facts["me"] = HeldFact(
        value=Scalar(ScalarKind.IDENTIFIER, user_id),
        basis=Basis.OFFICIAL_RECORD,
    )
    _seed_profile_facts(context, profile)

    store = SupabaseConversations(database)
    # Scope the conversation to the ASKING student, so one student's id can never
    # load another's thread even if a client sent a guessed conversation_id.
    thread_key = f"{user_id}:{conversation_id}" if conversation_id else None
    history = await store.history(thread_key) if thread_key else []

    # With a wall-clock budget the turn count must not be the thing that stops a
    # run first, so raise it out of the way and let the clock govern.
    turns = max(max_turns, 100) if time_budget_s is not None else max_turns
    result = await run_loop(
        question,
        adapter,
        context,
        max_turns=turns,
        on_progress=on_progress,
        time_budget_s=time_budget_s,
        history=history,
        seeded_facts=SEEDED_FACT_NAMES,
        started_at=started_at,
    )

    # Record the exchange so the NEXT message can continue it. Only a real
    # student-facing answer is worth remembering; a bare non-answer would just
    # clutter the thread.
    if thread_key and result.answer is not None:
        await store.append(thread_key, question, result.answer.text)

    return result


UNKNOWN_STUDENT = "no student record"
"""Marks a refusal caused by the CALLER naming a student who does not exist.

Shared with `runner._error_for`, which otherwise summarises a refusal into one
generic sentence to avoid leaking developer diagnostics. This one is not a
diagnostic -- it is a client mistake with a precise cause, like an empty prompt
-- and a caller who mistypes an id deserves to be told that rather than
"the answer could not be grounded in the student's records".
"""

_GRADUATE_PROGRAM_TYPES = frozenset({"msc", "ma", "phd", "me", "meng", "mba"})


async def _profile_of(database: Any, user_id: str) -> Any:
    """The student's profile row, or None if there is no such student.

    One query, serving two purposes: proving the student EXISTS before a run
    starts, and carrying the degree level retrieval needs. Both were previously
    unasked -- the level not at all, and the existence never.

    A read that FAILS is not the same as a student who does not exist, so an
    error propagates rather than returning None: a database outage reported as
    "no such student" is the confident-wrong-answer failure wearing a different
    hat.
    """
    rows = await database.fetch(
        'select p."userId", p."programType", p."programSlug", p."catalogYear", '
        'p."currentSemesterCode", p."maxCreditsPerSemester", '
        # The credit standing, in the query that was already running. Every
        # question about credits, remaining work or graduation timing needed
        # these three numbers and spent one to two turns per run deriving them:
        # find degree_programs, find completed, sum creditsCounted, subtract.
        # At ~15s a turn against a 60s ceiling that is a quarter of the budget
        # to learn something one join already knew.
        'd."totalCredits" as "creditsRequired", '
        'coalesce(t.done, 0) as "creditsCompleted" '
        "from student_profiles p "
        'left join degree_programs d on d."_id" = p."degreeId" '
        "left join lateral ("
        '    select sum(cc."creditsCounted") as done from completed_courses cc'
        '    where cc."userId" = p."userId" and cc."passed"'
        ") t on true "
        'where p."userId" = $1',
        user_id,
    )
    return rows[0] if rows else None


_SEEDED_PROFILE_FIELDS: tuple[tuple[str, str, ScalarKind], ...] = (
    ("programSlug", "program_slug", ScalarKind.IDENTIFIER),
    ("catalogYear", "catalog_year", ScalarKind.QUANTITY),
    ("currentSemesterCode", "current_semester", ScalarKind.IDENTIFIER),
    ("maxCreditsPerSemester", "max_credits_per_semester", ScalarKind.QUANTITY),
)
"""Profile columns handed to the loop as opening facts.

This query already ran, to prove the student exists and to pick the right
rulebook. It was selecting two of its columns and discarding the rest -- and
then every planning run opened like this:

    turn 1  find    -> profile          re-fetching the row we already held
    turn 2  compute -> program_slug     unpacking it
    turn 3  ...the actual work begins

Two model calls per planning question to re-derive something the server had in
hand before the loop started. Widening the select costs no extra round trip and
no extra tokens; seeding the results deletes both turns.

Only stable identity fields, deliberately. Anything the student's ANSWER depends
on -- credits, grades, courses -- stays behind a tool call, so it arrives with a
basis and a completeness the loop can reason about rather than as a fact that
appeared from nowhere.
"""

_CREDIT_STANDING_FACTS: frozenset[str] = frozenset(
    {"credits_completed", "credits_required", "credits_needed"}
)

IDENTITY_FACTS: frozenset[str] = frozenset(
    {"me"} | {fact_name for _column, fact_name, _kind in _SEEDED_PROFILE_FIELDS}
)
"""Seeded facts that say WHO is asking, not what the run found out.

Repeating these back is the question restated: a partial answer reading "max
credits per semester: 18" tells a student something they were never asking
about, and a live partial did exactly that."""

SEEDED_FACT_NAMES: frozenset[str] = IDENTITY_FACTS | _CREDIT_STANDING_FACTS
"""Every fact the ROUTE puts in the context before the loop's first turn.

Derived from the tuples above rather than listed again, because the last time
these were kept separately they drifted: `run_loop` decided whether a decline
was honest by testing `name != "me"`, this list grew by four, and the agent
quietly lost the ability to decline anything at all -- an out-of-scope question
ran three turns and came back "I wasn't able to work that out from your
records." Adding a seeded field must not be able to break that again.

Split from `IDENTITY_FACTS` because the two callers want different things. The
decline guard asks "did the model FETCH anything", so everything seeded belongs
here. A partial answer asks "what did the run ESTABLISH", and the credit
standing is worth reporting there even though it was seeded -- "you have
completed 129.5 of 155 credits" is a real partial answer, where the student's
own id is not."""


def _seed_profile_facts(context: Any, profile: Any) -> None:
    for column, fact_name, kind in _SEEDED_PROFILE_FIELDS:
        value = profile[column] if column in profile.keys() else None
        if value in (None, ""):
            continue  # absent beats invented; the model can still `find` it
        context.facts[fact_name] = HeldFact(
            value=Scalar(kind, float(value) if kind is ScalarKind.QUANTITY else str(value)),
            basis=Basis.OFFICIAL_RECORD,
            derivation="from the student's profile, read when the run started",
        )
    _seed_credit_standing(context, profile)


def _seed_credit_standing(context: Any, profile: Any) -> None:
    """Credits required, completed, and the gap between them.

    A DELIBERATE reversal of the rule above it, which says answer-bearing facts
    stay behind a tool call so they arrive with a basis and a completeness the
    loop can reason about. Two things changed since that was written.

    The ceiling turned out to be 60s rather than the 300s this project believed
    it had, so a turn is now a twentieth of the whole request. Deriving these
    three numbers cost one to two turns of every credit question -- find
    degree_programs, find completed, sum `creditsCounted`, subtract -- and they
    come free in the profile query that already runs.

    And `remaining` is the word this domain gets wrong most often: it has meant
    the credits still to EARN and the credits still on OFFER in different runs of
    the same question, differing by a factor of two, and each confusion has cost
    a defect. Deriving the gap ONCE, in SQL, is what stops it being derived two
    ways.

    The rule's actual concern is traceability, and it is preserved: each fact
    carries the basis and a derivation naming the columns it came from, so the
    answer layer renders "25.5 (degree_programs.totalCredits minus completed
    credits)" exactly as it would for a fact the model fetched itself.

    Absent beats invented, as ever -- a student whose degree has no
    `totalCredits` gets `credits_completed` alone and the model can still work
    the rest out.
    """
    required = profile["creditsRequired"] if "creditsRequired" in profile.keys() else None
    completed = profile["creditsCompleted"] if "creditsCompleted" in profile.keys() else None

    if completed is not None:
        context.facts["credits_completed"] = HeldFact(
            value=Scalar(ScalarKind.QUANTITY, float(completed)),
            basis=Basis.OFFICIAL_RECORD,
            derivation="sum of completed_courses.creditsCounted over passed rows",
        )
    if required in (None, ""):
        return
    context.facts["credits_required"] = HeldFact(
        value=Scalar(ScalarKind.QUANTITY, float(required)),
        basis=Basis.OFFICIAL_RECORD,
        derivation="degree_programs.totalCredits for this student's degree",
    )
    if completed is not None and float(required) > float(completed):
        context.facts["credits_needed"] = HeldFact(
            value=Scalar(ScalarKind.QUANTITY, float(required) - float(completed)),
            basis=Basis.OFFICIAL_RECORD,
            derivation=(
                "degree_programs.totalCredits minus completed credits -- the credits still to "
                "EARN, which is NOT the credits of the courses still on offer in the track"
            ),
        )


def _audience_of_profile(profile: Any) -> str | None:
    """The student's degree level, so retrieval never quotes the wrong rulebook.

    The Technion's undergraduate and graduate regulations both answer "what is
    the English requirement", and a live BSc student was told "All graduate
    students must demonstrate English proficiency" because the graduate page
    out-ranked the undergraduate one. Nothing downstream could catch that -- the
    quote was faithful and the citation real -- so the level has to be known
    BEFORE the corpus is searched.

    Returns None for an unrecognised programType, which leaves retrieval
    unfiltered. An unknown level must not silently narrow the corpus to nothing.
    """
    program_type = profile["programType"] if profile is not None else None
    if not program_type:
        return None
    return "graduate" if str(program_type).strip().lower() in _GRADUATE_PROGRAM_TYPES else "undergraduate"


def _extractor_override(chat: Any | None) -> dict[str, Any]:
    """Route the prose extractor through the SAME client the caller supplied.

    `build_wiring` builds its own extractor from `build_chat_llm`, which is
    correct when nobody is watching and wrong when someone is: `interpret` and
    `extract_list` are model calls, the spec requires every model call to appear
    in `steps`, and an extractor holding a second, untraced client would make
    those calls invisible. The override is empty when no client was supplied, so
    the normal path is untouched.
    """
    return {"extractor": ModelExtractor(chat)} if chat is not None else {}


def to_advice(result: LoopResult) -> Advice:
    """Map a loop result to the fields the route ships. Pure and total."""
    answer = _answer_text(result)
    return Advice(
        answer=answer,
        confidence=_confidence(result),
        course_ids=_course_ids(answer, result),
        status=_STATUS_BY_OUTCOME.get(result.outcome, "incomplete"),
        sources=_sources(result),
        outcome=result.outcome,
    )


def _answer_text(result: LoopResult) -> str:
    """The student-facing prose for every outcome the loop can reach."""
    if result.outcome == "answered" and result.answer is not None:
        return result.answer.text
    if result.outcome == "declined":
        # The model's own words for why it is out of scope.
        return result.reason or "That is outside what I can help with."
    if result.outcome == "proposed" and result.proposal is not None:
        name = course_display_name(result.proposal.target) or result.proposal.target
        return (
            f"I've prepared a request to {result.proposal.action} {name}. Nothing has been "
            "changed yet -- it needs your confirmation before anything happens."
        )
    # refused / stalled / exhausted -- the reason is diagnostic, not for a
    # student. But the run is rarely empty-handed, and saying nothing throws
    # away work the student can use.
    return _partial_from_facts(result) or _COULD_NOT_ANSWER


_PARTIAL_PREFIX_BY_OUTCOME = {
    "exhausted": "I ran out of time before I could finish that, but here is what I established "
                 "from your records:",
    "refused": "I could not put the whole answer together with confidence, but here is what I "
               "established from your records:",
    "stalled": "I stopped making progress on that, but here is what I established from your "
               "records:",
}
"""Why the run did not finish, said accurately.

Every partial used to open "I ran out of time", whatever had happened. Measured:
a run that reached its answer in three turns and was REFUSED told the student it
had run out of time, at 15.5 seconds of a 240-second budget. The sentence was a
claim about a cause, and the system knew it to be false.

A small thing, and exactly the kind this project is about: the outcome is right
there in the result, and asserting a different one is the same error as any
other confident wrong number."""
_PARTIAL_SUFFIX = (
    "Ask me for one piece at a time -- a single term's plan, or just the number of "
    "semesters -- and I can finish it."
)

_HEBREW = re.compile(r"[֐-׿]")

_PARTIAL_PREFIX_BY_OUTCOME_HE = {
    "exhausted": "לא הספקתי לסיים את זה, אבל הנה מה שכן הצלחתי לברר מהרשומות שלך:",
    "refused": "לא הצלחתי להרכיב את התשובה המלאה בביטחון, אבל הנה מה שכן בררתי "
               "מהרשומות שלך:",
    "stalled": "נתקעתי באמצע, אבל הנה מה שכן הצלחתי לברר מהרשומות שלך:",
}
_PARTIAL_SUFFIX_HE = (
    "אפשר לשאול אותי דבר אחד בכל פעם — תכנון של סמסטר בודד, או רק מספר "
    "הסמסטרים — ואז אוכל להשלים."
)
"""The same three sentences, for a question asked in Hebrew.

The ANSWERED path needed nothing: asked in Hebrew the model replies in Hebrew,
nine live runs out of ten. This path is different in kind -- it is assembled in
code, precisely because there is no budget left for a model call -- so it said
what it was written to say regardless of who was reading. A Hebrew question that
ran out of budget came back apologising in English and then listing its facts.

A lookup, not a translation call: the failing path is the one that must not cost
anything."""
def _not_worth_reporting() -> frozenset[str]:
    """Facts the ROUTE seeded, which the run did not learn.

    Echoing them back is the question restated, not a partial answer, so a
    partial made only of these is worse than admitting nothing. Hand-listing
    them missed `max_credits_per_semester` and a live partial came back as the
    single line "max credits per semester: 18" -- seeded before the first turn,
    and presented as what the run had established.

    `IDENTITY_FACTS`, not `SEEDED_FACT_NAMES`: the credit standing is seeded too
    and IS worth reporting -- "you have completed 129.5 of 155 credits" is a real
    partial answer, where the student's own id is not."""
    return IDENTITY_FACTS


def _partial_from_facts(result: LoopResult) -> str | None:
    """What the run DID establish, when it could not reach an answer.

    `config.py` has promised for a while that a run out of budget "ships a
    grounded partial answer rather than being killed". It never did: every
    non-answer became the same one sentence, and a run that had derived the
    student's completed credits, the credits they still need and their
    per-semester cap reported none of it.

    That matters more now the ceiling is 60s rather than the 300s this project
    believed it had, because the questions that exhaust it are the substantial
    ones -- and they are precisely the runs holding the most.

    Deterministic, and only SCALARS the run actually derived, rendered with the
    names they were derived under. No model call: there is by definition no time
    left for one. Nothing is inferred or combined -- combining is the step that
    ran out of time, and guessing at it here would be inventing the answer the
    loop declined to give.
    """
    scalars = [
        (name, held)
        for name, held in (result.facts or {}).items()
        if name not in _not_worth_reporting()
        and isinstance(held.value, Scalar)
        and held.value.kind is not ScalarKind.TEXT
        and held.value.value not in (None, "")
    ]
    if not scalars:
        return None
    lines = [f"- {name.replace('_', ' ')}: {_render_scalar(held.value)}" for name, held in scalars]
    hebrew = bool(_HEBREW.search(result.question or ""))
    prefixes = _PARTIAL_PREFIX_BY_OUTCOME_HE if hebrew else _PARTIAL_PREFIX_BY_OUTCOME
    prefix = prefixes.get(result.outcome, prefixes["refused"])
    suffix = _PARTIAL_SUFFIX_HE if hebrew else _PARTIAL_SUFFIX
    # The fact NAMES stay as derived -- they are the names the run used, and
    # inventing Hebrew for them here would be labelling a number with a word no
    # part of the system ever called it.
    return "\n".join([prefix, *lines, "", suffix])


def _render_scalar(value: Scalar) -> str:
    number = value.value
    if isinstance(number, bool):
        return "yes" if number else "no"
    if isinstance(number, float) and number.is_integer():
        return str(int(number))
    return str(number)


def _confidence(result: LoopResult) -> str:
    """low / medium / high, banded by the answer's weakest grounded basis.

    A non-answer is always low. An answer is only as strong as the weakest thing
    it stands on, so an interpreted or predicted fact honestly pulls the band
    down from a pure official record -- the same principle the basis ordering
    enforces everywhere else in the layer.
    """
    if result.outcome != "answered" or result.answer is None:
        return "low"
    strength = result.answer.basis.strength
    if strength >= Basis.OFFICIAL_RECORD.strength:
        return "high"
    if strength >= Basis.LLM_INTERPRETATION.strength:
        return "medium"
    return "low"


def _course_ids(answer: str, result: LoopResult) -> list[str]:
    """Course codes the answer names that a grounded fact also carries.

    Not model-authored: facts are the loop's only channel for admitted data, so
    intersecting the answer's codes against the facts keeps a hallucinated
    8-digit number out even if it reached the prose. Mirrors the V2 route's
    `_mentioned_course_ids`, which this replaces.
    """
    if not answer:
        return []
    import json

    grounded = course_codes_in(
        json.dumps([held.value for held in result.facts.values()], default=str)
    )
    return sorted(course_codes_in(answer) & grounded)


def course_references(course_ids: list[str]) -> list[dict[str, str]]:
    """Each id with its display name, falling back to the bare id."""
    return [{"id": cid, "name": course_display_name(cid) or cid} for cid in course_ids]


def _sources(result: LoopResult) -> list[str]:
    """A provenance hint per corpus search the loop ran -- the query term, taken
    from the transcript, never the passage text."""
    sources: set[str] = set()
    for turn in result.transcript:
        if turn.action == "call" and turn.detail.startswith("search_corpus("):
            # The transcript records `search_corpus({"query": "..."}) -> ...`.
            start = turn.detail.find('"query": "')
            if start != -1:
                start += len('"query": "')
                end = turn.detail.find('"', start)
                if end != -1:
                    sources.add(f"search: {turn.detail[start:end]}")
    return sorted(sources)


__all__ = ["Advice", "course_references", "run_advice", "to_advice"]
