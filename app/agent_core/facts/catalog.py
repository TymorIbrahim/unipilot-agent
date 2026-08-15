"""The tool catalog -- phase 9b of docs/agent/tools_implementation_plan.md.

The other half of constructibility. The codec accepts good JSON; this is what
makes the model write good JSON in the first place.

**Generated, not hand-written.** The operator table and the sugar list are
rendered from `OPERATORS` and `SUGAR` at call time, so adding an operator
updates the prompt and cannot be forgotten. The previous system kept tool names
in hand-maintained prose -- including cross-references inside OTHER tools'
notes -- and they drifted: a tool deleted from the registry went on being
described, which corrupted an ablation experiment before anyone noticed the
prompt still mentioned it.

Every example here is executed by the test suite. A prompt that teaches the
model malformed JSON is worse than one that says nothing, because the model has
no reason to doubt it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agent_core.facts.operators import OPERATORS, SUGAR


@dataclass(frozen=True)
class ToolSpec:
    name: str
    purpose: str
    when: str
    example: dict[str, Any]
    requires: str = ""
    """A dependency on the CONTEXT -- a service the process must have wired."""

    needs_source: str = ""
    """A capability the MODEL must be able to obtain by calling tools.

    Distinct from `requires`, and the distinction cost a round: wiring
    `prerequisite_edges()` as a Python function made `traverse` reachable from
    test code while leaving it unreachable from the model, because no `find`
    source yields edges. The reachability test passed by seeding the fact
    itself -- something no model can do. Reachable-from-code is not reachable.
    """


PRIMITIVES: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="find",
        purpose="Read structured records matching a predicate.",
        when=(
            "Any time you need records. Fetching one thing by identity is just the predicate "
            "`id = X` -- there is no separate get-by-id tool. To filter by a value you already "
            'HOLD, write {"fact": "name"} as the value: {"path": "userId", "op": "=", '
            '"value": {"fact": "me"}}. Writing the fact NAME as a plain string filters for the '
            "literal text of the name and matches nothing.\n"
            "     A held fact is a COLLECTION even when it has one record, so to filter by a field "
            'of it name the field: {"path": "course", "op": "=", "value": {"fact": "next_course", '
            '"field": "courseNumber"}} pulls the single courseNumber out of a one-record fact '
            "(and refuses if it holds more than one).\n"
            "     To fetch the records a collection you HOLD points at -- the catalog rows for the "
            'courses on your transcript, say -- filter with `in` over that fact\'s field: '
            '{"path": "_id", "op": "in", "value": {"fact": "completed", "field": "courseId"}}. '
            "That is the join you reach for constantly (a transcript stores courseId, not the "
            "course code); do NOT pull a whole catalog into `compute` to match against.\n"
            "     `limit` defaults to 200, and a fetch that hits it is TRUNCATED -- a count or "
            "total over a truncated collection is refused as a confident wrong number. Filter it "
            "down, or raise the limit past the true total."
        ),
        example={
            "tool": "find",
            "as": "my_courses",
            "args": {
                "source": "completed_courses",
                "predicate": {"path": "userId", "op": "=", "value": {"fact": "me"}},
                "limit": 200,
            },
        },
    ),
    ToolSpec(
        name="search_corpus",
        purpose="Find passages of the knowledge base -- the program and policy wiki.",
        when=(
            "For anything WRITTEN rather than recorded as a student's data. This is where the "
            "DEGREE STRUCTURE lives: a track/program's required courses, its ELECTIVE categories "
            "and how many credits of each are needed, the credit breakdown, course descriptions, "
            "and the academic regulations. The `find` sources hold a student's own records "
            "(transcript, plan, profile) and the raw catalog -- they do NOT hold which courses a "
            "degree requires or which count as electives. That is here. Returns CANDIDATES with "
            "scores, never an answer; reading one is a separate `interpret` step.\n"
            "     Two KINDS of page answer differently, and picking the wrong kind gives a real "
            "quote to the wrong question. A `regulations-*` page states the rules that apply to "
            "EVERY student -- the English and Hebrew requirements, the pass mark, GPA minimums, "
            "graduation conditions. A `track-*` page states what THIS degree adds: its course "
            "list, its electives, and notes about specific courses. For 'what must I do to "
            "graduate' read the REGULATIONS; a track page's line on the same topic is usually a "
            "note about one course, and answering from it gives a deadline where the requirement "
            "was asked for. When both look relevant, read both and say which is which."
        ),
        example={"tool": "search_corpus", "as": "policy_hits", "args": {"query": "industrial engineering track required courses and electives", "limit": 5}},
        requires="retriever",
    ),
    ToolSpec(
        name="interpret",
        purpose="Read ONE typed value out of ONE passage, with a citation.",
        when=(
            "After `search_corpus`, on the passage you judged relevant. State the kind you expect "
            "(quantity/identifier/text/bool/date). The value must appear in the passage: extract, "
            "never calculate. Arithmetic belongs to `compute`, where it can be audited."
        ),
        example={
            "tool": "interpret",
            "as": "required_credits",
            "args": {"slug": "track-ise", "question": "how many credits does the degree require", "expect": "quantity"},
        },
        requires="extractor",
    ),
    ToolSpec(
        name="extract_list",
        purpose="Read the SET of values ONE passage lists, as a collection, with a citation.",
        when=(
            "The plural of `interpret`: when a passage ENUMERATES things you need as data -- the "
            "course codes a wiki section lists as electives, the members of a group -- and you "
            "want them as a collection to classify against, not one value. Returns records each "
            "holding one `value` of the kind you state; every value is verified to appear in the "
            "passage, so a code the model invents is dropped. Then a `find`/`select` with "
            '`in {"fact": "<this>", "field": "value"}` labels a set of records by membership -- '
            "e.g. a remaining course is an ELECTIVE iff its number is in the elective-code set you "
            "extracted from the track's Faculty-Electives section. One call, not one per course."
        ),
        example={
            "tool": "extract_list",
            "as": "elective_codes",
            "args": {"slug": "track-ise", "question": "every faculty-elective course code listed", "expect": "identifier"},
        },
        requires="extractor",
    ),
    ToolSpec(
        name="compute",
        purpose="Derive new facts from facts you already hold.",
        when=(
            "Filtering, counting, totalling, comparing, joining, set differences -- anything that "
            "reads facts and produces a new one. Prefer ONE call carrying several named pipelines "
            "over several calls: pipelines may reference each other's results by name, and a "
            "failure in one leaves the others intact. To pull a single value out of a one-record "
            "collection -- an id to look something else up by -- use aggregate `only` with the "
            "field name; it refuses if the collection does not hold exactly one record.\n"
            "     A pipeline's `source` is a FACT YOU HOLD or another pipeline in the same call -- "
            "never a data source. Reading storage is `find`'s job. A `find` in the SAME reply can "
            "then use a key this call computed, because calls run in order.\n"
            "     To compute ONE value straight FROM held scalars -- a GPA from points and credits, "
            "an average, a threshold -- a pipeline drops `source`/`stages` and gives a `value` "
            'expression: {"name": "gpa", "value": {"div": [{"fact": "points"}, {"fact": "credits"}]}}. '
            "The expression is the same one an `extend` field takes (`{\"fact\":..}` for a held "
            "scalar, `{\"value\":..}` for a literal, and `add`/`sub`/`mul`/`div`), only without "
            "`{\"path\":..}` -- there are no rows to read. No carrier collection, no `aggregate only`."
        ),
        example={
            "tool": "compute",
            "args": {
                "pipelines": [
                    {
                        "name": "earned",
                        "source": "transcript",
                        "stages": [{"op": "aggregate", "agg": "sum", "field": "creditsEarned"}],
                    },
                    {
                        "name": "still_needed",
                        "source": "required",
                        "stages": [{"op": "arith", "fn": "sub", "other": "earned"}],
                    },
                ]
            },
        },
    ),
    ToolSpec(
        name="traverse",
        purpose="Follow a relationship as far as it goes.",
        when=(
            "When the answer depends on a chain of unknown length -- prerequisites of "
            "prerequisites. `compute` cannot do this: a pipeline of N joins reaches exactly N "
            "levels, and the chain's depth is a property of the data. `edges` names a collection "
            "of edge records you FETCHED FIRST with `find` -- see the edge source in the data "
            "sources below. Edges sharing a `group` are ALTERNATIVES (any one satisfies the "
            "requirement); different groups are each required. Carry `group` through if that "
            "distinction matters, or you will report a choice as several obligations."
        ),
        example={
            "tool": "traverse",
            "as": "prereq_chain",
            "args": {
                "edges": "prereq_edges",
                "start": "00970800",
                "from": "course",
                "to": "requires",
                "carry": ["group"],
            },
        },
        needs_source="edges",
    ),
    ToolSpec(
        name="forecast",
        purpose="Project a pattern forward from a complete history.",
        when=(
            "For questions about a period that has not happened -- will this run next spring. "
            "`observations` names a collection you have already fetched, one record per past "
            "occurrence, with a field naming the period. Needs the WHOLE history: a projection "
            "from a partial one gets reported as though it were the whole record.\n"
            "     ALWAYS give `cycle_path` -- the field naming the repeating unit the period sits "
            "inside, `academicYear` for a semester. Without it the rate is the share of all "
            "offerings that fell in the target period, which is a different question and inverts "
            "the answer for any course offered in more than one term: a course that ran EVERY "
            "spring for three years, but also in winter and summer, scores 0.43 and comes back "
            "'will not run'."
        ),
        example={
            "tool": "forecast",
            "as": "spring_forecast",
            "args": {
                "observations": "past_offerings",
                "period_path": "semesterName",
                "cycle_path": "academicYear",
                "target": "spring",
            },
        },
    ),
    ToolSpec(
        name="optimize",
        purpose="Search for an assignment of items to slots that satisfies constraints.",
        when=(
            "For building a plan rather than describing one. `items` and `slots` both name "
            "collections you have already fetched or computed -- the courses to place, and the "
            "semesters to place them in. `item_id` and `slot_id` name the field that identifies "
            "each one, and both must be UNIQUE within their collection (a course offered in two "
            "semesters must appear once in items, not twice). Constraints are precedence (A "
            "before B), capacity (a slot's total), and eligibility (which slots an item may use). "
            "The result is a PROPOSAL about the future, marked simulated.\n"
            "     Pick the OBJECTIVE by the question: `minimize_slots` schedules EVERY item across "
            "as many semesters as it takes (\"how many semesters to finish\") and fails if they "
            "cannot all be placed; `fill` fills a FIXED, small set of slots to capacity and leaves "
            "the overflow \"(unscheduled)\" (\"what goes in my NEXT two semesters\"). For a "
            "next-N-semesters plan use `fill`, order `items` by priority (mandatory before "
            "elective) since it places greedily, and read the plan as the placed rows plus the "
            "(unscheduled) remainder."
        ),
        example={
            "tool": "optimize",
            "as": "plan",
            "args": {
                "items": "courses_to_place",
                "item_id": "courseNumber",
                "slots": "my_semesters",
                "slot_id": "semesterCode",
                "slot_index": "order",
                "constraints": [{"kind": "capacity", "attribute": "credits", "limit": 20}],
                "objective": "minimize_slots",
            },
        },
        needs_source="slots",
    ),
    ToolSpec(
        name="propose",
        purpose="Describe a change for a person to approve. Nothing happens until they do.",
        when=(
            "The only tool that can alter anything, and it alters nothing by itself. Name the "
            "facts that justify it -- whoever confirms has to be able to see why."
        ),
        example={
            "tool": "propose",
            "as": "registration",
            "args": {
                "action": "register",
                "target": "00960211",
                "payload": {"semester": "spring-2026"},
                "grounds": ["prereqs_met"],
            },
        },
    ),
)


COMPOSITES: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="plan_term",
        purpose="Build a conflict-free, exam-safe schedule for one or more terms from candidate courses.",
        when=(
            "Once you HOLD the eligible candidate courses -- the remaining courses to consider, each "
            "tagged mandatory or elective -- call this to seat them into the actual timetable of a "
            "term. Unlike `optimize`, which places items into abstract slots, this builds the REAL "
            "weekly schedule using the university's own planner: it keeps only courses offered that "
            "term, assigns non-conflicting lecture/tutorial/lab groups, checks exam dates, respects "
            "the credit cap and the no-additional-credit rule, and FLAGS an unmet prerequisite "
            "rather than guessing. It returns the courses actually PLACED -- each carrying its "
            "credits -- plus the ones that did not fit and why. The result is SIMULATED, a proposal "
            "about a term that has not happened.\n"
            "     This is the tool for 'plan my next semester': it removes the fragile "
            "offerings -> place -> split-by-term hand-wiring by doing it in one deterministic call. "
            "Give `terms` by NAME -- \"winter\", \"spring\", \"summer\" (a year-coded \"2025-1\" is "
            "also accepted) -- and `candidates` as course numbers each with a category, \"mandatory\" "
            "or \"elective\". Each placed course comes back tagged with the `term` you asked for, so "
            "you split the plan on that same string. Read back the placed courses and their `credits` "
            "to derive anything per-course -- a minimum grade to hold a GPA, say -- with `compute`, "
            "exactly as you would off any other collection."
        ),
        example={
            "tool": "plan_term",
            "as": "winter_plan",
            "args": {
                "terms": ["winter"],
                "candidates": [{"courseNumber": "00940412", "category": "mandatory"}],
                "max_credits": 20,
            },
        },
        requires="settings",
    ),
)
"""Domain COMPOSITES -- deliberately NOT primitives.

`plan_term` wraps a whole sequence (offerings filter, conflict-free placement,
exam check, per-term split) that the general primitives can express but that the
model kept mis-wiring on the last mile. It is quarantined here, apart from
`PRIMITIVES`, precisely because it is the composite the primitive set exists to
avoid: keeping the two lists separate keeps the "nine primitives" invariant
honest while still advertising the shortcut where it earns its place."""


_CATALOG: tuple[ToolSpec, ...] = PRIMITIVES + COMPOSITES


def available_tools(context: Any = None) -> tuple[ToolSpec, ...]:
    """The tools whose dependencies are actually wired.

    Advertising a tool the wiring cannot serve is worse than omitting it: the
    model reaches for it on exactly the question it fits, fails, and spends turns
    discovering that the capability it was promised does not exist. Measured on
    the live context: 8 tools documented, 3 usable.
    """
    if context is None:
        return _CATALOG
    obtainable = getattr(context, "obtainable", frozenset()) or frozenset()
    return tuple(
        spec for spec in _CATALOG
        if (not spec.requires or getattr(context, spec.requires, None) is not None)
        and (not spec.needs_source or spec.needs_source in obtainable)
    )


def render_catalog(context: Any = None) -> str:
    """The tool section of the system prompt, limited to what is wired."""
    blocks = [_render_tool(spec) for spec in available_tools(context)]
    return "\n\n".join([*blocks, _render_operators(), _render_sugar()])


def _render_tool(spec: ToolSpec) -> str:
    import json

    return (
        f"## {spec.name}\n"
        f"{spec.purpose}\n"
        f"WHEN: {spec.when}\n"
        f"EXAMPLE: {json.dumps(spec.example, ensure_ascii=False)}"
    )


def _render_operators() -> str:
    """The `compute` operator table, rendered from the table itself.

    Hand-maintaining this list is how a prompt ends up describing an operator
    that no longer exists, or omitting one that does.
    """
    lines = []
    for name, spec in sorted(OPERATORS.items()):
        lines.append(f"  {name:12} {spec.summary}")
        if spec.usage:
            # Naming an operator without showing its arguments left the model to
            # discover every one by rejection -- a turn each, fourteen times.
            lines.append(f"  {'':12} {spec.usage}")
    return "## compute operators (stages of a pipeline)\n" + "\n".join(lines)


def _render_sugar() -> str:
    """Sugar, with the expansion, so a rejection points at the fix.

    These are NOT stage operators. Naming them here -- rather than letting the
    model discover them by having one rejected -- costs a few lines and saves a
    turn each time.
    """
    lines = [f"  {name:12} not a stage operator; write its expansion: {how}" for name, how in sorted(SUGAR.items())]
    return "## shorthands you may be tempted to write\n" + "\n".join(lines)


def tool_names() -> frozenset[str]:
    return frozenset(spec.name for spec in _CATALOG)


__all__ = ["COMPOSITES", "PRIMITIVES", "ToolSpec", "available_tools", "render_catalog", "tool_names"]
