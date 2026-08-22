"""Dispatch -- phase 9d of docs/agent/tools_implementation_plan.md.

Routes a parsed tool call to one of the eight primitives and admits what comes
back as a named fact.

The thing that makes the layer compose is that tool arguments name FACTS, not
data. `compute` reads pipelines whose sources are held facts, `traverse` walks a
held edge collection, `optimize` places held items into held slots. So the
working set is the only channel between tools, and no tool ever receives a
payload the model typed out -- which is what stops a model hand-copying a
transcript into an argument and reshaping it on the way.

Every tool call names its result with `as`, except `compute`, whose pipelines
name themselves.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Union

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.catalog import tool_names
from app.agent_core.facts.codec import ParseError, parse_pipelines, parse_predicate
from app.agent_core.facts.find import SourceSchema, find
from app.agent_core.facts.forecast import forecast
from app.agent_core.facts.operators import DataDefect, Defect, ExpressionDefect
from app.agent_core.facts.optimize import (
    Capacity,
    Eligibility,
    Infeasible,
    Item,
    Objective,
    Precedence,
    Slot,
    optimize,
)
from app.agent_core.facts.predicate import (
    Always,
    And,
    Comparison,
    FactRef,
    Not,
    Op,
    Or,
    Path,
    Predicate,
)
from app.agent_core.facts.propose import Proposal, propose
from app.agent_core.facts.prose import Passage, interpret, interpret_list, search_corpus
from app.agent_core.facts.runner import Blocked, Failed, Succeeded, run_pipelines
from app.agent_core.facts.traverse import traverse
from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind
from app.agent_core.loop.course_names import canonical_course_code
from app.clients.internal_api_client import InternalApiClientError, fetch_term_plan


@dataclass
class DispatchContext:
    """Everything a tool might need, and the working set it reads and writes."""

    facts: dict[str, HeldFact] = field(default_factory=dict)
    schemas: Mapping[str, SourceSchema] = field(default_factory=dict)
    database: Any = None
    settings: Any = None
    """App settings, for the one tool that reaches an `api`-side endpoint over HTTP
    (`plan_term`). Carried here because the internal client keys the base URL and
    service token off it; `None` in tests, which the tool reports as a defect."""
    retriever: Any = None
    extractor: Any = None
    passages: dict[str, Passage] = field(default_factory=dict)
    obtainable: frozenset[str] = frozenset()
    """Input kinds the MODEL can obtain by calling tools -- e.g. "edges", "slots".

    A tool needing one it cannot get is not advertised. Empty by default,
    because the honest default is "this cannot be fed" rather than "assume it
    can": the first version assumed, and advertised two tools the model had no
    route to.
    """

    def collection(self, name: str) -> Union[Collection, ExpressionDefect]:
        held = self.facts.get(name)
        if held is None:
            # The same near-miss repair the ANSWER boundary already does. A live
            # plan run asked for `typed_candidates` while holding `candidates`,
            # got the list of every held name, and spent a turn re-reading it --
            # a list of a dozen names does not point at the one that is a prefix
            # away. Suggests only; nothing is renamed.
            from app.agent_core.facts.answer import _did_you_mean

            return ExpressionDefect(
                0,
                f"no fact named '{name}'; held: {sorted(self.facts)}"
                + _did_you_mean((name,), self.facts),
            )
        if not isinstance(held.value, Collection):
            return ExpressionDefect(0, f"'{name}' is a scalar, but a collection is needed here")
        return held.value


@dataclass(frozen=True)
class Dispatched:
    """What a call produced: named facts, plus per-name defects.

    Both, not either. `compute` can succeed on four pipelines and fail on a
    fifth, and discarding the four would make the loop redo work that was
    already right.
    """

    facts: dict[str, HeldFact] = field(default_factory=dict)
    defects: dict[str, Defect] = field(default_factory=dict)
    proposal: Proposal | None = None


async def dispatch(call: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    tool = call.get("tool")
    if tool not in tool_names():
        return _defect(
            "call",
            ExpressionDefect(0, f"unknown tool {tool!r}; available: {sorted(tool_names())}"),
        )

    args = call.get("args") or {}
    name = call.get("as")

    if tool != "compute" and not name:
        return _defect(
            "call",
            ExpressionDefect(
                0,
                f"'{tool}' needs an 'as' naming its result, so later calls can refer to it. "
                "(Only 'compute' is exempt -- its pipelines name themselves.)",
            ),
        )

    unsubstituted = _slot_written_into_an_argument(args, context)
    if unsubstituted:
        slot, value = unsubstituted
        return _defect(
            name or "call",
            ExpressionDefect(
                0,
                f"'{{{slot}}}' was written into a tool argument as text. Slots are filled in "
                f"ANSWERS, never in arguments -- this reached the tool literally. To use the "
                f'value, either write {{"fact": "{slot}"}} where a predicate takes a value, or '
                f"type the value itself: {value!r}.",
            ),
        )

    try:
        handler = _HANDLERS[tool]
        return await handler(name, args, context)
    except ParseError as error:
        return _defect(name or "call", ExpressionDefect(0, str(error)))
    except Exception as error:  # noqa: BLE001 -- see below; this must not be narrowed
        # Defense in depth. A malformed tool request must come back as a DEFECT
        # the loop can report and the model can repair -- never as an exception
        # that ends the run.
        #
        # Live run, case 1: the model wrote `"start": {"fact": "course_number"}`,
        # generalising the `{"fact": ...}` idiom that works in every predicate
        # value. `traverse` did `seen = {start}` on a dict, and the resulting
        # `TypeError: unhashable type: 'dict'` unwound through the loop and
        # aborted the whole eval on question two of ten. One bad argument shape
        # cost nine questions.
        #
        # The specific shape is now accepted (`_literal` below), but the blanket
        # guard stays: the next unhandled shape must cost one defect, not a run.
        return _defect(
            name or "call",
            ExpressionDefect(
                0,
                f"'{tool}' could not run with those arguments -- "
                f"{type(error).__name__}: {error}. Check the argument shapes against the "
                "tool's example.",
            ),
        )


def _defect(name: str, defect: Defect) -> Dispatched:
    return Dispatched(defects={name: defect})


def _fact(
    name: str, value: Any, basis: Basis, citation: Any = None, derivation: str | None = None
) -> Dispatched:
    return Dispatched(
        facts={name: HeldFact(value=value, basis=basis, citation=citation, derivation=derivation)}
    )


def _resolve_fact_refs(
    predicate: Predicate, facts: Mapping[str, HeldFact]
) -> Union[Predicate, ExpressionDefect]:
    """Replace every `FactRef` with the value of the fact it names.

    Done here rather than in the grammar because the working set is only in
    scope at dispatch. An unresolvable reference is refused loudly: filtering by
    a fact that does not exist would otherwise match nothing and return a
    legitimate-looking empty result, which is the failure mode that cost several
    live turns.
    """
    if isinstance(predicate, Always):
        return predicate
    if isinstance(predicate, Not):
        inner = _resolve_fact_refs(predicate.term, facts)
        return inner if isinstance(inner, ExpressionDefect) else Not(inner)
    if isinstance(predicate, (And, Or)):
        terms = []
        for term in predicate.terms:
            resolved = _resolve_fact_refs(term, facts)
            if isinstance(resolved, ExpressionDefect):
                return resolved
            terms.append(resolved)
        return type(predicate)(tuple(terms))

    if not isinstance(predicate.value, FactRef):
        return predicate

    ref = predicate.value
    held = facts.get(ref.name)
    if held is None:
        return ExpressionDefect(
            0,
            f"the filter refers to fact '{ref.name}', which is not held. "
            f"Available: {sorted(facts)}.",
        )

    if ref.field is not None:
        # A field drawn from a held collection: the SET (with `in`, a semi-join)
        # or the SINGLE value (with `=`, one-record extraction). Both because the
        # model reaches for both.
        return _resolve_field_ref(predicate, ref, held)

    if not isinstance(held.value, Scalar):
        return ExpressionDefect(
            0,
            f"fact '{ref.name}' is a collection; a filter value must be a single value. To match "
            f'against a field of it, add "field": {{...}} for a set membership test, or aggregate '
            "it to one value first.",
        )
    return Comparison(predicate.path, predicate.op, held.value)


def _resolve_field_ref(
    predicate: Comparison, ref: FactRef, held: HeldFact
) -> Union[Comparison, ExpressionDefect]:
    """`FactRef(name, field)` -> the field's value(s) from a held collection.

    Two readings, chosen by the operator, because the model reaches for BOTH:
      - `in`  -> the SET of that field's values across the collection (semi-join)
      - `=`, `<`, ... -> the SINGLE value, when the collection holds exactly one
        record. This is the same one-record extraction `only` does, and a live
        run hit the wall without it: the model held one course in `next_course`
        and wrote `course = {"fact": "next_course", "field": "courseNumber"}`
        turn after turn, which is exactly right and used to be refused.
    """
    if not isinstance(held.value, Collection):
        return ExpressionDefect(
            0, f"fact '{ref.name}' is a single value, so it has no field '{ref.field}' to draw from."
        )

    path = Path.parse(ref.field)
    values: list[Scalar] = []
    seen: set = set()
    for record in held.value.records:
        resolved = path.resolve(record)
        if not isinstance(resolved, Scalar):
            return ExpressionDefect(
                0,
                f"'{ref.field}' is missing on a record of '{ref.name}', so the value would be "
                "silently omitted. Every record must carry the field a fact-reference draws on.",
            )
        if resolved.value not in seen:
            seen.add(resolved.value)
            values.append(resolved)

    if predicate.op is Op.IN:
        return Comparison(predicate.path, predicate.op, tuple(values))

    # A scalar operator wants ONE value.
    if len(values) != 1:
        return ExpressionDefect(
            0,
            f"'{ref.name}.{ref.field}' has {len(values)} distinct values, but '{predicate.op.value}' "
            "compares against one. Use 'in' to match the whole set, or filter the fact to a single "
            "record first.",
        )
    return Comparison(predicate.path, predicate.op, values[0])


async def _find(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    source = args.get("source")
    schema = context.schemas.get(source)
    if schema is None:
        return _defect(name, ExpressionDefect(0, f"unknown source {source!r}; available: {sorted(context.schemas)}"))

    predicate = parse_predicate(args["predicate"]) if args.get("predicate") else None
    if predicate is not None:
        predicate = _resolve_fact_refs(predicate, context.facts)
        if isinstance(predicate, ExpressionDefect):
            return _defect(name, predicate)
    result = await find(context.database, schema, predicate=predicate, limit=args.get("limit", 200))
    if isinstance(result, (ExpressionDefect, DataDefect)):
        return _defect(name, result)
    filtered = " matching a filter" if predicate is not None else ""
    return _fact(name, result, schema.basis, derivation=f"read from {schema.collection}{filtered}")


async def _search_corpus(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    if context.retriever is None:
        return _defect(name, ExpressionDefect(0, "no corpus is configured, so prose cannot be searched here"))
    hits = await search_corpus(context.retriever, args.get("query", ""), limit=args.get("limit", 5))
    # Remember the passages so `interpret`/`extract_list` can be handed one by
    # slug rather than having the model retype its text -- retyped prose is prose
    # that can drift.
    #
    # ACCUMULATE per slug, do not overwrite. The wiki is heading-segmented, so one
    # page comes back as several chunks that all share its slug; keyed by slug
    # alone, the last chunk clobbered the rest, and `extract_list("track-ise")`
    # then saw only the final section. A live plan lost the whole Faculty-Elective
    # course list exactly this way -- the electives chunk was overwritten by a
    # later one, so the set came back with a single code. Concatenating every
    # retrieved chunk of a page (capped) means the slug names the page's content,
    # which is what the model reasonably expects.
    page_of = getattr(context.retriever, "page", None)
    for record in hits.records:
        slug = record.fields["slug"].value
        # Prefer the WHOLE page when the retriever can give it: accumulating
        # retrieved chunks still only ever holds the sections that made top-k, so
        # extract_list can miss the very list it needs (a live plan got 2 of ~40
        # required codes, and the duplicate-course refusal that followed traced
        # straight back to that thin set). The full page has every section, so a
        # single hit on it is enough. Fall back to chunk accumulation for any
        # retriever that cannot serve a full page.
        full = page_of(slug) if page_of is not None else None
        if full:
            excerpt = full[:_PASSAGE_STASH_CAP]
        else:
            excerpt = record.fields["excerpt"].value
            prior = context.passages.get(slug)
            if prior is not None and excerpt not in prior.excerpt:
                excerpt = f"{prior.excerpt}\n\n{excerpt}"[:_PASSAGE_STASH_CAP]
        context.passages[slug] = Passage(
            slug=slug,
            title=record.fields["title"].value,
            excerpt=excerpt,
            score=record.fields["score"].value,
        )
    return _fact(name, hits, Basis.WIKI_DERIVED)


_PASSAGE_STASH_CAP = 20000
"""How much page text the stash keeps for a later `interpret`/`extract_list`.

Sized to hold a WHOLE track page (the ISE page is ~15.9k chars) so the full-page
stash is not silently truncated mid-electives -- the section sits past the
halfway point, so the old 8k cap would have dropped it even when the full page
was available. Still bounded, so a pathologically large page cannot make an
unbounded extractor prompt."""


async def _interpret(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    if context.extractor is None:
        return _defect(name, ExpressionDefect(0, "no interpreter is configured, so prose cannot be read here"))
    slug = _literal(args, "slug", context)
    if isinstance(slug, ExpressionDefect):
        return _defect(name, slug)
    passage = context.passages.get(slug)
    if passage is None:
        return _defect(
            name,
            ExpressionDefect(0, f"no retrieved passage '{slug}'; search first. Retrieved: {sorted(context.passages)}"),
        )

    expect = args.get("expect", "text")
    kinds = _INTERPRETABLE_KINDS
    if expect == ScalarKind.BOOL.value:
        return _defect(name, ExpressionDefect(0, _BOOL_IS_NOT_EXTRACTABLE))
    if expect not in kinds:
        return _defect(name, ExpressionDefect(0, f"unknown expect {expect!r}; available: {sorted(kinds)}"))

    result = await interpret(context.extractor, passage, args.get("question", ""), expect=kinds[expect])
    if isinstance(result, DataDefect):
        return _defect(name, result)
    value = result.value
    if kinds[expect] is ScalarKind.IDENTIFIER and isinstance(value, Scalar):
        # Same wiki-code fix as extract_list: a single course code read from prose
        # is a digit short and would not join to the catalog. Restore the zero.
        value = Scalar(value.kind, canonical_course_code(str(value.value)))
    return _fact(name, value, result.basis, citation=result.citation)


async def _extract_list(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    """The plural of `interpret`: the SET of values a retrieved passage lists.

    Same shape as `_interpret` -- a passage is named by the slug `search_corpus`
    already stashed -- but the result is a COLLECTION (records with one `value`
    field) so the model can `select ... in {this, field: "value"}` to classify a
    set of records against what the wiki enumerates, instead of one model call
    per candidate.
    """
    if context.extractor is None:
        return _defect(name, ExpressionDefect(0, "no interpreter is configured, so prose cannot be read here"))
    slug = _literal(args, "slug", context)
    if isinstance(slug, ExpressionDefect):
        return _defect(name, slug)
    passage = context.passages.get(slug)
    if passage is None:
        return _defect(
            name,
            ExpressionDefect(0, f"no retrieved passage '{slug}'; search first. Retrieved: {sorted(context.passages)}"),
        )

    expect = args.get("expect", "identifier")
    kinds = _INTERPRETABLE_KINDS
    if expect == ScalarKind.BOOL.value:
        return _defect(name, ExpressionDefect(0, _BOOL_IS_NOT_EXTRACTABLE))
    if expect not in kinds:
        return _defect(name, ExpressionDefect(0, f"unknown expect {expect!r}; available: {sorted(kinds)}"))

    result = await interpret_list(context.extractor, passage, args.get("question", ""), expect=kinds[expect])
    if isinstance(result, DataDefect):
        return _defect(name, result)
    value = result.value
    if kinds[expect] is ScalarKind.IDENTIFIER:
        # The wiki renders course codes one digit short (`[[00960600-..|0960600]]`),
        # so an extracted set of 7-digit labels silently joins to nothing against
        # the catalog's 8-digit courseNumber. Restore the dropped leading zero.
        value = _canonicalise_codes(value)
    return _fact(name, value, result.basis, citation=result.citation)


def _canonicalise_codes(collection: Collection) -> Collection:
    """A copy of `collection` with every `value` field's course code canonicalised.

    Immutable, like everything else here: new records, new collection. Only bare
    7-digit runs change (see `canonical_course_code`), so a non-course identifier
    passes through untouched."""
    records = tuple(
        Record(
            fields={
                field_name: (
                    Scalar(field.kind, canonical_course_code(str(field.value)))
                    if field_name == "value" and isinstance(field, Scalar)
                    else field
                )
                for field_name, field in record.fields.items()
            },
            basis=record.basis,
            field_basis=record.field_basis,
        )
        for record in collection.records
    )
    return Collection(records=records, completeness=collection.completeness)


async def _compute(_name: Any, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    if not args.get("pipelines"):
        # A `compute` with no pipelines used to return NOTHING -- no facts and no
        # defect -- so the turn vanished without explanation and the model
        # repeated it verbatim. Live, twice in one run:
        #
        #   compute({"ceil_div": [{"fact": "credits_needed"},
        #                         {"fact": "max_credits_per_semester"}]}) -> 0 facts
        #
        # The expression is correct; it is simply at the top level, where
        # `compute` expects a list of named pipelines. A silent no-op is the
        # worst possible response to that, because nothing distinguishes it from
        # a computation that legitimately produced no rows.
        return _defect(
            _name,
            ExpressionDefect(
                0,
                "`compute` takes a LIST of named pipelines, and this call has none. Wrap the "
                'expression: {"pipelines": [{"name": "semesters", "value": '
                '{"ceil_div": [{"fact": "a"}, {"fact": "b"}]}}]}. A pipeline over a collection '
                'names its `source` instead; one over held scalars needs only `name` and `value`.',
            ),
        )
    pipelines = parse_pipelines(args.get("pipelines", []))

    def resolve_predicates(pipeline: Any, available: Mapping[str, Any]) -> Any:
        """Resolve this pipeline's filters against what exists WHEN IT RUNS.

        Late, not up front, so a sibling defined in the same call can be filtered
        by. `available` holds raw values while `_resolve_fact_refs` reads
        `HeldFact`s, so siblings are wrapped -- with the basis they were computed
        under where the working set knows it, and OFFICIAL_RECORD is never
        assumed for something derived.
        """
        facts = dict(context.facts)
        for name, value in available.items():
            if name not in facts:
                facts[name] = HeldFact(value=value, basis=Basis.SIMULATED)
        for stage in pipeline.stages:
            if "predicate" in stage.args:
                resolved = _resolve_fact_refs(stage.args["predicate"], facts)
                if isinstance(resolved, ExpressionDefect):
                    return resolved
                stage.args["predicate"] = resolved
        return None
    # EVERY held fact, scalars included. The runner publishes and consumes
    # scalars deliberately -- that is how "spring total vs autumn total" works --
    # and filtering them out here made a computed value unreferenceable by the
    # very next pipeline. A live run held both operands of a subtraction and
    # could not express it.
    env = {name: held.value for name, held in context.facts.items()}

    outcomes = run_pipelines(pipelines, env, resolve_predicates)

    produced: dict[str, HeldFact] = {}
    defects: dict[str, Defect] = {}
    for name, outcome in outcomes.items():
        if isinstance(outcome, Succeeded):
            pipeline = next((p for p in pipelines if p.name == name), None)
            produced[name] = HeldFact(
                value=outcome.value,
                basis=outcome.basis,
                derivation=_describe_pipeline(pipeline) if pipeline else None,
            )
        elif isinstance(outcome, Failed):
            defects[name] = outcome.defect
        elif isinstance(outcome, Blocked):
            defects[name] = ExpressionDefect(0, f"not run: it depends on '{outcome.waiting_on}', which failed")
    return Dispatched(facts=produced, defects=defects)


def _describe_pipeline(pipeline: Any) -> str:
    """A one-line account of how a pipeline's result was produced.

    Deliberately mechanical -- the stage names and the fields they touched --
    because the point is to show what the value IS, independently of what the
    model chose to call it.
    """
    steps = []
    for stage in pipeline.stages:
        detail = stage.args.get("path") or stage.args.get("on") or stage.args.get("other")
        function = stage.args.get("op")
        label = stage.op
        if function is not None and stage.op in ("aggregate", "arith", "compare"):
            label = f"{stage.op}:{getattr(function, 'value', function)}"
        steps.append(f"{label}({getattr(detail, 'dotted', detail)})" if detail else label)
    return f"{pipeline.source} -> " + " -> ".join(steps) if steps else pipeline.source


def _literal(
    args: Mapping[str, Any], key: str, context: DispatchContext, default: str = ""
) -> Union[str, ExpressionDefect]:
    """A model-supplied scalar argument, accepting the `{"fact": name}` idiom.

    The model generalised that idiom here from predicate values, where it is
    required, and it was right to: `traverse`'s start course is exactly the kind
    of value that should come from a fact rather than be retyped. Retyping it is
    how a computed identifier gets laundered into a literal.

    So the instinct is honoured rather than corrected -- but a name that does not
    resolve is refused loudly, because a `start` that silently became the string
    "{'fact': 'x'}" would traverse from a node that does not exist and return an
    empty result that looks like "nothing is required".
    """
    value = args.get(key, default)
    if not isinstance(value, Mapping):
        return str(value)

    if "fact" not in value:
        return ExpressionDefect(
            0, f"'{key}' must be a value or {{\"fact\": \"name\"}}; got an object with {sorted(value)}"
        )
    held = context.facts.get(value["fact"])
    if held is None:
        # `sorted(facts)` -- an undefined name, on the one line that runs when
        # the model gets an argument wrong. Instead of "not held; available:
        # [...]" it raised NameError, the catch-all turned that into "could not
        # run with those arguments", and the model had nothing to repair from.
        # Measured on `interpret`: the runs that hit it took 10-11 steps against
        # a usual 4, and one gave up entirely.
        from app.agent_core.facts.answer import _did_you_mean

        return ExpressionDefect(
            0,
            f"'{key}' refers to fact '{value['fact']}', which is not held. "
            f"Available: {sorted(context.facts)}."
            + _did_you_mean((str(value["fact"]),), context.facts),
        )
    if not isinstance(held.value, Scalar):
        return ExpressionDefect(
            0, f"'{key}' refers to '{value['fact']}', which is a collection; a single value is needed."
        )
    return str(held.value.value)


async def _traverse(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    edges = context.collection(args.get("edges", ""))
    if isinstance(edges, ExpressionDefect):
        return _defect(name, edges)
    start = _literal(args, "start", context)
    if isinstance(start, ExpressionDefect):
        return _defect(name, start)
    reached = traverse(
        edges,
        start=start,
        from_path=Path.parse(args.get("from", "from")),
        to_path=Path.parse(args.get("to", "to")),
        max_depth=args.get("max_depth", 10),
        carry=tuple(args.get("carry", ())),
    )
    basis = min((r.basis for r in reached.records), key=lambda b: b.strength, default=Basis.OFFICIAL_RECORD)
    return _fact(name, reached, basis)


async def _forecast(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    observations = context.collection(args.get("observations", ""))
    if isinstance(observations, ExpressionDefect):
        return _defect(name, observations)
    target = _literal(args, "target", context)
    if isinstance(target, ExpressionDefect):
        return _defect(name, target)
    cycle = args.get("cycle_path")
    result = forecast(
        observations,
        period_path=Path.parse(args.get("period_path", "period")),
        target=target,
        cycle_path=Path.parse(cycle) if cycle else None,
    )
    if isinstance(result, DataDefect):
        return _defect(name, result)
    return _fact(name, result.value, result.basis)


async def _optimize(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    items_source = context.collection(args.get("items", ""))
    if isinstance(items_source, ExpressionDefect):
        return _defect(name, items_source)
    slots_source = context.collection(args.get("slots", ""))
    if isinstance(slots_source, ExpressionDefect):
        return _defect(name, slots_source)

    # Identity is NAMED, never guessed. The first version scanned for the first
    # field called id/item/slot/courseNumber/node and fell back to "whatever
    # scalar came first" -- which, on real slots produced by unnesting a plan's
    # semesters, handed every slot the parent plan's `_id`. Slots sharing an id
    # silently pool their capacity, so the solver would return a plan that
    # violates the very limit it was given, and nothing would say so.
    # ITEMS are de-duplicated by their id, keeping the first occurrence. A course
    # offered in two semesters is still ONE course to place, and the natural way
    # to build items -- join the remaining courses to their offerings -- yields
    # one row per offering, so the same course appears twice. Rejecting that sent
    # the model into a dedup fight it kept losing on the last mile of a plan.
    # SLOTS stay strict: two slots sharing an id silently pool capacity, which is
    # the bug the uniqueness check was added for, so that one is not relaxed.
    item_records = _dedup_by(items_source.records, args.get("item_id"))
    if isinstance(item_records, ExpressionDefect):
        return _defect(name, item_records)
    item_ids = _identify(item_records, args.get("item_id"), "item_id", "items")
    if isinstance(item_ids, ExpressionDefect):
        return _defect(name, item_ids)
    slot_ids = _identify(slots_source.records, args.get("slot_id"), "slot_id", "slots")
    if isinstance(slot_ids, ExpressionDefect):
        return _defect(name, slot_ids)

    items = tuple(
        Item(id=identity, attributes={k: v for k, v in record.fields.items() if isinstance(v, Scalar)})
        for identity, record in zip(item_ids, item_records)
    )

    index_path = args.get("slot_index")
    slots = tuple(
        Slot(
            id=identity,
            # Position is the honest default: `find` sorts by key and `unnest`
            # preserves array order, so the sequence is already deterministic.
            index=int(_quantity(record, index_path, default=position)) if index_path else position,
            attributes={k: v for k, v in record.fields.items() if isinstance(v, Scalar)},
        )
        for position, (identity, record) in enumerate(zip(slot_ids, slots_source.records))
    )

    constraints = []
    for raw in args.get("constraints", ()):
        kind = raw.get("kind")
        if kind == "precedence":
            constraints.append(Precedence(before=raw["before"], after=raw["after"]))
        elif kind == "capacity":
            constraints.append(Capacity(attribute=raw["attribute"], limit=float(raw["limit"])))
        elif kind == "eligibility":
            constraints.append(Eligibility(item=raw["item"], slot_predicate=parse_predicate(raw["slot"])))
        else:
            return _defect(
                name,
                ExpressionDefect(0, f"unknown constraint kind {kind!r}; available: precedence, capacity, eligibility"),
            )

    objectives = {o.value: o for o in Objective}
    objective = objectives.get(args.get("objective", "minimize_slots"))
    if objective is None:
        return _defect(name, ExpressionDefect(0, f"unknown objective; available: {sorted(objectives)}"))

    result = optimize(items=items, slots=slots, constraints=tuple(constraints), objective=objective)
    if isinstance(result, Infeasible):
        return _defect(name, DataDefect(0, result.reason))
    return _fact(name, result.assignment, Basis.SIMULATED)


async def _propose(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    grounds = tuple(args.get("grounds", ()))
    if not grounds:
        # Checked HERE as well as in `propose`, because the basis below is
        # `min()` over the grounds and arguments evaluate before the call --
        # so an empty `grounds` raised "min() iterable argument is empty" and
        # that, not the written explanation, is what reached the model. A defect
        # message is the model's only route to recovering from its own mistake,
        # and Python internals tell it nothing it can act on.
        return _defect(
            name,
            ExpressionDefect(
                0,
                f"cannot propose to {args.get('action')} {args.get('target')} with no grounds: "
                "name the facts that justify it, or the person confirming has nothing to judge.",
            ),
        )
    missing = [g for g in grounds if g not in context.facts]
    if missing:
        return _defect(
            name,
            ExpressionDefect(0, f"grounds name facts that are not held: {missing}. Held: {sorted(context.facts)}"),
        )

    action = _literal(args, "action", context)
    target = _literal(args, "target", context)
    for resolved in (action, target):
        if isinstance(resolved, ExpressionDefect):
            return _defect(name, resolved)

    try:
        proposal = propose(
            action=action,
            target=target,
            payload={k: Scalar(ScalarKind.IDENTIFIER, str(v)) for k, v in (args.get("payload") or {}).items()},
            grounds=grounds,
            # A proposal is only as sound as the weakest thing behind it.
            basis=min((context.facts[g].basis for g in grounds), key=lambda b: b.strength),
        )
    except ValueError as error:
        return _defect(name, ExpressionDefect(0, str(error)))

    return Dispatched(proposal=proposal)


async def _plan_term(name: str, args: Mapping[str, Any], context: DispatchContext) -> Dispatched:
    """Build a conflict-free term plan by calling the api planner, and admit the
    PLACED courses as one SIMULATED collection the model reads credits back off.

    The heavy lifting -- offerings filter, non-conflicting group assignment, exam
    checks, per-term split -- happens api-side over the university's own planner;
    this only threads the student id (the `me` fact), forwards the call, and
    grounds the result. A failure comes back as a DEFECT, never an exception."""
    user_id = _held_id(context, "me")
    if user_id is None:
        return _defect(
            name,
            ExpressionDefect(0, "no 'me' fact holds the student id, which plan_term needs to reach the plan service"),
        )
    # Arguments BEFORE configuration. A malformed call is malformed whether or
    # not the plan service is reachable, and reporting "not configured" for it
    # sends the model to fix the wrong thing -- it also made two tests of this
    # very check pass for that reason rather than for the shape they asserted.
    terms = _string_list(args.get("terms"))
    if not terms:
        return _defect(name, ExpressionDefect(0, '\'terms\' must be a non-empty list of term names, e.g. ["winter"] or ["winter", "spring"]'))
    repeated = sorted({t for t in terms if terms.count(t) > 1})
    if repeated:
        # A placed course is tagged with the term NAME, which is the whole
        # contract for splitting the plan afterwards. Ask for the same name
        # twice and two separate terms come back indistinguishable: a live
        # "how many semesters" run asked for ["winter","spring","summer",
        # "winter","spring","summer"], then selected term == "winter" and
        # merged both winters into one 23-credit term against an 18 cap. The
        # planner had capped each term correctly; the labels destroyed it.
        return _defect(
            name,
            ExpressionDefect(
                0,
                f"'terms' repeats {', '.join(repr(t) for t in repeated)}. Each placed course is "
                "tagged with the term NAME, so two terms sharing one name come back "
                "indistinguishable and any split on that name silently merges them. Use distinct "
                'names -- year-coded ones work: ["2026-1", "2026-2"].',
            ),
        )
    if context.settings is None:
        return _defect(name, DataDefect(0, "the plan service is not configured for this run, so plan_term cannot run"))

    candidates = _plan_candidates(args.get("candidates"), context)
    if isinstance(candidates, ExpressionDefect):
        return _defect(name, candidates)
    if not candidates:
        return _defect(
            name,
            ExpressionDefect(0, "'candidates' must name a held collection of courses, or be a non-empty list of {courseNumber, category}"),
        )

    # `credit_target` bounds the candidate set to what the degree still needs:
    # every mandatory course, then electives until the target. Optional, and
    # skipped when the credits are unknown -- bounding a set whose credits are
    # all zero would keep the mandatory courses alone and silently shorten the
    # degree.
    target = _resolve_number(args.get("credit_target"), context)
    if isinstance(target, ExpressionDefect):
        return _defect(name, target)
    if target is not None and any(c.get("_credits") for c in candidates):
        candidates = _bounded_by_credits(candidates, target)

    try:
        result = await fetch_term_plan(
            user_id=user_id,
            semester_codes=terms,
            candidates=_strip_internal(candidates),
            max_credits=_optional_number(args.get("max_credits")),
            settings=context.settings,
        )
    except InternalApiClientError as error:
        return _defect(name, DataDefect(0, f"the plan service could not build the plan: {error.detail}"))

    # TWO facts: the placed courses, and the per-term totals. The second is
    # returned because the model was spending a whole turn rebuilding it --
    # `distinct on term`, then `select term == "winter"`, then sum credits --
    # from rows the planner had just grouped. Measured on three live runs, every
    # one paid one to two turns for it, at ~15s a turn against a 60s ceiling.
    #
    # Named off the caller's own `as`, so a plan called `plan` is summarised as
    # `plan_by_term` and the pairing is guessable rather than something to look
    # up.
    facts = _fact(name, _placed_collection(result), Basis.SIMULATED, derivation=_plan_summary(result))
    summary = _term_totals(result)
    if summary is not None:
        facts.facts[f"{name}_by_term"] = HeldFact(
            value=summary,
            basis=Basis.SIMULATED,
            derivation="one row per term of the plan, with its course count and credit total",
        )
    return facts


_INTERPRETABLE_KINDS = {
    kind.value: kind for kind in ScalarKind if kind is not ScalarKind.BOOL
}
"""What `interpret` and `extract_list` may be asked for.

BOOL is excluded because it cannot pass their own guard. Every extracted value
must APPEAR in the passage it cites -- that is the whole grounding of the prose
side -- and "True" appears in no regulation. The kind was advertised anyway, so
a model asking a page a yes/no question got:

    interpretation of 'regulations-undergraduate' returned True, which does not
    appear in the passage

19 times across the measured runs, the largest single cause of wasted turns
after the wrong-KIND refusal. A tool that offers an option its own validator
must reject is a trap, not a capability."""

_BOOL_IS_NOT_EXTRACTABLE = (
    'expect "bool" is not available here, and would not work if it were: every extracted value '
    "must APPEAR in the passage, and \"True\" never does. A yes/no about a page is a JUDGEMENT, "
    "not an extraction. Pull the governing PHRASE instead -- expect \"text\", e.g. \"a mandatory "
    "course with a failing last grade may be re-registered with no time limit\" -- and say yes or "
    "no in your own words around it. Only NUMBERS have to be slots; a verdict you have read does "
    "not."
)

_SLOT_IN_TEXT = re.compile(r"\{(\w+)\}")


def _slot_written_into_an_argument(
    args: Any, context: DispatchContext
) -> Union[tuple[str, Any], None]:
    """A `{fact_name}` slot typed into a string argument, where nothing fills it.

    Slot substitution belongs to the ANSWER. A tool argument is passed through
    verbatim, so `{"query": "{program_slug} required courses"}` searches the
    corpus for the characters "{program_slug}" and quietly returns whatever that
    tokenises to.

    Measured, live: a plan run did exactly this on its FIRST search. The query
    matched four unrelated tracks, `extract_list` could not find the student's
    own page among them, and the model spent NINE of its sixteen turns
    re-searching -- because nothing told it the query it sent was not the query
    it wrote. 116s became 194s on the same question.

    Only reports a slot naming a fact actually held. Braces are ordinary
    characters and a query may legitimately contain them; a name that resolves
    is what makes this a mistake rather than a string.
    """
    if not context.facts:
        return None
    if isinstance(args, str):
        for slot in _SLOT_IN_TEXT.findall(args):
            held = context.facts.get(slot)
            if held is not None:
                return slot, getattr(held.value, "value", held.value)
        return None
    if isinstance(args, Mapping):
        for value in args.values():
            found = _slot_written_into_an_argument(value, context)
            if found:
                return found
        return None
    if isinstance(args, (list, tuple)):
        for item in args:
            found = _slot_written_into_an_argument(item, context)
            if found:
                return found
    return None


def _held_id(context: DispatchContext, fact_name: str) -> Union[str, None]:
    held = context.facts.get(fact_name)
    if held is None or not isinstance(held.value, Scalar) or held.value.value in (None, ""):
        return None
    return str(held.value.value)


def _string_list(raw: Any) -> list[str]:
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _candidate_list(raw: Any) -> list[dict[str, str]]:
    """Normalise candidates to {courseNumber, category}, accepting a bare code or
    a {courseNumber, category} object. Codes are canonicalised to 8 digits."""
    if not isinstance(raw, (list, tuple)):
        return []
    candidates: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            number, category = item.strip(), "elective"
        elif isinstance(item, Mapping):
            number = str(item.get("courseNumber") or item.get("number") or "").strip()
            category = str(item.get("category") or "elective")
        else:
            continue
        if number:
            candidates.append({"courseNumber": canonical_course_code(number), "category": category})
    return candidates


def _plan_candidates(
    raw: Any, context: DispatchContext
) -> Union[list[dict[str, str]], ExpressionDefect]:
    """Candidates the NORMAL way a tool takes a set: the NAME of a held collection.

    The model builds the typed remaining-courses fact, then passes its name -- a
    bare "name" or {"fact": "name"} -- exactly as `optimize` takes `items`; a
    literal list still works for a hand-written set. Each record contributes its
    courseNumber and, from a `category`/`type` field, its planning priority."""
    fact_name = _fact_name(raw)
    if fact_name is not None:
        collection = context.collection(fact_name)
        if isinstance(collection, ExpressionDefect):
            return collection
        return _candidates_from_records(collection.records)
    return _candidate_list(raw)


def _fact_name(raw: Any) -> Union[str, None]:
    """The fact name in a collection reference: a bare "name" or {"fact": "name"}."""
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    if isinstance(raw, Mapping) and "fact" in raw:
        return str(raw.get("fact") or "").strip() or None
    return None


def _candidates_from_records(records: tuple[Record, ...]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for record in records:
        number = _field_text(record, "courseNumber")
        if number:
            candidate = {
                "courseNumber": canonical_course_code(number),
                "category": _record_category(record),
            }
            # Carried for `credit_target` below and stripped before the plan
            # service sees it -- the service takes a course and a priority.
            credits = record.fields.get("credits")
            if isinstance(credits, Scalar) and isinstance(credits.value, (int, float)):
                candidate["_credits"] = float(credits.value)
            candidates.append(candidate)
    return candidates


def _bounded_by_credits(
    candidates: list[dict[str, Any]], target: float
) -> list[dict[str, Any]]:
    """Every mandatory course, then electives only until `target` is reached.

    The rule the recipe spells out, applied here instead of costing a turn. It
    is pure arithmetic over facts already held -- there is one right answer, and
    the model was spending ~15s of a 45s budget computing it.

    It is also the step most expensive to get wrong. Handing the planner the
    WHOLE unfinished track is what produced "4 semesters" where the truth is 2:
    50.0 credits of courses scheduled against a 25.5-credit requirement, with
    every individual term legal, so no per-term check could see it.

    Mandatory first and never dropped: those must be taken whatever the total
    comes to, so the target bounds the ELECTIVES only. Overshooting on the last
    elective is expected -- courses are indivisible -- which is why the answer
    check tolerates any overshoot that does not change the term count.
    """
    if target <= 0:
        return candidates
    mandatory = [c for c in candidates if c.get("category") == "mandatory"]
    electives = [c for c in candidates if c.get("category") != "mandatory"]

    running = sum(float(c.get("_credits") or 0) for c in mandatory)
    chosen = list(mandatory)
    for elective in electives:
        if running >= target:
            break
        chosen.append(elective)
        running += float(elective.get("_credits") or 0)
    return chosen


def _strip_internal(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in c.items() if not k.startswith("_")} for c in candidates]


def _record_category(record: Record) -> str:
    """A course's planning priority. The model's own label -- `category`, or the
    `type` it extends from the wiki ("required"/"elective") -- maps to the two the
    planner ranks by: a required/mandatory course is seated before an elective."""
    raw = (_field_text(record, "category") or _field_text(record, "type")).lower()
    return "mandatory" if raw in {"mandatory", "required"} else "elective"


def _field_text(record: Record, field: str) -> str:
    value = record.fields.get(field)
    return str(value.value) if isinstance(value, Scalar) and value.value is not None else ""


def _optional_number(raw: Any) -> Union[float, None]:
    return float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None


def _placed_collection(result: Mapping[str, Any]) -> Collection:
    """The placed courses across all terms, one record each, carrying the fields
    the model reads back -- `credits` above all, for a per-course figure."""
    records: list[Record] = []
    for term in result.get("terms") or []:
        term_code = str(term.get("semesterCode") or "")
        for placed in term.get("placedCourses") or []:
            fields: dict[str, Scalar] = {
                "courseNumber": Scalar(ScalarKind.IDENTIFIER, str(placed.get("courseNumber") or "")),
                "credits": Scalar(ScalarKind.QUANTITY, float(placed.get("credits") or 0)),
                "term": Scalar(ScalarKind.IDENTIFIER, term_code),
                "category": Scalar(ScalarKind.TEXT, str(placed.get("category") or "")),
                "prereqStatus": Scalar(
                    ScalarKind.TEXT, _READABLE_STATUS.get(
                        str(placed.get("prereqStatus") or ""), str(placed.get("prereqStatus") or "")
                    )
                ),
                # NO coreqStatus. `_coreq_status` reads `corequisitesText`, which
                # is not a column, so it returns "none" for every course always --
                # and "none" reaching the model reads as "this course has no
                # corequisites" rather than "never checked". It also cost width on
                # a record that `:detail` already refuses for being too wide. Its
                # wording stays in _READABLE_STATUS for the day the field is
                # seeded; see `_coreq_status` in app/planning/term_plan.py.
                # Always present -- a per-course answer PROJECTs it, and a project
                # fails on any row that lacks the field -- so fall back to the
                # number when the offering carried no title.
                "courseTitle": Scalar(
                    ScalarKind.TEXT, str(placed.get("courseTitle") or placed.get("courseNumber") or "")
                ),
            }
            records.append(Record(fields=fields, basis=Basis.SIMULATED))
    return Collection(
        records=tuple(records),
        completeness=Completeness(complete=True, total=len(records)),
    )


_READABLE_STATUS = {
    "satisfied": "met",
    "check_prerequisites": "NOT met -- check before you register",
    "check_corequisites": "co-requisite NOT met -- check before you register",
    "none": "none",
}
"""The planner's status enum, in words a student can act on.

These reach the reader. A `:detail` slot prints every field of a placed course,
so a live plan showed "prereq check_prerequisites" beside a course the student
is not eligible for -- which is the warning arriving in a vocabulary that only
the planner speaks. The whole point of fixing that flag today was to tell
somebody; an enum name tells them nothing.

Mapped HERE, where the planner's result becomes a fact, rather than in the
planner: `prereqStatus` is its internal vocabulary and the term-plan tests
assert on it, while this is the presentation boundary. An unmapped value falls
through unchanged rather than being blanked, so a status added later degrades to
its raw name instead of disappearing.
"""


def _plan_summary(result: Mapping[str, Any]) -> str:
    terms = result.get("terms") or []
    placed = sum(len(term.get("placedCourses") or []) for term in terms)
    unscheduled = len(result.get("unscheduled") or [])
    return (
        f"plan_term placed {placed} course(s) conflict-free across {len(terms)} term(s); "
        f"{unscheduled} did not fit"
    )


def _dedup_by(records: Sequence[Any], path: Any) -> Union[tuple[Any, ...], ExpressionDefect]:
    """Records with a unique value at `path`, keeping the FIRST of each.

    So the same course reached through two offerings collapses to one item. The
    first occurrence wins, which keeps the result deterministic (`find` and
    `unnest` both preserve a stable order).
    """
    if not path:
        return ExpressionDefect(
            0,
            "'item_id' is required: name the field that identifies each item, so duplicates from a "
            "join can be collapsed to one.",
        )
    resolved = Path.parse(getattr(path, "dotted", str(path)))
    seen: set = set()
    kept: list[Any] = []
    for record in records:
        value = resolved.resolve(record)
        key = value.value if isinstance(value, Scalar) else None
        if key is not None and key not in seen:
            seen.add(key)
            kept.append(record)
    return tuple(kept)


def _identify(
    records: Sequence[Any], path: Any, argument: str, what: str
) -> Union[tuple[str, ...], ExpressionDefect]:
    """The identity of every record, read from a named field.

    Refuses three things a guess would have swallowed: no field named, the
    field missing on a record, and two records sharing an identity. The last
    matters most -- `optimize` keys placement and capacity by id, so duplicates
    do not collide loudly, they merge: two semesters become one slot with twice
    the room, or one course silently replaces another in the plan.
    """
    if not path:
        return ExpressionDefect(
            0,
            f"'{argument}' is required: name the field that identifies each of the {what}, e.g. "
            f'"{argument}": "courseNumber". It is not guessed, because guessing wrong produces a '
            "plan that looks valid and is not.",
        )

    dotted = getattr(path, "dotted", str(path))
    resolved = Path.parse(dotted)
    identities: list[str] = []
    for record in records:
        value = resolved.resolve(record)
        if not isinstance(value, Scalar):
            return ExpressionDefect(
                0,
                f"a record in {what} has no '{dotted}' to identify it. Available fields: "
                f"{sorted(record.fields)}.",
            )
        identities.append(str(value.value))

    duplicates = sorted({name for name in identities if identities.count(name) > 1})
    if duplicates:
        return ExpressionDefect(
            0,
            f"'{dotted}' is not unique across {what}: {duplicates[:5]} appear more than once. "
            "Identities are pooled, so duplicates would quietly merge two of them into one.",
        )
    return tuple(identities)


def _quantity(record: Any, path: Any, default: float) -> float:
    dotted = getattr(path, "dotted", str(path))
    value = Path.parse(dotted).resolve(record)
    return float(value.value) if isinstance(value, Scalar) and value.is_quantity else default


_HANDLERS = {
    "find": _find,
    "search_corpus": _search_corpus,
    "interpret": _interpret,
    "extract_list": _extract_list,
    "compute": _compute,
    "traverse": _traverse,
    "forecast": _forecast,
    "optimize": _optimize,
    "propose": _propose,
    "plan_term": _plan_term,
}


__all__ = ["DispatchContext", "Dispatched", "dispatch"]


def _term_totals(result: Mapping[str, Any]) -> "Collection | None":
    """One row per term of the plan: its code, course count and credit total.

    The arithmetic the answer needs and the model was paying a turn to redo.
    Built from the same `placedCourses` the plan itself is built from, so the
    two cannot disagree -- which a model-side regrouping could, and did: a live
    run selected `term == "winter"` against a plan holding two winters and
    reported their sum as one 23-credit term.

    None when the planner placed nothing, because a summary of no terms is not a
    fact about the plan, it is an empty collection that reads as "no semesters".
    """
    rows: list[Record] = []
    for term in result.get("terms") or []:
        placed = term.get("placedCourses") or []
        if not placed:
            continue
        rows.append(Record(
            fields={
                "term": Scalar(ScalarKind.IDENTIFIER, str(term.get("semesterCode") or "")),
                "courses": Scalar(ScalarKind.QUANTITY, float(len(placed))),
                "credits": Scalar(
                    ScalarKind.QUANTITY,
                    float(sum(float(course.get("credits") or 0) for course in placed)),
                ),
            },
            basis=Basis.SIMULATED,
        ))
    if not rows:
        return None
    return Collection(
        records=tuple(rows),
        completeness=Completeness(complete=True, total=len(rows)),
    )


def _resolve_number(raw: Any, context: DispatchContext) -> Union[float, None, ExpressionDefect]:
    """A number written directly, or the name of a held scalar fact.

    `{"fact": "credits_needed"}` is the idiom every predicate value already
    uses, and the credit gap IS a held fact -- seeded at the start of the run --
    so requiring a typed digit here would ask the model to launder a fact into a
    literal, which is the one thing the grounding invariant forbids.
    """
    if raw is None:
        return None
    number = _optional_number(raw)
    if number is not None:
        return number
    name = _fact_name(raw)
    if name is None:
        return ExpressionDefect(
            0,
            f"expected a number or {{\"fact\": \"name\"}}, got {raw!r}.",
        )
    held = context.facts.get(name)
    if held is None:
        return ExpressionDefect(0, f"no fact named '{name}'; held: {sorted(context.facts)}")
    value = getattr(held.value, "value", None)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return ExpressionDefect(0, f"'{name}' is not a number, so it cannot be a credit target")
    return float(value)
