"""The turn loop -- phase 10 of docs/agent/tools_implementation_plan.md.

The model is scripted, so what is under test is the LOOP: whether it terminates,
whether it feeds failures back usefully, and whether the governors fire on the
behaviours that motivated this whole redesign.

Two of those behaviours are named directly in the tests below, because they are
the ones that were observed live and started the work:

  - finding the answer and then wandering into empty turns
  - being rejected with no legal move and burning the budget rediscovering it
"""

from __future__ import annotations

from app.agent_core.facts.answer import HeldFact
from app.agent_core.facts.dispatch import DispatchContext
from app.agent_core.facts.loop import NO_PROGRESS_LIMIT, run_loop
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _coll(*ids: str) -> Collection:
    return Collection(
        records=tuple(Record(fields={"id": Scalar(I, i)}, basis=Basis.OFFICIAL_RECORD) for i in ids),
        completeness=Completeness(complete=True, total=len(ids)),
    )


class _ScriptedModel:
    """Replays a fixed list of replies, and records the prompts it saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    async def respond(self, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else {}


def _context(**facts) -> DispatchContext:
    return DispatchContext(
        facts={name: HeldFact(value=value, basis=Basis.OFFICIAL_RECORD) for name, value in facts.items()}
    )


class TestHappyPath:
    async def test_it_answers_from_facts_it_already_holds(self) -> None:
        model = _ScriptedModel({"answer": "You have {count} courses."})
        context = _context(count=Scalar(Q, 3.0))
        result = await run_loop("how many?", model, context)
        assert result.outcome == "answered"
        assert result.answer.text == "You have 3 courses."
        assert result.turns == 1

    async def test_it_calls_a_tool_then_answers(self) -> None:
        model = _ScriptedModel(
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}}]},
            {"answer": "You need {n} more."},
        )
        result = await run_loop("how many?", model, _context(required=_coll("a", "b")))
        assert result.outcome == "answered"
        assert result.answer.text == "You need 2 more."
        assert result.turns == 2

    async def test_facts_accumulate_across_turns(self) -> None:
        model = _ScriptedModel(
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}}]},
            {"answer": "{n}"},
        )
        result = await run_loop("q", model, _context(required=_coll("a")))
        assert "n" in result.facts and "required" in result.facts


class TestFailuresComeBack:
    async def test_a_tool_defect_is_reported_to_the_next_turn(self) -> None:
        """A failure the model never hears about is a failure it repeats."""
        model = _ScriptedModel(
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "bad", "source": "required",
                 "stages": [{"op": "aggregate", "agg": "sum", "path": "ghost"}]}
            ]}}]},
            {"answer": "{count} things"},
        )
        context = _context(required=_coll("a"), count=Scalar(Q, 1.0))
        await run_loop("q", model, context)
        assert "bad" in model.prompts[1] and "ghost" in model.prompts[1]

    async def test_a_rejected_answer_comes_back_with_its_reason(self) -> None:
        """The retry has to differ from its predecessor, which it can only do if
        it is told what was wrong."""
        model = _ScriptedModel(
            {"answer": "You have 3 courses."},          # a typed number
            {"answer": "You have {count} courses."},    # corrected
        )
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "answered"
        assert "refused" in model.prompts[1].lower()
        assert "no fact" in model.prompts[1]

    async def test_an_unsound_plan_is_sent_back_with_the_specific_reason(self) -> None:
        """Grounding passes -- the -82 is a real fact, correctly slotted -- but the
        verify step catches that no grade is negative, and hands the reason back so
        the retry can floor it. The corrected plan then ships."""
        bad = Collection(
            records=(Record(
                fields={"number": Scalar(I, "00940395"), "credits": Scalar(Q, 1.5), "min_grade": Scalar(Q, -82.0)},
                basis=Basis.OFFICIAL_RECORD,
            ),),
            completeness=Completeness(complete=True, total=1),
        )
        ok = Collection(
            records=(Record(
                fields={"number": Scalar(I, "00940412"), "credits": Scalar(Q, 4.0), "min_grade": Scalar(Q, 19.25)},
                basis=Basis.OFFICIAL_RECORD,
            ),),
            completeness=Completeness(complete=True, total=1),
        )
        model = _ScriptedModel(
            {"answer": "Winter plan: {bad:detail}"},   # unsound: a negative minimum
            {"answer": "Winter plan: {ok:detail}"},    # corrected: floored to a real grade
        )
        context = _context(bad=bad, ok=ok, total_points=Scalar(Q, 5243.0), total_credits=Scalar(Q, 62.5))

        result = await run_loop("plan winter to keep my GPA above 80", model, context)

        assert result.outcome == "answered"
        # One record, so no list bullet stranded mid-sentence, and the
        # projected `min_grade` reads as "min grade".
        assert result.answer.text == (
            "Winter plan: number 00940412 · credits 4 · min grade 19.25"
        )
        assert "refused" in model.prompts[1].lower() and "below 0" in model.prompts[1]


class TestGovernors:
    async def test_repeated_rejection_stops_rather_than_spending_the_budget(self) -> None:
        """The old loop could be rejected every turn with no legal move and kept
        trying until the budget ran out, then shipped something unverified.
        Stopping and saying why is the better failure."""
        model = _ScriptedModel(*[{"answer": "You have 3 courses."} for _ in range(8)])
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)), max_turns=8)
        assert result.outcome == "refused"
        assert result.turns < 8, "it should stop early rather than exhaust the budget"
        assert result.answer is None, "an unverified answer must never be returned"

    async def test_empty_turns_terminate_the_loop(self) -> None:
        """'Finding the answer then wandering into empty turns' -- named at the
        start of this work as the behaviour to eliminate."""
        model = _ScriptedModel(*[{} for _ in range(NO_PROGRESS_LIMIT + 3)])
        result = await run_loop("q", model, _context(count=Scalar(Q, 1.0)))
        assert result.outcome == "stalled"
        assert result.turns == NO_PROGRESS_LIMIT, "it must stop AT the limit, not merely eventually"

    async def test_calls_that_produce_no_new_facts_count_as_no_progress(self) -> None:
        """Busy is not the same as progressing. Repeating a failing call looks
        active and achieves nothing."""
        failing = {"calls": [{"tool": "compute", "args": {"pipelines": [
            {"name": "x", "source": "nonexistent", "stages": []}
        ]}}]}
        model = _ScriptedModel(failing, failing, failing, failing)
        result = await run_loop("q", model, _context(count=Scalar(Q, 1.0)))
        assert result.outcome == "stalled"
        assert result.turns <= NO_PROGRESS_LIMIT + 1

    async def test_the_turn_budget_is_honoured(self) -> None:
        # Each turn derives something GENUINELY new (a different aggregate), so
        # the budget is what stops this run and not the repeat guard below.
        aggregates = ["count", "sum", "avg", "min", "max"]
        alternating = [
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": f"n{i}", "source": "required",
                 "stages": [{"op": "aggregate", "agg": agg, "field": "credits"}]}
            ]}}]}
            for i, agg in enumerate(aggregates * 2)
        ]
        model = _ScriptedModel(*alternating)
        result = await run_loop("q", model, _context(required=_coll("a")), max_turns=3)
        assert result.outcome == "exhausted"
        assert result.turns == 3
        assert "budget" in result.reason

    async def test_re_deriving_under_a_new_name_is_not_progress(self) -> None:
        """The measured failure: ten live runs spent their budget re-deriving
        values they already held, each lap under a fresh name. Every lap produced
        a non-empty fact, so the old no-progress guard -- which asked only
        whether a fact arrived -- reset on every one of them and never fired.

        A run that laps like this must stall at the limit, not at the clock.
        """
        lap = lambda name: {"calls": [{"tool": "compute", "args": {"pipelines": [
            {"name": name, "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
        ]}}]}
        model = _ScriptedModel(*[
            lap(n) for n in ("prereq_groups", "required_groups", "obligation_groups",
                             "met_groups", "eligible", "req_codes")
        ])
        result = await run_loop("q", model, _context(required=_coll("a", "b")), max_turns=6)
        assert result.outcome == "stalled", "renaming an identical derivation is not progress"
        assert result.turns <= NO_PROGRESS_LIMIT + 1, "it must stop at the limit, not the budget"

    async def test_an_identical_repeated_call_is_not_progress(self) -> None:
        """The other measured shape: the SAME search re-issued verbatim. One live
        run sent one query sixteen times across twenty-four steps."""
        same = {"calls": [{"tool": "compute", "args": {"pipelines": [
            {"name": "n", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
        ]}}]}
        model = _ScriptedModel(*[same] * 6)
        result = await run_loop("q", model, _context(required=_coll("a")), max_turns=6)
        assert result.outcome == "stalled"
        assert result.turns <= NO_PROGRESS_LIMIT + 1

    async def test_a_repeat_is_reported_back_to_the_model(self) -> None:
        """Stalling silently would waste the turns before the limit. The model is
        told it already made the call, so it can change course instead."""
        same = {"calls": [{"tool": "compute", "args": {"pipelines": [
            {"name": "n", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
        ]}}]}
        model = _ScriptedModel(same, same, {"answer": "You have {n}."})
        result = await run_loop("q", model, _context(required=_coll("a", "b")))
        assert result.outcome == "answered"
        assert "already ran" in model.prompts[-1], "the repeat must be named in the next prompt"

    async def test_a_genuinely_new_derivation_still_counts(self) -> None:
        """The guard must not punish real work: same source, different question."""
        model = _ScriptedModel(
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "how_many", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}}]},
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "total", "source": "required",
                 "stages": [{"op": "aggregate", "agg": "sum", "field": "credits"}]}
            ]}}]},
            {"answer": "You have {how_many}."},
        )
        result = await run_loop("q", model, _context(required=_coll("a", "b")))
        assert result.outcome == "answered"
        assert result.turns == 3, "neither derivation should have been treated as a repeat"


class TestWhatTheModelSees:
    async def test_the_turn_prompt_carries_the_question_and_the_facts(self) -> None:
        model = _ScriptedModel({"answer": "{count}"})
        await run_loop("how many?", model, _context(count=Scalar(Q, 1.0)))
        prompt = model.prompts[0]
        assert "how many?" in prompt
        assert "count = 1" in prompt, "held facts must be visible"

    async def test_the_turn_prompt_no_longer_repeats_the_static_catalog(self) -> None:
        """The catalog and source list moved to the SYSTEM prompt. They are
        constant for a run -- 15,216 characters of a late turn's 18,411 -- and
        sitting after the question they broke every prompt-prefix cache, since
        the prefix diverges at the question and everything behind it is re-read.
        """
        model = _ScriptedModel({"answer": "{count}"})
        await run_loop("how many?", model, _context(count=Scalar(Q, 1.0)))
        assert "## compute" not in model.prompts[0]
        assert "data sources for `find`" not in model.prompts[0]

    def test_the_system_prompt_carries_the_catalog_and_the_sources(self) -> None:
        """Moved, not dropped -- the model still has to be told what it can call."""
        from app.agent_core.facts.adapter import build_system_prompt
        from app.agent_core.facts.sources import REGISTRY

        system = build_system_prompt(DispatchContext(schemas=REGISTRY))
        assert "## compute" in system, "the tool catalog must be present"
        assert "data sources for `find`" in system
        assert "academic advising agent" in system, "and the standing instructions"

    def test_an_unwired_tool_stays_out_of_the_system_prompt(self) -> None:
        """The catalog is context-dependent, which is why the system prompt is
        built per run rather than imported as a constant. A system prompt
        promising a tool the dispatcher would refuse is the catalog-honesty
        failure with a new hiding place."""
        from app.agent_core.facts.adapter import build_system_prompt
        from app.agent_core.facts.sources import REGISTRY

        bare = build_system_prompt(DispatchContext(schemas=REGISTRY))
        wired = build_system_prompt(
            DispatchContext(schemas=REGISTRY, retriever=object(), extractor=object())
        )
        assert "## search_corpus" not in bare
        assert "## search_corpus" in wired

    async def test_the_prompt_shows_shapes_not_payloads(self) -> None:
        """The prompt must grow with the NUMBER of facts, not their size, or one
        large fetch crowds out everything needed to reason with it."""
        big = _coll(*[f"course-{n}" for n in range(400)])
        model = _ScriptedModel({"answer": "{count}"})
        await run_loop("q", model, _context(courses=big, count=Scalar(Q, 1.0)))
        assert "course-399" not in model.prompts[0]
        assert "400 records" in model.prompts[0]


class TestTranscript:
    async def test_every_turn_is_recorded(self) -> None:
        model = _ScriptedModel(
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": "n", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}}]},
            {"answer": "{n}"},
        )
        result = await run_loop("q", model, _context(required=_coll("a")))
        assert [t.action for t in result.transcript] == ["call", "answer"]

    async def test_a_rejection_is_recorded_with_its_reason(self) -> None:
        model = _ScriptedModel({"answer": "3 courses"}, {"answer": "{count}"})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        rejected = [t for t in result.transcript if t.action == "rejected"]
        assert len(rejected) == 1 and "no fact" in rejected[0].detail


class TestDecline:
    """An out-of-scope question concludes cleanly, without pretending to answer.

    A live run looped `search_corpus` over an academic corpus for eight turns on
    "what's the weather in Haifa" -- it could not answer (no facts to cite) and
    could not stop (no way to say "not my domain"). Decline is that missing
    conclusion, and it is distinct from a grounding failure: there was nothing to
    ground.
    """

    async def test_a_decline_concludes_the_loop_at_once(self) -> None:
        model = _ScriptedModel({"decline": "I can only help with your studies, not the weather."})
        result = await run_loop("What's the weather in Haifa?", model, _context())
        assert result.outcome == "declined"
        assert result.turns == 1
        assert "weather" in result.reason
        assert result.answer is None

    async def test_a_decline_needs_no_facts(self) -> None:
        """The whole point: it is allowed to stand on nothing, where an answer
        is not."""
        model = _ScriptedModel({"decline": "Out of scope."})
        result = await run_loop("Anything.", model, _context())
        assert result.outcome == "declined"

    async def test_a_decline_after_fetching_records_is_refused_not_accepted(self) -> None:
        """The planning failure: the model fetched 49 curriculum courses, the
        transcript, and the offerings, then declined the hard synthesis at turn
        2. Once records are in hand the question is in scope -- a decline there
        is giving up, so it is sent back to keep working, not concluded."""
        model = _ScriptedModel(
            {"decline": "I need to derive requirements and thresholds first."},
            {"answer": "You hold {courses} courses to work from."},
        )
        context = _context(courses=_coll("00940224", "00960211"))
        result = await run_loop("Plan my next two semesters.", model, context)
        # It did NOT conclude on the decline; it was pushed to answer.
        assert result.outcome == "answered"
        assert any(t.action == "decline-refused" for t in result.transcript)

    async def test_a_persistent_post_fetch_decline_concludes_as_refused(self) -> None:
        """If it will not push through, the loop stops honestly rather than
        spinning -- the route ships a grounded partial or graceful message."""
        model = _ScriptedModel(*[{"decline": "still can't"} for _ in range(6)])
        context = _context(courses=_coll("00940224"))
        result = await run_loop("Plan it.", model, context)
        assert result.outcome == "refused"


class TestProposalFeedback:
    """`propose` returns no fact, so its call summary reads "-> 0 facts" -- which
    looks like a failure. A live run re-proposed EIGHT times chasing a success
    signal, then exhausted its budget. The loop must SAY the proposal landed."""

    async def test_a_proposal_concludes_the_loop_at_once(self) -> None:
        """Terminal, not a fact to narrate in a second turn. A live run
        re-proposed eight times because `propose` returns no success signal;
        another had its narration refused for slotting an ObjectId. The proposal
        itself is the conclusion."""
        context = _context(course=_coll("00960211"))
        model = _ScriptedModel(
            {"calls": [{"tool": "propose", "as": "p", "args": {
                "action": "register", "target": "00960211", "grounds": ["course"]}}]},
        )
        result = await run_loop("register me for 00960211", model, context)

        assert result.outcome == "proposed"
        assert result.proposal is not None
        assert result.turns == 1
        # It did NOT ask the model for a second turn.
        assert len(model.prompts) == 1


class TestConversationHistory:
    """A follow-up run sees the prior exchange, so "continue" resolves -- but
    the history is CONTEXT, never a fact the model can cite."""

    async def test_prior_exchanges_reach_the_model(self) -> None:
        from app.agent_core.facts.conversation import Exchange

        model = _ScriptedModel({"answer": "Continuing: you hold {c}."})
        context = _context(c=_coll("00940224"))
        await run_loop(
            "yes, continue",
            model,
            context,
            history=[Exchange("plan my two semesters", "Your track has 49 courses.")],
        )
        prompt = model.prompts[0]
        assert "CONVERSATION SO FAR" in prompt
        assert "plan my two semesters" in prompt
        assert "49 courses" in prompt

    async def test_history_is_marked_as_context_not_grounded_fact(self) -> None:
        from app.agent_core.facts.conversation import Exchange

        model = _ScriptedModel({"answer": "ok {c}."})
        await run_loop(
            "continue", model, _context(c=_coll("00940224")),
            history=[Exchange("q", "a prior answer with a number 155")],
        )
        # The prior answer's "155" is in the history block, but that block is
        # explicitly prior-context; it is not a slot and does not ground an answer.
        assert "re-derive every fact fresh" in model.prompts[0]

    async def test_no_history_leaves_the_prompt_unchanged(self) -> None:
        model = _ScriptedModel({"answer": "You hold {c}."})
        await run_loop("q", model, _context(c=_coll("00940224")))
        assert "CONVERSATION SO FAR" not in model.prompts[0]


class TestSeededFactsAreNotFetchedFacts:
    """Seeding the profile silently disabled every out-of-scope decline.

    `run_advice` opens a run by putting the student's identity and four profile
    columns into the context -- a performance fix that deleted two turns from
    every planning question. The decline guard, written earlier, asked whether
    any fact other than `me` was held, and from that day the answer was always
    yes. Asked for the weather, the agent could no longer decline: the decline
    was refused as "you already fetched records", retried, and returned as
    `refused`, which the route renders to the student as "I wasn't able to work
    that out from your records with confidence."

    Nothing failed loudly and no test saw it, because the tests here build their
    context by hand and never seeded a profile. So the seeded names are now
    PASSED IN from the one place that seeds them, and this pins the behaviour
    against the real set.
    """

    def _seeded(self) -> dict:
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        values = {
            "me": Scalar(ScalarKind.IDENTIFIER, "6a578a2da43a2cfe1bcc791c"),
            "program_slug": Scalar(ScalarKind.IDENTIFIER, "track-information-systems-engineering"),
            "catalog_year": Scalar(Q, 2025.0),
            "current_semester": Scalar(ScalarKind.IDENTIFIER, "2025-2"),
            "max_credits_per_semester": Scalar(Q, 18.0),
            # The credit standing, seeded from the same profile query since the
            # 60s ceiling made a turn a twentieth of the whole request.
            "credits_completed": Scalar(Q, 129.5),
            "credits_required": Scalar(Q, 155.0),
            "credits_needed": Scalar(Q, 25.5),
        }
        assert set(values) == set(SEEDED_FACT_NAMES), "the real seeded set changed"
        return values

    async def test_an_out_of_scope_question_can_still_be_declined(self) -> None:
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        model = _ScriptedModel({"decline": "I can only help with your studies, not the weather."})
        result = await run_loop(
            "What's the weather in Haifa?", model, _context(**self._seeded()),
            seeded_facts=SEEDED_FACT_NAMES,
        )
        assert result.outcome == "declined"
        assert result.turns == 1, "a decline must conclude at once, not spend the budget"

    async def test_a_decline_after_a_real_fetch_is_still_refused(self) -> None:
        """The other half: seeding must not make the guard toothless either."""
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        model = _ScriptedModel(
            {"decline": "I need to derive requirements first."},
            {"answer": "You hold {courses} courses to work from."},
        )
        context = _context(**self._seeded(), courses=_coll("00940224", "00960211"))
        result = await run_loop(
            "Plan my next two semesters.", model, context, seeded_facts=SEEDED_FACT_NAMES,
        )
        assert result.outcome == "answered"
        assert any(t.action == "decline-refused" for t in result.transcript)


class TestAnAnswerBeforeAnythingIsFetched:
    """Two of three live `semesters_to_graduate` runs used the answer channel to
    say they were not ready -- one on turn 1, holding nothing:

        "I'm missing the curriculum and transcript facts needed to derive your
         graduation timeline. I need to fetch your track courses first."

    That is a protocol mistake, not an answer the facts failed to support, and
    charging it to REJECTION_LIMIT spent a third of the run's tolerance for
    genuinely unsupportable answers before the work began.
    """

    async def test_the_first_ones_do_not_consume_a_rejection(self) -> None:
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        premature = {"answer": "I need to fetch your records first."}
        model = _ScriptedModel(
            premature, premature,
            {"calls": [{"tool": "find", "as": "courses", "args": {"source": "courses"}}]},
        )
        context = _context(me=Scalar(ScalarKind.IDENTIFIER, "6a57"))
        result = await run_loop(
            "How many semesters?", model, context, seeded_facts=SEEDED_FACT_NAMES, max_turns=3,
        )
        assert result.outcome != "refused", "a premature answer was charged as a rejection"
        assert sum(1 for t in result.transcript if t.action == "premature-answer") == 2

    async def test_endless_premature_answers_still_conclude(self) -> None:
        """Forgiving them removed the only brake on a run that never fetches.

        Asked "what can you not answer about my degree?" -- a question with
        nothing to fetch by its nature -- the deployed loop made 35 consecutive
        answer attempts, zero tool calls, and ran 169.6s until the clock stopped
        it. Every attempt was premature, so none was charged, so nothing ever
        concluded."""
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        premature = {"answer": "I am not ready yet."}
        model = _ScriptedModel(*[premature] * 12)
        result = await run_loop(
            "what can you not answer?", model, _context(me=Scalar(ScalarKind.IDENTIFIER, "6a57")),
            seeded_facts=SEEDED_FACT_NAMES, max_turns=12,
        )
        assert result.outcome == "refused", "the run never concluded"
        assert result.turns <= 6, f"took {result.turns} turns to stop"

    async def test_the_model_is_told_what_to_do_instead(self) -> None:
        from app.agent_core.facts.service import SEEDED_FACT_NAMES

        model = _ScriptedModel({"answer": "I need to fetch your records first."},
                               {"answer": "Still not ready."})
        await run_loop("q", model, _context(me=Scalar(ScalarKind.IDENTIFIER, "6a57")),
                       seeded_facts=SEEDED_FACT_NAMES, max_turns=2)
        assert "before fetching anything" in model.prompts[-1]

    async def test_a_genuinely_ungrounded_answer_still_counts(self) -> None:
        """Once facts ARE held, a bad answer is a rejection like any other."""
        model = _ScriptedModel({"answer": "You have 42 courses."},
                               {"answer": "You have 42 courses."},
                               {"answer": "You have 42 courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "refused"


class TestARepeatedCallThatFails:
    """The one repetition the loop never warned about.

    Asked "who teaches 00960211?" -- something the schema does not record at all
    -- the deployed agent spent turns 6, 7 and 8 issuing the IDENTICAL
    `interpret` call, collecting the identical defect each time, and exhausted
    its budget having learned nothing after turn 3.

    The repeat warning needed facts to report ("you already ran this and it
    returned ..."), and a defect produces none, so the `elif outcome.facts`
    branch skipped exactly the case where repeating is most obviously futile.
    The per-turn defect note says what broke; it never said "you have already
    tried precisely this".
    """

    async def test_the_second_identical_failing_call_is_named_as_a_repeat(self) -> None:
        call = {"tool": "find", "as": "x", "args": {"source": "nope"}}
        model = _ScriptedModel({"calls": [call]}, {"calls": [call]},
                               {"answer": "You have {count} courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)), max_turns=3)
        assert "already ran" in model.prompts[-1], "a failing repeat went unwarned"
        assert "DIFFERENT route" in model.prompts[-1], "say what to do instead"
        assert result.outcome == "answered"

    async def test_the_first_failure_is_not_called_a_repeat(self) -> None:
        model = _ScriptedModel({"calls": [{"tool": "find", "as": "x", "args": {"source": "nope"}}]},
                               {"answer": "You have {count} courses."})
        await run_loop("q", model, _context(count=Scalar(Q, 3.0)), max_turns=2)
        assert "already ran" not in model.prompts[-1]

    async def test_a_successful_repeat_still_reports_what_it_returned(self) -> None:
        """The original branch must keep working -- it names the VALUE, which is
        what lets the model use the fact instead of re-fetching it."""
        call = {"tool": "compute",
                "args": {"pipelines": [{"name": "n", "value": {"add": [{"value": 1}, {"value": 1}]}}]}}
        model = _ScriptedModel({"calls": [call]}, {"calls": [call]},
                               {"answer": "You have {count} courses."})
        await run_loop("q", model, _context(count=Scalar(Q, 3.0)), max_turns=3)
        assert "already ran" in model.prompts[-1]
        assert "Repeating it cannot change the result" in model.prompts[-1]


class TestAnsweringIsAReplyShapeNotATool:
    """`{"calls": [{"tool": "answer", ...}]}` cost a turn in four of ten live
    requests, and one run spent its second-to-last turn on it.

    There is no `answer` tool and there cannot be -- it ends the run rather than
    producing a fact -- so the dispatcher reported "unknown tool 'answer';
    available: [...]" and the model wrote the same text again next turn in the
    right envelope. The intent was never in doubt.

    Being forgiving about the ENVELOPE must not be forgiving about the contents:
    the lifted text goes down the ordinary answer path, grounding and
    post-conditions included.
    """

    async def test_the_answer_call_is_honoured(self) -> None:
        model = _ScriptedModel({"calls": [{"tool": "answer",
                                           "args": {"answer": "You have {count} courses."}}]})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "answered"
        assert result.answer.text == "You have 3 courses."
        assert result.turns == 1, "the turn must not be spent learning the protocol"

    async def test_it_is_still_grounded(self) -> None:
        """A typed digit inside a lifted answer is refused exactly as usual."""
        model = _ScriptedModel(
            {"calls": [{"tool": "answer", "args": {"answer": "You have 42 courses."}}]},
            {"answer": "You have {count} courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "answered"
        assert any(t.action in ("rejected", "premature-answer") for t in result.transcript)

    async def test_a_decline_call_is_honoured_too(self) -> None:
        model = _ScriptedModel({"calls": [{"tool": "decline",
                                           "args": {"decline": "Not about your studies."}}]})
        result = await run_loop("weather?", model, _context())
        assert result.outcome == "declined"

    async def test_a_real_reply_is_never_rewritten(self) -> None:
        model = _ScriptedModel({"answer": "You have {count} courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.answer.text == "You have 3 courses."

    async def test_ordinary_calls_are_untouched(self) -> None:
        model = _ScriptedModel(
            {"calls": [{"tool": "find", "as": "x", "args": {"source": "courses"}}]},
            {"answer": "You have {count} courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "answered"
        assert any(t.action == "call" for t in result.transcript)

    async def test_an_answer_call_with_no_text_falls_through(self) -> None:
        """Nothing to lift means the dispatcher reports it as it always did."""
        model = _ScriptedModel({"calls": [{"tool": "answer", "args": {}}]},
                               {"answer": "You have {count} courses."})
        result = await run_loop("q", model, _context(count=Scalar(Q, 3.0)))
        assert result.outcome == "answered"
