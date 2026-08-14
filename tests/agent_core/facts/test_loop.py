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
        assert result.answer.text == "Winter plan: - number 00940412 · credits 4 · min_grade 19.25"
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
        alternating = [
            {"calls": [{"tool": "compute", "args": {"pipelines": [
                {"name": f"n{i}", "source": "required", "stages": [{"op": "aggregate", "agg": "count"}]}
            ]}}]}
            for i in range(10)
        ]
        model = _ScriptedModel(*alternating)
        result = await run_loop("q", model, _context(required=_coll("a")), max_turns=3)
        assert result.outcome == "exhausted"
        assert result.turns == 3
        assert "budget" in result.reason


class TestWhatTheModelSees:
    async def test_the_prompt_carries_the_catalog_and_the_facts(self) -> None:
        model = _ScriptedModel({"answer": "{count}"})
        await run_loop("how many?", model, _context(count=Scalar(Q, 1.0)))
        prompt = model.prompts[0]
        assert "how many?" in prompt
        assert "## compute" in prompt, "the tool catalog must be present"
        assert "count = 1" in prompt, "held facts must be visible"

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
