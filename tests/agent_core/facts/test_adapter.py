"""The LLM adapter -- phase 11b of docs/agent/tools_implementation_plan.md.

No model is called here. What is under test is the seam: whether the untidy
shapes a real model actually produces survive the trip into the loop, and --
more importantly -- whether unparseable output is refused rather than guessed at.

The last class is the one that matters. Every other test says "this messy input
still works"; those say "this input is not understood, and nothing is invented
to cover it".
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.adapter import ChatModelAdapter, extract_reply


class TestShapesRealModelsProduce:
    def test_bare_json(self) -> None:
        assert extract_reply('{"answer": "You have {n} left."}') == {"answer": "You have {n} left."}

    def test_a_fenced_block(self) -> None:
        content = '```json\n{"answer": "hello {x}"}\n```'
        assert extract_reply(content)["answer"] == "hello {x}"

    def test_a_fence_without_a_language_tag(self) -> None:
        assert extract_reply('```\n{"answer": "hi {x}"}\n```')["answer"] == "hi {x}"

    def test_json_with_a_preamble(self) -> None:
        """'Sure! Here is the JSON:' is not a defect worth failing a turn over."""
        content = 'Sure, here is what I would do:\n\n{"calls": [{"tool": "find", "as": "x", "args": {}}]}'
        assert extract_reply(content)["calls"][0]["tool"] == "find"

    def test_json_with_trailing_commentary(self) -> None:
        content = '{"answer": "done {x}"}\n\nLet me know if you need anything else.'
        assert extract_reply(content)["answer"] == "done {x}"

    def test_a_mapping_passed_through_directly(self) -> None:
        """Some providers return parsed JSON already."""
        assert extract_reply({"answer": "hi {x}"}) == {"answer": "hi {x}"}

    def test_content_delivered_as_a_list_of_parts(self) -> None:
        parts = [{"text": '{"answer": '}, {"text": '"split {x}"}'}]
        assert extract_reply(parts)["answer"] == "split {x}"


class TestUnparseableIsRefusedNotGuessed:
    """The important half.

    An empty mapping makes the loop record an idle turn and say so. Inventing a
    call from output nobody could parse would convert a formatting slip into a
    confident action, which is strictly worse than a wasted turn.
    """

    def test_prose_with_no_json_yields_nothing(self) -> None:
        assert extract_reply("I think you should probably take three more courses.") == {}

    def test_malformed_json_yields_nothing(self) -> None:
        assert extract_reply('{"answer": "unterminated') == {}

    def test_valid_json_of_the_wrong_shape_yields_nothing(self) -> None:
        """A well-formed object the loop cannot act on is not an action."""
        assert extract_reply('{"thoughts": "hmm", "next_step": "look at the transcript"}') == {}

    def test_a_non_string_answer_is_rejected(self) -> None:
        assert extract_reply('{"answer": 42}') == {}

    def test_calls_that_are_not_objects_are_rejected(self) -> None:
        assert extract_reply('{"calls": ["find", "compute"]}') == {}

    def test_empty_content_yields_nothing(self) -> None:
        assert extract_reply("") == {}
        assert extract_reply(None) == {}

    def test_only_the_actionable_keys_survive(self) -> None:
        """Extra keys are dropped rather than forwarded -- the loop should never
        have to know what a model chose to include."""
        result = extract_reply('{"answer": "hi {x}", "confidence": 0.9, "thoughts": "..."}')
        assert result == {"answer": "hi {x}"}


class TestAdapter:
    async def test_it_sends_a_system_prompt_and_the_turn_prompt(self) -> None:
        sent = {}

        class _Chat:
            async def ainvoke(self, messages):
                sent["messages"] = messages
                return type("R", (), {"content": '{"answer": "ok {x}"}'})()

        result = await ChatModelAdapter(_Chat()).respond("TURN PROMPT")
        assert result == {"answer": "ok {x}"}
        assert sent["messages"][0]["role"] == "system"
        assert sent["messages"][1]["content"] == "TURN PROMPT"

    async def test_the_system_prompt_states_the_two_enforced_rules(self) -> None:
        """Rules the code enforces anyway, told to the model so it does not have
        to discover them by rejection."""
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "slot" in SYSTEM_PROMPT.lower()
        assert "name" in SYSTEM_PROMPT.lower() and "paste" in SYSTEM_PROMPT.lower()

    async def test_the_system_prompt_discourages_wandering(self) -> None:
        """The behaviour named at the start of this work: having the answer and
        continuing to look."""
        from app.agent_core.facts.adapter import SYSTEM_PROMPT

        assert "delay" in SYSTEM_PROMPT.lower() or "already answer" in SYSTEM_PROMPT.lower()


class TestBuild:
    def test_no_credentials_yields_no_adapter(self) -> None:
        """Callers get None rather than an object that fails on first use."""
        from app.agent_core.facts.adapter import build_adapter
        from app.config import Settings

        assert build_adapter(settings=Settings(llm_api_key="")) is None


class TestAnswerWrappedAsAToolCall:
    """A recurring live slip: the model wraps its final answer as a CALL to a
    non-existent `answer` tool because it is already in calls-mode. Three ISE
    cases lost a turn to it each. Absorbed at the seam."""

    def test_a_tool_named_answer_becomes_an_answer(self) -> None:
        from app.agent_core.facts.adapter import extract_reply

        reply = extract_reply('{"calls": [{"tool": "answer", "text": "You need 16 credits."}]}')
        assert reply == {"answer": "You need 16 credits."}

    def test_args_nested_text_is_found(self) -> None:
        from app.agent_core.facts.adapter import extract_reply

        reply = extract_reply('{"calls": [{"tool": "final_answer", "args": {"answer": "Done."}}]}')
        assert reply == {"answer": "Done."}

    def test_a_real_tool_call_is_left_alone(self) -> None:
        from app.agent_core.facts.adapter import extract_reply

        reply = extract_reply('{"calls": [{"tool": "find", "as": "x", "args": {"source": "courses"}}]}')
        assert "calls" in reply and reply["calls"][0]["tool"] == "find"
