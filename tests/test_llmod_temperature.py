"""A gpt-5 model accepts only its default temperature, and the two providers
disagree about whether sending one is an error.

Found on the swap to LLMod, one call before deploying it. OpenAI direct
accepted `temperature=0.0` and ignored it; LLMod routes through LiteLLM, which
returns 400:

    litellm.UnsupportedParamsError: gpt-5 models don't support temperature=0.0.
    Only temperature=1 is supported.

Every `/api/execute` would have returned 400 on submission day, with the whole
local suite green -- nothing in it makes a live call, so nothing in it could
have known. The determinism that 0.0 was for was never being honoured anyway:
the model has one temperature, and sending it was decoration.

Matched on the model NAME rather than the provider, because the constraint
belongs to the model. `MB5R2CF-azure/gpt-5.4-mini` and a bare `gpt-5.4-mini`
are the same model reached two ways, and only one of the routes enforced it.
"""

from __future__ import annotations

import pytest

from app.agent_core.reasoning.llm_client import _REFUSES_TEMPERATURE


class TestWhichModelsRefuseATemperature:
    @pytest.mark.parametrize(
        "model",
        ["MB5R2CF-azure/gpt-5.4-mini", "gpt-5.4-mini", "gpt-5", "o3-mini"],
    )
    def test_a_gpt5_or_o_series_model_gets_no_temperature(self, model: str) -> None:
        assert _REFUSES_TEMPERATURE.search(model)

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4-turbo", "claude-sonnet-4"])
    def test_every_other_model_still_gets_one(self, model: str) -> None:
        """Dropping it everywhere would silently change behaviour for models
        that DO honour it, which is a different bug wearing this fix's clothes."""
        assert not _REFUSES_TEMPERATURE.search(model)

    def test_the_prefixed_and_bare_names_agree(self) -> None:
        """The provider prefix must not decide it -- the same model reached two
        ways has the same constraint, and that is what made this invisible."""
        assert bool(_REFUSES_TEMPERATURE.search("MB5R2CF-azure/gpt-5.4-mini")) == bool(
            _REFUSES_TEMPERATURE.search("gpt-5.4-mini")
        )


class TestTheClientOmitsIt:
    def test_no_temperature_is_sent_for_the_graded_model(self, monkeypatch) -> None:
        captured: dict = {}

        class _FakeChat:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import langchain_openai

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChat)
        from app.agent_core.reasoning.llm_client import _cached_chat_llm

        _cached_chat_llm.cache_clear()
        _cached_chat_llm(
            api_key="k", base_url="https://api.llmod.ai/v1",
            model="MB5R2CF-azure/gpt-5.4-mini", temperature=0.0,
            timeout=30.0, thinking_enabled=False, reasoning_effort=None,
        )
        _cached_chat_llm.cache_clear()
        assert "temperature" not in captured
        assert captured["model"] == "MB5R2CF-azure/gpt-5.4-mini"

    def test_it_is_still_sent_for_a_model_that_accepts_one(self, monkeypatch) -> None:
        captured: dict = {}

        class _FakeChat:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        import langchain_openai

        monkeypatch.setattr(langchain_openai, "ChatOpenAI", _FakeChat)
        from app.agent_core.reasoning.llm_client import _cached_chat_llm

        _cached_chat_llm.cache_clear()
        _cached_chat_llm(
            api_key="k", base_url="", model="gpt-4o-mini", temperature=0.0,
            timeout=30.0, thinking_enabled=False, reasoning_effort=None,
        )
        _cached_chat_llm.cache_clear()
        assert captured["temperature"] == 0.0
