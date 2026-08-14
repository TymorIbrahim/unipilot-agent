"""Builds the chat model the reasoning loop talks to.

Deliberately kept at the import path and function name the ported reasoning core
already uses, so `facts/adapter.py` needs no edit: only the settings it reads
differ from UniPilot's, and that difference stops here.

Simpler than the original for one reason. UniPilot had to translate its
reasoning controls into whichever provider's wire format was active (DeepSeek
expresses "no thinking" differently from OpenAI). Here both the development
provider and the graded one speak the OpenAI format, so there is one shape to
emit and no translation table to maintain.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _reasoning_kwargs(*, thinking_enabled: bool, reasoning_effort: str | None) -> dict[str, Any]:
    """Translate our reasoning controls into OpenAI-format request fields.

    When thinking is on we pass `reasoning_effort` only if one was set, leaving
    the provider's own default in place otherwise. Forcing a value here would
    silently override a model whose sensible default we have no reason to
    second-guess -- and reasoning effort is the single biggest lever on both
    latency and spend, so it should change only when someone means it to.
    """
    if not thinking_enabled:
        return {}
    if not reasoning_effort:
        return {}
    return {"reasoning_effort": reasoning_effort}


@lru_cache(maxsize=8)
def _cached_chat_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: float | None,
    thinking_enabled: bool,
    reasoning_effort: str | None,
) -> Any:
    """One client per distinct configuration.

    Cached because constructing a client per turn wastes the underlying HTTP
    connection pool -- on a loop that makes eight to fourteen calls per request
    that is real latency. Every argument is part of the key on purpose: if
    `timeout` or `model` were excluded, the first caller's choice would silently
    become everyone's.
    """
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "api_key": api_key,
        **_reasoning_kwargs(thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatOpenAI(**kwargs)


def build_chat_llm(
    *,
    settings: Settings | None = None,
    temperature: float = 0.0,
    model: str | None = None,
    thinking_enabled: bool | None = None,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
) -> Any | None:
    """Return a chat client, or None when the agent is not configured.

    Returning None rather than raising keeps "no API key" a condition the
    caller can report cleanly -- the HTTP layer turns it into a plain error
    message instead of a stack trace.
    """
    cfg = settings or get_settings()
    api_key = cfg.llm_api_key.strip()
    if not api_key:
        return None

    try:
        import langchain_openai  # noqa: F401 -- availability check only
    except ImportError:
        logger.warning("chat_llm_unavailable: langchain_openai is not installed")
        return None

    return _cached_chat_llm(
        api_key=api_key,
        base_url=cfg.llm_base_url.strip().rstrip("/"),
        model=(model or cfg.llm_chat_model).strip(),
        temperature=temperature,
        timeout=timeout if timeout is not None else cfg.llm_timeout_s,
        thinking_enabled=(
            thinking_enabled if thinking_enabled is not None else cfg.llm_thinking_enabled
        ),
        reasoning_effort=reasoning_effort if reasoning_effort is not None else cfg.llm_reasoning_effort,
    )


def reset_chat_llm_cache() -> None:
    """Drop cached clients -- needed after a settings change, and in tests."""
    _cached_chat_llm.cache_clear()


__all__ = ["build_chat_llm", "reset_chat_llm_cache"]
