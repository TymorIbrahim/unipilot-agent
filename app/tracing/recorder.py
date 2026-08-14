"""The steps recorder -- every LLM call, in order, as the spec requires.

`steps` is the graded window into how the agent thinks, so it has to be COMPLETE
and it has to be TRUE. Both properties come from where the recording happens: not
at the call sites, which a future edit could forget, but at the one seam every
model call already passes through -- the chat client itself. `TracedChat` wraps
that client, so a new call site is traced the moment it is written, and a call
cannot be made without being recorded.

The recorder holds no global state. One `StepRecorder` is created per request and
handed to the wrapper, which means concurrent requests cannot interleave their
traces -- a real hazard on a serverless platform that reuses a warm process for
overlapping invocations.

What gets recorded is the prompt as the model actually received it, and the reply
as the model actually sent it. Neither is summarised or cleaned up: a trace that
shows a tidied version of the conversation would be a description of the run
rather than a record of it, and the failures worth reading are exactly the ones a
tidy-up would hide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.tracing.modules import Module

_SYSTEM = "system"
_USER = "user"


@dataclass(frozen=True)
class Step:
    """One LLM call, in the exact shape `/api/execute` must emit."""

    module: str
    system_prompt: str
    user_prompt: str
    response: Any

    def as_dict(self) -> dict[str, Any]:
        # Key casing is the spec's, not ours: `System_prompt` / `User_prompt`.
        return {
            "module": self.module,
            "prompt": {"System_prompt": self.system_prompt, "User_prompt": self.user_prompt},
            "response": self.response,
        }


@dataclass
class StepRecorder:
    """Request-scoped, ordered log of every LLM call the agent made."""

    steps: list[Step] = field(default_factory=list)

    def record(self, module: Module | str, system_prompt: str, user_prompt: str, response: Any) -> None:
        name = module.name if isinstance(module, Module) else str(module)
        self.steps.append(
            Step(module=name, system_prompt=system_prompt, user_prompt=user_prompt, response=response)
        )

    def as_list(self) -> list[dict[str, Any]]:
        return [step.as_dict() for step in self.steps]

    def __len__(self) -> int:
        return len(self.steps)


def split_messages(messages: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """Split a chat message list into the spec's two prompt fields.

    Several of the agent's calls send only a user message; the spec's step schema
    still wants both keys, so an absent system prompt records as empty rather
    than being omitted -- a consumer reading `steps` should never have to handle
    two different step shapes.

    Multiple messages of the same role are joined rather than last-wins: dropping
    one would under-report what the model was actually told, which is the one
    thing this record exists to get right.
    """
    system: list[str] = []
    user: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).lower()
        content = message.get("content", "")
        text = content if isinstance(content, str) else str(content)
        if role == _SYSTEM:
            system.append(text)
        elif role == _USER:
            user.append(text)
    return "\n\n".join(system), "\n\n".join(user)


class TracedChat:
    """A chat client that records every call it makes, then delegates.

    Deliberately transparent: it implements `ainvoke` with the same signature and
    return value as the client it wraps, so the reasoning core is unaware it is
    being traced and needs no edit to support tracing.
    """

    def __init__(self, chat: Any, recorder: StepRecorder, module: Module | str) -> None:
        self._chat = chat
        self._recorder = recorder
        self._module = module

    def for_module(self, module: Module | str) -> TracedChat:
        """A view of the same client that stamps a different module name.

        The agent uses one chat client from several modules; each needs to appear
        under its own name in the trace, and the recorder must stay shared so the
        steps keep their true global order.
        """
        return TracedChat(self._chat, self._recorder, module)

    async def ainvoke(self, messages: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> Any:
        reply = await self._chat.ainvoke(messages, *args, **kwargs)
        system_prompt, user_prompt = split_messages(messages)
        self._recorder.record(
            self._module,
            system_prompt,
            user_prompt,
            getattr(reply, "content", reply),
        )
        return reply

    def __getattr__(self, name: str) -> Any:
        # Anything the reasoning core needs beyond `ainvoke` passes through, so
        # wrapping never removes a capability the unwrapped client had.
        return getattr(self._chat, name)


__all__ = ["Step", "StepRecorder", "TracedChat", "split_messages"]
