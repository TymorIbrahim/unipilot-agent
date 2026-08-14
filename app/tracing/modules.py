"""The single source of truth for module names.

The spec requires that sub-module names be IDENTICAL across three surfaces: the
architecture diagram, the `steps` array of every `/api/execute` response, and
every description we publish. Three hand-maintained lists would drift the first
time one was renamed, and the drift would be invisible until a grader compared
them.

So they are defined once, here, and every surface reads from this module: the
tracer stamps `Module.*` onto each step, `/api/agent_info` describes them from
`MODULES`, and the diagram is generated from the same table rather than drawn by
hand. Renaming a module is one edit, and it cannot leave a surface behind.
"""

from __future__ import annotations

from typing import Final, NamedTuple


class Module(NamedTuple):
    """One named component of the agent, as it appears everywhere."""

    name: str
    role: str
    calls_llm: bool


FRONT_DOOR: Final = Module(
    "FrontDoor",
    "Receives the raw prompt, resolves which student it concerns, and seeds the "
    "loop's opening facts. Deterministic — no model call.",
    calls_llm=False,
)

REASONING_LOOP: Final = Module(
    "ReasoningLoop",
    "The agent's thinking core. Each turn it reads the facts derived so far and "
    "decides the next move: call a tool, or answer. This is the only module that "
    "chooses what the agent does next.",
    calls_llm=True,
)

FACT_DISPATCH: Final = Module(
    "FactDispatch",
    "Executes the tool calls the ReasoningLoop requests against the data sources "
    "and admits the results as typed, provenance-tagged facts. Deterministic — no "
    "model call.",
    calls_llm=False,
)

INTERPRETER: Final = Module(
    "Interpreter",
    "Reads a single value out of a retrieved knowledge-base passage, and returns "
    "the quote it came from so the value can be verified against its source.",
    calls_llm=True,
)

LIST_INTERPRETER: Final = Module(
    "ListInterpreter",
    "The plural of Interpreter: extracts every listed value from a passage, each "
    "with its own supporting quote, so an invented entry is caught per element.",
    calls_llm=True,
)

ANSWER_BOUNDARY: Final = Module(
    "AnswerBoundary",
    "Refuses any answer containing a number the agent did not derive from a fact. "
    "This is the grounding guarantee, enforced in code rather than requested in a "
    "prompt. Deterministic — no model call.",
    calls_llm=False,
)

ANSWER_VERIFY: Final = Module(
    "AnswerVerify",
    "Replays a finished answer's own numbers against deterministic post-conditions "
    "— no impossible grade, no out-of-range GPA, and a plan's minimums must hold "
    "when its courses are taken together. Catches the answer that is correctly "
    "sourced but still wrong. Deterministic — no model call.",
    calls_llm=False,
)

MODULES: Final[tuple[Module, ...]] = (
    FRONT_DOOR,
    REASONING_LOOP,
    FACT_DISPATCH,
    INTERPRETER,
    LIST_INTERPRETER,
    ANSWER_BOUNDARY,
    ANSWER_VERIFY,
)

MODULE_NAMES: Final[frozenset[str]] = frozenset(module.name for module in MODULES)

__all__ = [
    "ANSWER_BOUNDARY",
    "ANSWER_VERIFY",
    "FACT_DISPATCH",
    "FRONT_DOOR",
    "INTERPRETER",
    "LIST_INTERPRETER",
    "MODULES",
    "MODULE_NAMES",
    "REASONING_LOOP",
    "Module",
]
