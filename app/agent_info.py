"""`GET /api/agent_info` -- what the agent is, and how to drive it.

Written to be read by someone who has never seen the agent and has one screen to
decide what to type. That means the CAN and CANNOT lists are as prominent as the
description: the fastest way to get a useless answer from a grounded agent is to
ask it something it has no facts for, and saying so up front prevents that.

`prompt_examples` are recorded from real runs, never written by hand. An example
whose `full_response` and `steps` were composed by a human would misrepresent
both what the agent says and how it gets there -- and `steps` in particular is
the one field a reader cannot verify without running the agent themselves.
"""

from __future__ import annotations

from typing import Any

from app.tracing.modules import MODULES

DESCRIPTION = (
    "A grounded academic advisor for Technion students. It answers questions about a "
    "student's own degree progress -- what they have completed, what remains, their GPA, "
    "which courses they are eligible for, and what a workable next semester looks like -- "
    "by deriving every fact from the student's record and the Technion course catalog.\n\n"
    "What it CAN do: report degree progress and remaining requirements; compute GPA and "
    "credit totals; explain what a course requires and whether the student is eligible; "
    "read a degree's structure out of the Technion knowledge base; build a conflict-free "
    "plan for an upcoming semester from courses actually offered that term; and work out "
    "what-if scenarios such as the minimum grades needed to reach a target GPA.\n\n"
    "What it CANNOT do (constraints): it never writes to any record -- it does not register "
    "for courses, submit work, or change a transcript; it answers only about the students in "
    "its dataset, not arbitrary people; it will not invent a number it did not derive, and "
    "will say it could not determine something rather than estimate it; and it declines "
    "questions outside academic advising instead of guessing at them.\n\n"
    "Its defining property is that every number in an answer is traceable to a fact the "
    "agent derived. A number the agent merely believes is refused in code before the answer "
    "can be returned, so a confident-sounding fabrication is not a failure mode it has."
)

PURPOSE = (
    "Replace the slow, error-prone work of reading a degree catalog against your own "
    "transcript -- turning a question like 'what do I still need, and what should I take "
    "next semester?' into a grounded answer you can check line by line."
)

PROMPT_TEMPLATE = (
    "Ask a question about your degree in plain language. Optionally frame it with:\n"
    "Question: <what you want to know>\n"
    "Context: <anything the agent should assume, e.g. 'I want to finish by winter 2027'>\n"
    "Constraint: <any limit, e.g. 'no more than 18 credits' or 'mornings only'>"
)

PROMPT_TEMPLATE_EXAMPLE = (
    "Question: which courses should I take next semester?\n"
    "Context: I want to finish my degree as early as possible\n"
    "Constraint: no more than 18 credits"
)

# Filled from real recorded runs once the reasoning core is wired and deployed.
# Deliberately empty rather than fabricated -- see the module docstring.
PROMPT_EXAMPLES: list[dict[str, Any]] = []


def agent_info() -> dict[str, Any]:
    """The exact response body the spec requires, plus the module glossary.

    `modules` is additive -- the spec neither requires nor forbids it -- and it
    earns its place by making the names in `steps` self-describing, so a reader
    inspecting a trace does not have to consult the architecture diagram to
    learn what `AnswerBoundary` is.
    """
    return {
        "description": DESCRIPTION,
        "purpose": PURPOSE,
        "prompt_template": {
            "template": PROMPT_TEMPLATE,
            "example": PROMPT_TEMPLATE_EXAMPLE,
        },
        "prompt_examples": PROMPT_EXAMPLES,
        "modules": [
            {"module": module.name, "role": module.role, "calls_llm": module.calls_llm}
            for module in MODULES
        ],
    }


__all__ = [
    "DESCRIPTION",
    "PROMPT_EXAMPLES",
    "PROMPT_TEMPLATE",
    "PROMPT_TEMPLATE_EXAMPLE",
    "PURPOSE",
    "agent_info",
]
