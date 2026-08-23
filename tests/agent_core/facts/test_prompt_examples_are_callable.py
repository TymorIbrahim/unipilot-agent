"""Every tool call the system prompt SHOWS must be one the dispatcher accepts.

The prompt told the model to derive a graduation timeline with "exactly this --
{"ceil_div": [{"fact": "credits_needed"}, {"fact": "max_credits_per_semester"}]}"
and that is not a call `compute` can take: the expression belongs inside a NAMED
pipeline. So the model copied the documented form, got "`compute` takes a LIST
of named pipelines, and this call has none", rewrote it correctly, and answered
on the third turn. Two wasted turns on every graduation question, and the model
was following instructions exactly.

This is the worst kind of defect in this system because it is invisible from
both ends: the prompt reads as correct, the dispatcher's error is correct, and
the run still answers, just slower. It only shows up as a wasted-turn count
nobody has attributed.

A prompt example is API documentation, and untested documentation drifts.
"""

from __future__ import annotations

import json
import re

from app.agent_core.facts.adapter import SYSTEM_PROMPT

# The arithmetic heads that may only ever appear inside a pipeline's `value`.
_ARITH = ("ceil_div", "add", "subtract", "multiply", "divide", "max", "min")


def _json_objects(text: str) -> list[str]:
    """Every balanced {...} run in the prompt, longest-first per start offset.

    The prompt's examples are hand-written JSON inside prose, so they cannot be
    found by parsing the whole document -- only by scanning for balanced braces
    and trying each candidate.
    """
    found = []
    for start in (m.start() for m in re.finditer(r"\{", text)):
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    found.append(text[start:index + 1])
                    break
    return found


def _parsed_examples() -> list[dict]:
    out = []
    for blob in _json_objects(SYSTEM_PROMPT):
        try:
            value = json.loads(blob)
        except ValueError:
            continue  # prose, or a {fact_name} slot -- not an example
        if isinstance(value, dict):
            out.append(value)
    return out


class TestTheComputeExamplesAreCallable:
    def test_the_prompt_contains_a_compute_example_at_all(self) -> None:
        """If this fails the rest is vacuous -- the scanner stopped finding them."""
        assert any(
            "pipelines" in json.dumps(example) for example in _parsed_examples()
        ), "no `pipelines` example found in SYSTEM_PROMPT"

    def test_no_arithmetic_head_sits_directly_in_args(self) -> None:
        """`{"ceil_div": [...]}` as a whole args object is the shape that failed.

        Scoped to objects that actually name a `tool`. The brace scanner also
        returns every NESTED fragment, and an arithmetic head is exactly what a
        pipeline's `value` is supposed to be -- checking those too failed on the
        corrected example, which is the test being wrong rather than the prompt.
        """
        for example in _parsed_examples():
            if "tool" not in example:
                continue
            args = example.get("args")
            if not isinstance(args, dict):
                continue
            bare = [op for op in _ARITH if op in args]
            assert not bare, (
                f"SYSTEM_PROMPT shows {bare} at the top of a `compute` args object. "
                "`compute` takes {'pipelines': [{'name': ..., 'value': {...}}]}, so "
                "the model copies this, fails, and spends a turn recovering. "
                f"Offending example: {json.dumps(example)[:200]}"
            )

    def test_every_pipeline_example_has_a_name_and_a_value(self) -> None:
        for example in _parsed_examples():
            pipelines = (example.get("args") or example).get("pipelines")
            if not isinstance(pipelines, list):
                continue
            for pipeline in pipelines:
                if not isinstance(pipeline, dict):
                    continue
                assert "name" in pipeline, f"pipeline without a name: {pipeline}"
                assert "value" in pipeline or "source" in pipeline, (
                    f"pipeline with neither `value` nor `source`: {pipeline}"
                )


class TestTheGraduationRecipeIsTheCallableOne:
    """The specific recipe the wasted turns were traced to."""

    def test_ceil_div_is_shown_wrapped_in_a_pipeline(self) -> None:
        """The recipe must appear, and appear as a call that would succeed."""
        assert "ceil_div" in SYSTEM_PROMPT, "the graduation recipe is gone entirely"

        wrapped = [
            example
            for example in _parsed_examples()
            if "ceil_div" in json.dumps(example)
            and isinstance((example.get("args") or example).get("pipelines"), list)
        ]
        assert wrapped, (
            "no ceil_div example in SYSTEM_PROMPT is wrapped in `pipelines`. The "
            "model copies the documented form verbatim, so a bare "
            '{"ceil_div": [...]} costs a turn on every graduation question.'
        )
