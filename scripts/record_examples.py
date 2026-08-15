"""Record `prompt_examples` for `/api/agent_info` from real agent runs.

The spec wants each example to carry its `prompt`, its `full_response` and its
`steps`. Those are the three fields a reader cannot check without running the
agent themselves, which is exactly why they are RECORDED here rather than
written by hand: a hand-composed trace would be a description of how the agent
works, and the point of publishing one is that it is evidence.

    ./.venv/bin/python scripts/record_examples.py

Writes `data/prompt_examples.json`, which `app/agent_info.py` serves. Costs real
model calls -- one full agent run per example -- so it is a deliberate,
committed act, not something a test or a deploy re-runs.

Recorded against whichever provider `.env` points chat at. Development uses
OpenAI directly on the same model, so building this set does not consume the
course's LLMod budget; the traces are identical in shape either way, because the
prompts and the loop are the same.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_PATH = REPO_ROOT / "data" / "prompt_examples.json"

# Chosen to span the CAN list in `agent_info.DESCRIPTION`, and to make the
# ARCHITECTURE legible from the traces alone: between them these exercise every
# module that appears in the diagram. A set of four similar questions would
# publish four copies of one code path.
QUESTIONS: tuple[dict[str, str], ...] = (
    {
        "label": "transcript arithmetic",
        # find -> compute: the simplest complete path, and the one that shows a
        # number being derived rather than recalled.
        "prompt": "How many credits have I completed so far?",
    },
    {
        "label": "degree progress",
        # Joins the student's record to their degree programme -- two sources,
        # one subtraction.
        "prompt": "What is the total credit requirement for my degree, and how many do I still need?",
    },
    {
        "label": "knowledge base",
        # The only path that reaches the Interpreter module: the answer is not in
        # any table, it is prose in the regulations, and it comes back with the
        # sentence it was read from.
        "prompt": "What is the English language requirement I have to satisfy to graduate?",
    },
    {
        "label": "eligibility",
        # Prerequisites, so `traverse` over the materialised edge table.
        "prompt": "Am I eligible to take 00960211, and what does it require?",
    },
)


async def record(questions: tuple[dict[str, str], ...], student_id: str | None) -> list[dict[str, Any]]:
    from app.runner import run_agent

    examples: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        started = time.time()
        print(f"[{index}/{len(questions)}] {question['label']}: {question['prompt']}", flush=True)
        result = await run_agent(question["prompt"], student_id=student_id)
        elapsed = time.time() - started

        if not result.ok:
            # Recorded failures would publish a broken agent as documentation.
            # Loud, and the whole set is abandoned rather than partially written.
            raise SystemExit(
                f"  run failed after {elapsed:.1f}s: {result.error}\n"
                "  Nothing written. Fix the run before recording examples."
            )

        print(
            f"  ok in {elapsed:.1f}s, {len(result.steps)} steps, "
            f"modules: {', '.join(dict.fromkeys(step['module'] for step in result.steps))}",
            flush=True,
        )
        examples.append(
            {
                "prompt": question["prompt"],
                "full_response": result.answer,
                "steps": result.steps,
            }
        )
    return examples


async def main_async(arguments: argparse.Namespace) -> int:
    from app.config import get_settings

    settings = get_settings()
    print(f"provider={settings.chat_provider()} model={settings.llm_chat_model}")
    if settings.submission_ready():
        # Recording is a handful of full runs. Doing it on the graded budget
        # when an identical dev provider is configured is money set on fire.
        print("  NOTE: chat is pointed at LLMod -- this will spend the course budget.")

    selected = QUESTIONS if not arguments.only else tuple(
        q for q in QUESTIONS if q["label"] in arguments.only
    )
    if not selected:
        raise SystemExit(f"no questions matched --only {arguments.only}")

    examples = await record(selected, arguments.student)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(examples, ensure_ascii=False, indent=2)
    OUTPUT_PATH.write_text(payload, encoding="utf-8")
    print(
        f"\nwrote {OUTPUT_PATH.relative_to(REPO_ROOT)}: {len(examples)} examples, "
        f"{len(payload) / 1e6:.2f} MB"
    )

    from app.db.postgres import close_pool

    await close_pool()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", default=None, help="student id (default: the primary demo student)")
    parser.add_argument("--only", nargs="*", default=None, help="record only these labels")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
