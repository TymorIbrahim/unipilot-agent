"""Capture the exact Hebrew the model uses for a verdict, before guarding it.

Two checks are still English-only: `check_eligibility_is_not_self_contradictory`
(which caught "you meet 0 of 1 prerequisite groups, so you are eligible") and
`check_claimed_pass_is_on_the_transcript`. Both are dark in Hebrew for the same
reason the period checks were.

Writing Hebrew regexes for them from imagination is the mistake this repo keeps
paying for -- a guard that looks right, matches nothing the model actually
produces, and never fires. So this asks the questions that FORCE those two
shapes and prints the wording verbatim. The regex comes after the data.

    python evaluation/probe_hebrew_verdicts.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from challenge import execute  # noqa: E402

OUT = Path(__file__).parent / "probe_hebrew_verdicts.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"

PROBES = [
    # Forces an eligibility verdict plus a prerequisite-group count -- the two
    # halves whose contradiction the English guard exists to catch.
    ("eligible_he", "האם אני יכול לקחת את הקורס 01040174 בסמסטר הבא?"),
    # Forces a claim about a course already on the transcript.
    ("passed_he", "עברתי את הקורס 00940412?"),
    # An eligibility NO, which must name what would turn it into a yes.
    ("blocked_he", "מה חוסם אותי מלקחת את 00970800?"),
]


async def main() -> None:
    rows = []
    async with httpx.AsyncClient() as client:
        for name, prompt in PROBES:
            result = await execute(client, prompt, SENIOR)
            answer = str(result.get("response") or result.get("error") or "")
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}")
            print(f"  asked : {prompt}")
            print(f"  ANSWER: {answer}")
            rows.append({"name": name, "prompt": prompt, "answer": answer,
                         "status": result.get("status")})
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
