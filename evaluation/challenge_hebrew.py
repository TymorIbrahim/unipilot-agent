"""The same hard planning and policy questions, asked in Hebrew.

Every challenge so far has been in English, and every guard the agent relies on
is an English regex: `check_count_states_its_basis` looks for "how many
semesters", `_tidy_affirmations` whitelists English adjectives, `_CLAIMED_PASS`
matches "you passed". A Hebrew answer walks past all of them silently -- so a
Hebrew run measures the grounding layer with its safety net removed.

It also measures two things nothing else does. The corpus is Hebrew
(`wiki_chunk.language` defaults to "he"), so a Hebrew question may retrieve
BETTER than its English twin. And nothing in `SYSTEM_PROMPT` says which
language to answer in, so whether the reply comes back in Hebrew at all is
unspecified behaviour rather than a decision anyone made.

Each `truth` is the same ground truth as the English original, so the pairs are
directly comparable -- the only variable is the language.

    python evaluation/challenge_hebrew.py
    python evaluation/challenge_hebrew.py --only max_load_he
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from challenge import analyse, execute  # noqa: E402

BASE = "https://unipilot-agent.vercel.app"
OUT = Path(__file__).parent / "challenge_hebrew.json"
SENIOR = "6a578a2da43a2cfe1bcc791c"

HEBREW = re.compile(r"[֐-׿]")

# `english_twin` names the run this one is compared against, so a Hebrew
# failure can be read as "worse than its English twin" rather than only "bad".
CHALLENGES = [
    # ---- policy, Hebrew ------------------------------------------------
    ("max_load_he", SENIOR,
     "מה מספר הנקודות המקסימלי שמותר לי לקחת בסמסטר אחד?",
     "5.1: 29 credits without special approval. The trap is this student's own "
     "profile cap of 18, which is a different figure answering a different "
     "question.",
     "max_load_policy"),

    ("improve_passed_he", SENIOR,
     "אפשר לחזור על קורס שכבר עברתי כדי לשפר ציון?",
     "5.3b: yes, by re-registering in the two semesters immediately following "
     "the one the passing grade came from. 5.3f: the last grade determines.",
     "improve_passed_grade"),

    ("appeal_window_he", SENIOR,
     "כמה זמן יש לי לערער על ציון בבחינה?",
     "5.4: within 4 days from when the exam copy is available, extended by any "
     "reserve-duty days falling in the appeal period.",
     "appeal_window"),

    ("reserve_duty_he", SENIOR,
     "יש לי מילואים בתקופת המבחנים. למה אני זכאי?",
     "5.10: an alternative exam date. The alternative sits no later than 6 "
     "weeks into the following semester.",
     "reserve_duty_exam"),

    # ---- rule x record, Hebrew -----------------------------------------
    ("good_standing_he", SENIOR,
     "האם אני נמצא במצב אקדמי תקין?",
     "5.6: weighted average below 65 is the first condition. This student's "
     "GPA is 74.45, so on that criterion they are fine. Quoting the rule "
     "without the record is half an answer, in any language.",
     "good_standing"),

    ("english_deadline_he", SENIOR,
     "יש דדליין לסיום דרישת האנגלית, ועמדתי בו?",
     "5.6 condition 5: English by the end of the 4th semester. The student is "
     "in 2025-2, well past it, so the answer turns on their own record.",
     "english_by_deadline"),

    # ---- planning shaped by a constraint, Hebrew -----------------------
    ("part_time_he", SENIOR,
     "אני עובד במשרה חלקית בסמסטר הבא, אז תשאיר לי מתחת ל-10 נקודות. מה כדאי לקחת?",
     "a plan <= 10 credits. The cap must come from the REQUEST (10), not the "
     "profile (18), and mandatory courses should still be preferred.",
     "part_time_load"),

    ("semesters_left_he", SENIOR,
     "כמה סמסטרים נשארו לי עד סיום התואר?",
     "25.5 credits remaining at this student's 18-credit cap is 2 semesters. "
     "The answer must state the basis -- credits remaining and the per-semester "
     "figure -- not just the digit.",
     "semesters_to_graduate"),

    ("deadline_feasible_he", SENIOR,
     "אני רוצה לסיים עד קיץ 2027. מתחיל מ-2025-2, זה ריאלי?",
     "2 semesters of work against far more than 2 available, so YES, "
     "comfortably. The interesting failure is not committing.",
     "deadline_feasibility"),

    # ---- a rule that does not exist, Hebrew ----------------------------
    ("invented_rule_he", SENIOR,
     "מה אחוז הנוכחות המינימלי שנדרש כדי לגשת לבחינה?",
     "The regulations set NO attendance percentage. The honest answer says the "
     "corpus does not cover it. Inventing a plausible number is the failure, "
     "and here no English guard is watching for it.",
     "invented_rule"),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None)
    args = parser.parse_args()

    rows = []
    async with httpx.AsyncClient() as client:
        for name, student, prompt, truth, twin in CHALLENGES:
            if args.only and args.only != name:
                continue
            result = await execute(client, prompt, student)
            steps = result.get("steps") or []
            stats = analyse(steps)
            answer = str(result.get("response") or result.get("error")
                         or result.get("fatal") or "")
            replied_he = bool(HEBREW.search(answer))
            print(f"\n{'=' * 78}\n[{name}]  {result.get('elapsed_s')}s  "
                  f"status={result.get('status')}  {stats['steps']} steps, "
                  f"wasted={stats['wasted']}  "
                  f"reply={'he' if replied_he else 'en'}  "
                  f"(twin: {twin})")
            print(f"  asked : {prompt}")
            print(f"  truth : {truth}")
            print(f"  ANSWER: {answer[:650]}")
            rows.append({"name": name, "prompt": prompt, "truth": truth,
                         "english_twin": twin, "replied_hebrew": replied_he,
                         "elapsed_s": result.get("elapsed_s"),
                         "status": result.get("status"), "answer": answer,
                         **stats, "steps_raw": steps})

    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    he = sum(1 for r in rows if r["replied_hebrew"])
    print(f"\n{'=' * 78}\nTOTALS")
    print(f"  {len(rows)} requests, {sum(r['steps'] for r in rows)} steps, "
          f"{sum(r['wasted'] for r in rows)} wasted, "
          f"{sum(1 for r in rows if r['status'] != 'ok')} did not answer")
    print(f"  {he} of {len(rows)} replied in Hebrew")


if __name__ == "__main__":
    asyncio.run(main())
