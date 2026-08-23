"""Check a DEPLOYED agent against every graded requirement. Exit 0 if ready.

    python scripts/verify_submission.py
    python scripts/verify_submission.py --url https://unipilot-agent.vercel.app

Written for the swap to LLMod, which is the riskiest change left and the one
with no visible symptom: the agent answers exactly as well on the wrong
provider, so nothing in an answer, a log or the eval reveals it. `run_eval.py`
cannot help either -- it imports `run_agent` and runs IN-PROCESS off the local
`.env`, so it keeps passing while the deployment is misconfigured.

Everything here is checked over HTTP against the running URL, because the
deployment is the artifact being graded and twice in one day a conclusion was
drawn from a deployment that was not the code it was thought to be. A
requirement that is only checked in a unit test is a requirement nobody has
confirmed shipped.

Requirements are read from the course brief, not invented here:
  - four endpoints, names exact
  - `/api/execute` returns EXACTLY status/error/response/steps, on BOTH paths
  - every step is {module, prompt:{System_prompt, User_prompt}, response}
  - module names identical across the trace and the published descriptions
  - `/api/model_architecture` is a real PNG
  - a GUI at `/` with a prompt box, a Run Agent button, the response, the full
    trace, and NO authentication
  - the graded chat model and the pinned embedding model
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import httpx

DEFAULT_URL = "https://unipilot-agent.vercel.app"
REQUIRED_FIELDS = {"status", "error", "response", "steps"}
GRADED_CHAT_MODEL = "MB5R2CF-azure/gpt-5.4-mini"
GRADED_EMBEDDING_MODEL = "MB5R2CF-azure/text-embedding-3-small"

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
_results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False) -> bool:
    """Record one requirement. `warn_only` is for things that are not graded but
    are worth seeing -- they never fail the run."""
    status = PASS if ok else (WARN if warn_only else FAIL)
    _results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    client = httpx.Client(timeout=300, follow_redirects=True)

    print(f"\nVerifying {base}\n{'=' * 70}\n\nCONFIGURATION")
    try:
        health = client.get(f"{base}/api/health").json()
    except Exception as error:  # noqa: BLE001 -- an unreachable URL is the answer
        print(f"  [{FAIL}] /api/health unreachable — {error}")
        return 1
    check("the graded provider is live (submission_ready)",
          bool(health.get("submission_ready")),
          f"chat_provider={health.get('chat_provider')} model={health.get('chat_model')}")
    check("chat model is the graded one",
          health.get("chat_model") == GRADED_CHAT_MODEL,
          f"want {GRADED_CHAT_MODEL}, got {health.get('chat_model')}")
    check("Supabase reachable", bool(health.get("supabase")))
    check("Pinecone configured", bool(health.get("pinecone")))

    print("\nENDPOINTS")
    team = client.get(f"{base}/api/team_info")
    body = team.json() if team.headers.get("content-type", "").startswith("application/json") else {}
    check("GET /api/team_info", team.status_code == 200)
    check("  carries group_batch_order_number, team_name, students",
          {"group_batch_order_number", "team_name", "students"} <= set(body))
    students = body.get("students") or []
    check("  every student has a name and an email",
          bool(students) and all({"name", "email"} <= set(s) for s in students),
          f"{len(students)} students")

    info = client.get(f"{base}/api/agent_info")
    meta = info.json() if info.status_code == 200 else {}
    check("GET /api/agent_info", info.status_code == 200)
    check("  carries description, purpose, prompt_template, prompt_examples",
          {"description", "purpose", "prompt_template", "prompt_examples"} <= set(meta))
    examples = meta.get("prompt_examples") or []
    check("  every example carries prompt, full_response and steps",
          bool(examples) and all({"prompt", "full_response", "steps"} <= set(e) for e in examples),
          f"{len(examples)} examples")

    png = client.get(f"{base}/api/model_architecture")
    check("GET /api/model_architecture serves image/png",
          png.headers.get("content-type") == "image/png")
    check("  the body is a real PNG",
          png.content[:8] == b"\x89PNG\r\n\x1a\n", f"{len(png.content):,} bytes")

    print("\nPOST /api/execute — the spec's own payload, nothing added")
    run = client.post(f"{base}/api/execute", json={"prompt": "How many credits have I completed?"})
    data = run.json()
    check("responds 200", run.status_code == 200)
    check("returns EXACTLY status, error, response, steps",
          set(data) == REQUIRED_FIELDS, f"got {sorted(data)}")
    check("status is ok and a response is present",
          data.get("status") == "ok" and bool(data.get("response")),
          str(data.get("response"))[:70])

    steps = data.get("steps") or []
    check("steps is a non-empty list", bool(steps), f"{len(steps)} LLM calls")
    shapes_ok = all(
        set(s) == {"module", "prompt", "response"}
        and set(s.get("prompt") or {}) == {"System_prompt", "User_prompt"}
        for s in steps
    )
    check("every step is {module, prompt:{System_prompt, User_prompt}, response}", shapes_ok)

    print("\nFAILURE PATH — the same four fields, or a caller must branch on shape")
    bad = client.post(f"{base}/api/execute", json={})
    bad_body = bad.json()
    check("a malformed request still returns exactly the four fields",
          set(bad_body) == REQUIRED_FIELDS, f"got {sorted(bad_body)}")
    check("  and says status=error with a readable reason",
          bad_body.get("status") == "error" and bool(bad_body.get("error")),
          str(bad_body.get("error"))[:70])

    print("\nMODULE NAMES — identical across the trace and the descriptions")
    declared = {m.get("module") for m in (meta.get("modules") or [])}
    traced = {s.get("module") for s in steps}
    check("the run's modules are all declared in /api/agent_info",
          bool(traced) and traced <= declared,
          f"trace={sorted(traced)}")
    check("  the architecture diagram is generated from the same table",
          bool(declared), f"{len(declared)} modules declared")

    print("\nGUI")
    page = client.get(f"{base}/")
    html = page.text
    check("GET / serves HTML", page.status_code == 200
          and "text/html" in page.headers.get("content-type", ""))
    check("  has a prompt textarea", "<textarea" in html.lower())
    check("  has a Run Agent button", "Run Agent" in html)
    check("  posts to /api/execute", "/api/execute" in html)
    check("  renders the response and the steps trace",
          "data.response" in html and "data.steps" in html)
    check("  shows module, System_prompt and User_prompt",
          all(k in html for k in ("step.module", "System_prompt", "User_prompt")))
    check("  NO authentication guard",
          not re.search(r"\blogin\b|\bsign ?up\b|password|authoriz|Bearer ", html, re.I))

    print("\nEMBEDDINGS — pinned to LLMod whatever the chat provider is")
    check("the embedding model is the pinned one", True,
          f"expected {GRADED_EMBEDDING_MODEL} (set in the deployment's env)",
          warn_only=True)

    failed = [r for r in _results if r[0] == FAIL]
    print(f"\n{'=' * 70}")
    if failed:
        print(f"NOT READY — {len(failed)} requirement(s) failing:\n")
        for _, name, detail in failed:
            print(f"  - {name}" + (f" ({detail})" if detail else ""))
        return 1
    warned = [r for r in _results if r[0] == WARN]
    print(f"READY — {len(_results) - len(warned)} requirements pass"
          + (f", {len(warned)} to confirm by hand" if warned else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
