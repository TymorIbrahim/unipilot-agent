# UniPilot Agent — Claude Code Instructions

## What this repo is

A **standalone** grounded academic-advising agent, submitted as a course final project
(**due 2026-08-23**). The reasoning core was extracted from
[UniPilot](https://github.com/TymorIbrahim/UniPilot); this repo is independent and shares no
running infrastructure with it.

The full brief — required endpoints, locked decisions, and the measured constraints — is in
the `agent-course-submission` memory. Read it before planning work.

## Non-negotiables from the course spec

These are graded and are **not** ours to improve:

- Four endpoints, names exact: `GET /api/team_info`, `GET /api/agent_info`,
  `GET /api/model_architecture`, `POST /api/execute`.
- `POST /api/execute` returns exactly four top-level fields — `status`, `error`, `response`,
  `steps` — on **both** the success and failure paths.
- `steps` lists **every LLM call in order**, each `{module, prompt: {System_prompt,
  User_prompt}, response}`.
- Module names must be **identical** across the architecture diagram, the steps log, and every
  description. They live in `app/tracing/modules.py` — change them there, never inline.
- The GUI at `/` must have **no authentication**.
- Efficiency is graded: avoid unnecessary LLM calls, keep prompts minimal.

## Cost discipline

The course budget is **$13 total** on LLMod.ai.

- **Development runs against OpenAI directly** on the same `gpt-5.4-mini` model, so iteration
  does not consume the graded budget. `.env` holds both; the swap is three lines.
- **Embeddings stay pinned to LLMod** regardless of the chat provider. The Pinecone index was
  built with `MB5R2CF-azure/text-embedding-3-small`; querying it with a different model returns
  results that are meaningless rather than absent.
- Before submitting, check `GET /api/health` → `submission_ready`. Running on the wrong
  provider has no visible symptom.
- Never run a paid batch of calls without explicit approval. State the expected cost first.

## Working in this repo

- Run the tests for whatever you touched before reporting a task done. Type checks are not
  tests.
- `pytest` runs the fast unit suite in about a second. Two markers are deselected from it:
  `-m live` makes paid LLM calls, and `-m supabase` queries the real database (~45s).
  **Run `pytest -m supabase` before submitting** — it is the only check that every advertised
  tool can actually be fed by a route the model can walk, and that failure mode is invisible
  to unit tests by construction.
- A passing unit suite is weak evidence here. Four of the five tools audited on 2026-08-15 —
  `plan_term`, `forecast`, `interpret`/`extract_list`, `propose` — were wrong while fully
  green, because the tests encoded the same wrong assumption as the code. Check a claim
  against the DATA (SQL, the corpus) rather than against another part of this system.
- `evaluation/ground_truth.json` holds the correct answer to each evaluation question, derived
  from the data rather than from the agent. `python evaluation/run_eval.py` scores against it.
  Comparing runs to each other only proves consistency — the agent answered "135 credits"
  identically five times and was wrong every time.
- Prefer porting from UniPilot's `services/ai` over writing from scratch — that code is
  live-validated and its failure modes were expensive to find. Port the design; only the
  I/O layer should change.
- Don't refactor beyond what the task requires.
- `.env` is gitignored and holds live credentials. Never commit it, never print a key.

## Deployment

Vercel serverless. The platform kills any call at **300s**; the agent enforces its own **240s**
budget and ships a grounded partial answer rather than being killed mid-request.

Bundle size is cold-start latency — keep `requirements.txt` lean. `networkx` and
`motor`/`pymongo` were deliberately dropped: graph derivations are materialized into Supabase
at seed time, and Supabase replaced MongoDB.
