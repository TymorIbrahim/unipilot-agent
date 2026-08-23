# UniPilot Agent

A grounded academic-advising AI agent for Technion students. It answers questions about a
student's degree — progress, remaining requirements, GPA, eligibility, and what a workable
next semester looks like — by **deriving every fact** from the student's record and the
Technion course catalog.

Its defining property: **a number the agent did not derive cannot appear in an answer.**
That rule is enforced in code, not requested in a prompt, so a confident fabrication is not
a failure mode this agent has.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/team_info` | Team and student details |
| `GET` | `/api/agent_info` | Description, purpose, prompt template, worked examples |
| `GET` | `/api/model_architecture` | Architecture diagram (`image/png`) |
| `POST` | `/api/execute` | `{status, error, response, steps}` |
| `GET` | `/api/health` | Configuration readiness |

`POST /api/execute` takes `{"prompt": "..."}` and optionally `{"student_id": "..."}` and
`{"conversation_id": "..."}`. When no student is named it answers for the primary demo
student, so the bare `{"prompt": ...}` form works on its own. A `conversation_id` threads
follow-ups: the prior exchanges are loaded so "how many credits is that in total?" resolves,
while the FACTS are re-derived every run, so an answer is always grounded in live records
rather than a snapshot.

The same four fields come back on success and on failure, so a caller never has to branch on
shape — only on `status`.

Questions can be asked in **English or Hebrew**, and the answer follows the question's
language, including the course names.

**`steps`** is the full trace: every LLM call the agent made, in order, each with the module
that made it, the exact system and user prompts it sent, and the raw reply it got back. The
module names in `steps` match the architecture diagram exactly — both are generated from
[`app/tracing/modules.py`](app/tracing/modules.py), so they cannot drift apart.

The web UI at `/` runs the agent and renders both the answer and the full trace. It has no
login.

## Architecture

| Module | Role | LLM? |
|---|---|---|
| `FrontDoor` | Resolves which student the prompt concerns, seeds the opening facts | — |
| `ReasoningLoop` | The thinking core: reads facts so far, decides the next move | ✅ |
| `FactDispatch` | Executes tool calls, admits results as typed, provenance-tagged facts | — |
| `Interpreter` | Reads one value out of a knowledge-base passage, with its supporting quote | ✅ |
| `ListInterpreter` | Extracts every listed value from a passage, each with its own quote | ✅ |
| `AnswerBoundary` | Refuses any answer containing an underived number | — |
| `AnswerVerify` | Replays the answer's numbers against deterministic post-conditions | — |

Four of the seven modules make **no LLM call at all**. That is the point: reasoning is
expensive and fallible, so everything that can be decided deterministically is.

## Data

| Store | Holds |
|---|---|
| **Supabase** | Student records, course catalog, offerings, degree programs, and the precomputed curriculum/prerequisite graph |
| **Pinecone** | Wiki chunk embeddings for semantic retrieval |
| Repo artifact | Precomputed chunk text + BM25 statistics for lexical retrieval |

Nothing is derived at request time that could be derived at build time — a serverless
function has no warm process to amortise that work against. [`scripts/seed.py`](scripts/) is
the single reproducible job that builds all three.

## Running locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in the credentials
uvicorn app.main:app --reload --port 8000
```

Then open <http://localhost:8000/> for the UI and <http://localhost:8000/api/health> to check
configuration. The app serves the interface itself, so the development and production
environments behave the same — in production Vercel's CDN answers `/` first, and locally
FastAPI does.

### Models

Both are served by LLMod, and the embedding model is pinned regardless of the chat provider
because the Pinecone index was built with it — querying that index with a different embedder
returns results that are meaningless rather than absent.

| Role | Model |
|---|---|
| Chat | `MB5R2CF-azure/gpt-5.4-mini` |
| Embedding | `MB5R2CF-azure/text-embedding-3-small` |

`GET /api/health` reports `chat_provider` and `submission_ready`, because running on the
wrong provider has no other symptom: the agent answers just as well and only the billing
shows it.

## Evaluation

The agent is scored against answers derived from the data — by SQL and by reading the
corpus — never by asking the agent. Comparing runs to each other proves only consistency,
and this agent was once consistently wrong about credits across five identical runs.

```bash
python evaluation/run_eval.py           # 9 questions x 3 repeats, vs evaluation/ground_truth.json
python evaluation/cross_student.py      # the core questions for EVERY student, truth from SQL
python evaluation/challenge_policy.py   # the regulations path (search -> interpret -> cite)
python evaluation/challenge_hebrew.py   # the same questions asked in Hebrew
python scripts/verify_submission.py     # every graded requirement, over HTTP, against the live URL
```

`run_eval.py` runs the agent **in process**, so it measures the code and reads the local
`.env`. `verify_submission.py` is the only one that tests the deployed artifact, which is
why it is the one to trust about configuration.

`cross_student.py` derives each expected figure from Postgres at run time rather than storing
it, so it cannot go stale when the data is re-seeded.

## Deployment

Deployed on Vercel. `vercel.json` routes `/api/*` to the ASGI app in `api/index.py` and
everything else to the static UI, with `maxDuration` set to the platform's 300s ceiling.

The agent enforces its **own** 240s budget below that. On expiry it ships the grounded
partial answer it has rather than being killed mid-request — a partial honest answer is
worth something, a timeout is worth nothing.

## Relationship to UniPilot

The reasoning core was extracted from [UniPilot](https://github.com/TymorIbrahim/UniPilot),
a larger academic-planning platform, and adapted to run standalone: Supabase in place of
MongoDB, no authentication, and a self-contained HTTP surface. This repository is
independent — it shares no running infrastructure with UniPilot.
