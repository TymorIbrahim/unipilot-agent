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

`POST /api/execute` takes `{"prompt": "..."}` and optionally `{"student_id": "..."}`. When no
student is named it answers for the primary demo student, so the bare `{"prompt": ...}` form
works on its own.

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

Open <http://localhost:8000/api/health> to check configuration, and serve `public/index.html`
for the UI.

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
