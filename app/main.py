"""The agent's HTTP surface -- the four endpoints the course spec requires.

Names, methods and response shapes are fixed by the spec and are not ours to
improve. In particular `/api/execute` returns exactly four top-level fields
(`status`, `error`, `response`, `steps`) on BOTH the success and failure paths,
so a caller never has to branch on shape to read a result -- only on `status`.

That symmetry is also why no exception is allowed to escape this layer: an
unhandled error would let FastAPI answer with its own `{"detail": ...}` shape,
which satisfies nobody's contract. Every failure is caught and rendered as the
same four fields, with the cause in `error`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent_info import agent_info
from app.config import get_settings
from app.team import team_info

logger = logging.getLogger(__name__)

_IMPORTED_AT = time.monotonic()
_SERVED_A_REQUEST = False
"""When this process began, and whether it has answered anything yet.

The platform's clock starts when the REQUEST arrives, which on a cold start is
before this module finishes importing -- a 45MB bundle and a 4,895-chunk wiki
corpus, measured at ~13s (18.2s cold against 5.0s warm for the same question).
The loop's own budget used to start when the loop started, so it planned against
240s of a 300s ceiling while the caller was already 13 seconds into a 60s one.

Measured: the function LOGGED `outcome=answered elapsed=48.2s` and the caller
still got nothing, because 13 + 48 crossed the cap while the response was being
written. The work was done and thrown away.

So the first request in a process is charged for the import; later ones are not,
because for them the import already happened in someone else's window."""

_MAX_COLD_START_CHARGE_S = 20.0
"""The most a first request may be charged for its process's import.

Measured cold start is ~13s (18.2s cold against 5.0s warm), so 20 covers it with
room. The bound exists because "imported" and "first request arrives" are not the
same moment: the platform can initialise a function and route to it later, and
charging the whole gap spends the budget before the run starts. Live, that
returned "I ran out of time before I could finish that" after 9.3 seconds of a
45-second budget."""

_REPO_ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE_PNG = _REPO_ROOT / "data" / "architecture.png"
_PUBLIC_DIR = _REPO_ROOT / "public"
"""The GUI Vercel serves from its CDN in production, and the app serves itself
in development -- see the mount in `create_app`."""

router = APIRouter(prefix="/api")


class ExecuteRequest(BaseModel):
    prompt: str = Field(min_length=1)
    # Not in the spec's input contract, and deliberately optional: the spec's
    # `{prompt}` alone cannot say WHOSE academic record a question is about, and
    # this agent reasons over a specific student. The GUI supplies it from a
    # plain selector (the spec forbids auth guards, not a chooser); when it is
    # absent we fall back to the primary demo student so a bare `{"prompt": ...}`
    # still works exactly as the spec describes.
    student_id: str | None = None
    # Also optional, and for the same reason: the spec's `{prompt}` has no way
    # to say "this continues what we were just talking about". Without it a
    # follow-up like "yes, continue" has no referent, and the conversation
    # store -- which exists and is wired into the reasoning core -- was
    # unreachable from the one endpoint that could use it. Absent means a
    # one-shot request, exactly as the spec describes.
    conversation_id: str | None = None


def _execute_response(
    *,
    status: str,
    error: str | None,
    response: str | None,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """The one response shape, built in one place so the paths cannot diverge."""
    return {"status": status, "error": error, "response": response, "steps": steps}


@router.get("/team_info")
async def get_team_info() -> dict[str, Any]:
    return team_info()


@router.get("/agent_info")
async def get_agent_info() -> dict[str, Any]:
    return agent_info()


@router.get("/model_architecture")
async def get_model_architecture() -> Any:
    if not ARCHITECTURE_PNG.is_file():
        # A missing diagram is a build problem, not a client problem, and saying
        # so plainly beats serving a placeholder image that looks like an answer.
        return JSONResponse(
            status_code=503,
            content={"error": "architecture diagram has not been generated yet"},
        )
    return FileResponse(ARCHITECTURE_PNG, media_type="image/png")


@router.post("/execute")
async def post_execute(payload: ExecuteRequest) -> dict[str, Any]:
    global _SERVED_A_REQUEST
    # A cold process is still paying for its import when the request arrives, so
    # the window opened at `_IMPORTED_AT`; a warm one starts here.
    #
    # BOUNDED, because "imported" and "first request" are not the same moment.
    # The platform can initialise a function and route to it later, and charging
    # the whole gap spent the budget before the run began: a live request
    # returned "I ran out of time" after 9.3 seconds of a 45s budget.
    now = time.monotonic()
    started_at = max(_IMPORTED_AT, now - _MAX_COLD_START_CHARGE_S) if not _SERVED_A_REQUEST else now
    _SERVED_A_REQUEST = True

    settings = get_settings()
    if not settings.llm_configured():
        return _execute_response(
            status="error",
            error="the agent is not configured: no LLM API key is set",
            response=None,
            steps=[],
        )

    try:
        from app.runner import run_agent

        result = await run_agent(
            payload.prompt,
            student_id=payload.student_id,
            conversation_id=payload.conversation_id,
            started_at=started_at,
        )
    except Exception as error:  # noqa: BLE001 -- see module docstring
        logger.exception("execute_failed")
        return _execute_response(
            status="error", error=f"{type(error).__name__}: {error}", response=None, steps=[]
        )

    return _execute_response(
        status="ok" if result.ok else "error",
        error=result.error,
        response=result.answer,
        steps=result.steps,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="UniPilot Agent",
        description="A grounded academic-advising agent.",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        """Render a malformed request in the spec's shape, not FastAPI's.

        FastAPI answers a validation failure with `{"detail": [...]}` and a 422.
        On `/api/execute` that breaks the guarantee that every response carries
        the same four fields, so a client parsing our contract would hit an
        unexpected shape at exactly the moment it most needs a readable error.
        """
        if request.url.path != "/api/execute":
            return JSONResponse(status_code=422, content={"detail": error.errors()})
        reasons = "; ".join(
            f"{'.'.join(str(part) for part in item.get('loc', ()) if part != 'body') or 'body'}: "
            f"{item.get('msg', 'invalid')}"
            for item in error.errors()
        )
        return JSONResponse(
            status_code=422,
            content=_execute_response(
                status="error",
                error=f"invalid request -- {reasons or 'malformed body'}",
                response=None,
                steps=[],
            ),
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        settings = get_settings()
        return {
            "status": "ok",
            "llm": settings.llm_configured(),
            "chat_provider": settings.chat_provider(),
            "chat_model": settings.llm_chat_model,
            # The one check that catches a submission still pointed at the
            # development provider -- a mistake with no other symptom.
            "submission_ready": settings.submission_ready(),
            "supabase": settings.supabase_configured(),
            "pinecone": settings.pinecone_configured(),
        }

    # The GUI, served by the app itself so `/` works where the app runs.
    #
    # In production Vercel serves `public/` from its CDN before a request ever
    # reaches this function, so the spec's "GUI on your root url" was satisfied
    # there and nowhere else: run the app locally and `/` was a 404, while every
    # `/api/*` route answered. The spec asks for BOTH environments to work, and
    # a GUI that only exists once deployed cannot be exercised before deploying.
    #
    # Mounted LAST so it can never shadow an API route, and only when the
    # directory is actually present -- a missing GUI is a build problem, and
    # failing to start over it would take the four endpoints down with it.
    if _PUBLIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=_PUBLIC_DIR, html=True), name="gui")
    else:  # pragma: no cover -- only in a build that dropped the asset
        logger.warning("no GUI at %s; '/' will 404 in this environment", _PUBLIC_DIR)

    return app


app = create_app()

__all__ = ["ExecuteRequest", "app", "create_app"]
