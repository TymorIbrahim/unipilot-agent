"""The response-shape contract the course spec fixes for us.

These are the tests worth having before the reasoning core lands: the spec grades
the SHAPE of what we return, and a shape regression is silent -- the endpoint
still answers 200, just with fields a grader's parser does not find. Pinning the
exact top-level key sets here means any future edit that drops or renames one
fails loudly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.tracing.modules import MODULE_NAMES

EXECUTE_KEYS = {"status", "error", "response", "steps"}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def stub_agent(monkeypatch: pytest.MonkeyPatch):
    """Replace the reasoning core for the SHAPE tests.

    These tests exist to pin the response contract, and the contract is
    `main.py`'s job -- it holds whether the four fields survive every path. Left
    unstubbed they run the real agent, which means a live Postgres, real model
    calls billed against a $13 budget, and ~10s per test for an assertion about
    dictionary keys. Worse, they would pass just as green while the agent was
    broken, because an error response has the same four fields.

    The end-to-end behaviour is checked by the `live`-marked test at the bottom
    of this file, where paying for it is a deliberate act.
    """
    import app.runner

    async def _fake(prompt: str, *, student_id: str | None = None,
                    conversation_id: str | None = None, started_at: float | None = None):
        _fake.seen = {"prompt": prompt, "student_id": student_id,
                      "conversation_id": conversation_id, "started_at": started_at}
        return app.runner.AgentResult(
            ok=True, answer="stubbed", error=None, steps=[]
        )

    _fake.seen = {}
    monkeypatch.setattr(app.runner, "run_agent", _fake)
    return _fake


def test_team_info_has_the_required_fields(client: TestClient) -> None:
    body = client.get("/api/team_info").json()

    assert set(body) == {"group_batch_order_number", "team_name", "students"}
    assert body["group_batch_order_number"] == "3_3"
    assert body["team_name"] == "UniPilot"
    assert all({"name", "email"} == set(student) for student in body["students"])


def test_team_info_lists_every_member_with_a_real_address(client: TestClient) -> None:
    """Placeholders here ship straight into the graded identity endpoint."""
    students = client.get("/api/team_info").json()["students"]

    from app.team import STUDENTS

    # Compared against the roster SOURCE rather than a hand-maintained count.
    # The literal `2` here had to be edited to add a third member, which makes
    # the test a step in a checklist instead of a check -- and a checklist step
    # is exactly what gets skipped.
    assert len(students) == len(STUDENTS)
    assert students == [dict(student) for student in STUDENTS]
    assert all("@" in student["email"] and "TODO" not in student["email"] for student in students)
    assert all(student["name"] and "TODO" not in student["name"] for student in students)
    assert len({student["email"] for student in students}) == len(students)


def test_agent_info_has_the_required_fields(client: TestClient) -> None:
    body = client.get("/api/agent_info").json()

    assert {"description", "purpose", "prompt_template", "prompt_examples"} <= set(body)
    assert "template" in body["prompt_template"]
    assert isinstance(body["prompt_examples"], list)


def test_agent_info_examples_carry_the_spec_shape(client: TestClient) -> None:
    """Each example must be self-contained evidence: the prompt, what the agent
    actually said, and every model call it made on the way.

    Checked through the ENDPOINT rather than against the file, because the file
    being present is not the same as it being served -- and an empty set here is
    a silent hole in a graded response.
    """
    examples = client.get("/api/agent_info").json()["prompt_examples"]

    assert examples, "no recorded examples are being served; run scripts/record_examples.py"
    for example in examples:
        assert {"prompt", "full_response", "steps"} <= set(example)
        assert example["prompt"] and example["full_response"]
        assert example["steps"], "an example with no steps documents nothing"
        for step in example["steps"]:
            assert set(step) == {"module", "prompt", "response"}
            assert set(step["prompt"]) == {"System_prompt", "User_prompt"}
            assert step["module"] in MODULE_NAMES


def test_agent_info_module_names_match_the_shared_registry(client: TestClient) -> None:
    """The spec requires one vocabulary across diagram, steps and descriptions."""
    body = client.get("/api/agent_info").json()

    assert {module["module"] for module in body["modules"]} == MODULE_NAMES


def test_execute_returns_exactly_the_four_required_fields(client: TestClient, stub_agent) -> None:
    body = client.post("/api/execute", json={"prompt": "how many credits remain?"}).json()

    assert set(body) == EXECUTE_KEYS
    assert body["status"] in {"ok", "error"}
    assert isinstance(body["steps"], list)


def test_execute_keeps_the_four_fields_when_the_request_is_invalid(client: TestClient) -> None:
    """A malformed body must not fall through to FastAPI's own error shape."""
    response = client.post("/api/execute", json={})

    assert response.status_code == 422
    assert set(response.json()) == EXECUTE_KEYS
    assert response.json()["status"] == "error"
    assert response.json()["error"]


def test_execute_accepts_a_bare_prompt_without_a_student(client: TestClient, stub_agent) -> None:
    """The spec's input contract is `{prompt}` alone; it must remain sufficient."""
    response = client.post("/api/execute", json={"prompt": "what is my GPA?"})

    assert response.status_code == 200
    assert set(response.json()) == EXECUTE_KEYS


def test_execute_threads_a_conversation_id_to_the_core(client: TestClient, stub_agent) -> None:
    """The conversation store was built, wired into the reasoning core, and then
    unreachable: this endpoint never passed an id, so every request started with
    no history and a follow-up like "yes, continue" had nothing to continue."""
    response = client.post(
        "/api/execute", json={"prompt": "yes, continue", "conversation_id": "thread-1"}
    )

    assert response.status_code == 200
    assert stub_agent.seen["conversation_id"] == "thread-1"


def test_execute_without_a_conversation_id_is_still_one_shot(
    client: TestClient, stub_agent
) -> None:
    """The spec's `{prompt}` alone must keep working exactly as before."""
    client.post("/api/execute", json={"prompt": "what is my GPA?"})

    assert stub_agent.seen["conversation_id"] is None


def test_execute_reports_a_failed_run_in_the_same_four_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure path is graded too, and it is the one that rots unnoticed.

    An exception escaping the core must still come back as the four fields with
    `status: error` -- not as FastAPI's own 500 shape, which a grader's parser
    would not recognise as a response at all.
    """
    import app.runner

    async def _explode(prompt: str, **_: object):
        raise RuntimeError("core exploded")

    monkeypatch.setattr(app.runner, "run_agent", _explode)
    body = client.post("/api/execute", json={"prompt": "anything"}).json()

    assert set(body) == EXECUTE_KEYS
    assert body["status"] == "error"
    assert "core exploded" in body["error"]
    assert body["steps"] == []


def test_health_reports_configuration_readiness(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert {
        "status",
        "llm",
        "chat_provider",
        "chat_model",
        "submission_ready",
        "supabase",
        "pinecone",
    } == set(body)


@pytest.mark.parametrize(
    ("base_url", "expected", "ready"),
    [
        ("https://api.llmod.ai/v1", "llmod", True),
        ("https://api.openai.com/v1", "api.openai.com", False),
    ],
)
def test_submission_ready_only_when_chat_runs_on_llmod(
    base_url: str, expected: str, ready: bool
) -> None:
    """Guards the one misconfiguration that produces no visible symptom."""
    settings = Settings(llm_api_key="test-key", llm_base_url=base_url)

    assert settings.chat_provider() == expected
    assert settings.submission_ready() is ready


@pytest.mark.live
def test_execute_answers_a_real_question_end_to_end(client: TestClient) -> None:
    """The whole agent, for real: Postgres, the model, the loop, the trace.

    Deselected by default -- it costs model calls. It is the only test that can
    catch the failure the stubbed shape tests structurally cannot: an agent that
    returns the four required fields perfectly while answering nothing.

    Asserts the SPEC's step schema rather than the answer's content. What the
    agent says is a judgement about a student's record and will change as the
    data does; that every model call is recorded, in order, under a module name
    the rest of the submission also uses, is a contract that must not.
    """
    body = client.post(
        "/api/execute",
        json={"prompt": "How many credits have I completed so far?"},
    ).json()

    assert set(body) == EXECUTE_KEYS
    assert body["status"] == "ok", body["error"]
    assert body["response"]
    assert body["steps"], "a real run must record at least one model call"

    for step in body["steps"]:
        assert set(step) == {"module", "prompt", "response"}
        assert set(step["prompt"]) == {"System_prompt", "User_prompt"}
        assert step["module"] in MODULE_NAMES, (
            f"step module {step['module']!r} is not in the shared registry, so the "
            "diagram and the trace would disagree"
        )


def test_credentials_never_appear_in_a_settings_repr() -> None:
    """A pydantic model prints its fields, so a failing assertion, a logged
    settings object, or a Vercel stack trace would otherwise carry live keys."""
    rendered = repr(Settings(llm_api_key="sk-secret", pinecone_api_key="pc-secret"))

    assert "sk-secret" not in rendered
    assert "pc-secret" not in rendered


def test_execute_hands_the_core_the_windows_true_start(client, stub_agent) -> None:
    """The budget must bound the REQUEST, not just the reasoning.

    On a cold start the process spends ~13s importing a 45MB bundle and a
    4,895-chunk corpus before the handler runs -- measured at 18.2s cold against
    5.0s warm for the same two-step question. The loop used to start its clock
    when the loop started, so it planned against 240s while the caller was
    already 13 seconds into a 60s ceiling. A live run logged
    `outcome=answered elapsed=48.2s` and the caller still received nothing:
    13 + 48 crossed the cap while the response was being written, and the whole
    run was discarded at the last moment.
    """
    response = client.post("/api/execute", json={"prompt": "hi"})
    assert response.status_code == 200
    assert stub_agent.seen["started_at"] is not None, (
        "the route must tell the core when the caller's window opened"
    )


def test_the_first_request_is_charged_for_the_import(client, stub_agent) -> None:
    """The cold-start cost belongs to whoever pays it.

    The first request in a process is still inside the import; later ones are
    not, because for them the import happened in someone else's window.
    """
    import app.main

    app.main._SERVED_A_REQUEST = False
    client.post("/api/execute", json={"prompt": "first"})
    first = stub_agent.seen["started_at"]
    client.post("/api/execute", json={"prompt": "second"})
    second = stub_agent.seen["started_at"]

    assert first == app.main._IMPORTED_AT, "a cold request's window opens at import"
    assert second > first, "a warm request's window opens when it arrives"


def test_a_stale_import_does_not_spend_the_budget_before_the_run(client, stub_agent) -> None:
    """"Imported" and "first request arrives" are not the same moment.

    The platform can initialise a function and route to it later. Charging the
    whole gap spent the budget before the run began: a live request came back
    "I ran out of time before I could finish that" after 9.3 seconds of a
    45-second budget, having made one tool call.

    So the cold-start charge is bounded. Measured cold start is ~13s; the bound
    is 20.
    """
    import time

    import app.main

    app.main._SERVED_A_REQUEST = False
    app.main._IMPORTED_AT = time.monotonic() - 600.0  # a process idle ten minutes

    client.post("/api/execute", json={"prompt": "hi"})
    charged = time.monotonic() - stub_agent.seen["started_at"]

    assert charged <= app.main._MAX_COLD_START_CHARGE_S + 1.0, (
        "an idle process charged the whole gap to the request's budget"
    )


def test_a_genuine_cold_start_is_still_charged(client, stub_agent) -> None:
    """The bound must not switch the accounting off -- the ~13s import is real
    and is what made a run that answered in 48.2s deliver nothing."""
    import time

    import app.main

    app.main._SERVED_A_REQUEST = False
    app.main._IMPORTED_AT = time.monotonic() - 12.0  # a normal cold start

    client.post("/api/execute", json={"prompt": "hi"})
    charged = time.monotonic() - stub_agent.seen["started_at"]

    assert charged >= 11.0, "a real cold start must still come out of the budget"


class TestTheGuiIsServedByTheAppItself:
    """The spec asks for a GUI at the root URL, in BOTH environments.

    Vercel serves `public/` from its CDN before a request reaches the function,
    so the requirement was met in production and nowhere else: running the app
    locally, every `/api/*` route answered and `/` was a 404. A GUI that only
    exists once deployed cannot be exercised before deploying, which is exactly
    what "make sure it works on your development environment" is asking for.
    """

    def test_the_root_url_serves_the_gui(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        response = TestClient(app).get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_the_gui_has_what_the_spec_requires(self) -> None:
        from fastapi.testclient import TestClient

        from app.main import app

        page = TestClient(app).get("/").text
        assert "<textarea" in page.lower()
        assert "Run Agent" in page
        assert "/api/execute" in page

    def test_the_mount_does_not_shadow_the_api(self) -> None:
        """Mounted last for this reason -- a catch-all at `/` must not win."""
        from fastapi.testclient import TestClient

        client = TestClient(__import__("app.main", fromlist=["app"]).app)
        for path in ("/api/team_info", "/api/agent_info", "/api/health"):
            assert client.get(path).status_code == 200, path
        assert client.get("/api/model_architecture").headers["content-type"] == "image/png"
