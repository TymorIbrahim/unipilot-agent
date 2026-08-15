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

    async def _fake(prompt: str, *, student_id: str | None = None):
        return app.runner.AgentResult(
            ok=True, answer="stubbed", error=None, steps=[]
        )

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

    assert len(students) == 2
    assert all("@" in student["email"] and "TODO" not in student["email"] for student in students)
    assert all(student["name"] and "TODO" not in student["name"] for student in students)


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


def test_execute_reports_a_failed_run_in_the_same_four_fields(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure path is graded too, and it is the one that rots unnoticed.

    An exception escaping the core must still come back as the four fields with
    `status: error` -- not as FastAPI's own 500 shape, which a grader's parser
    would not recognise as a response at all.
    """
    import app.runner

    async def _explode(prompt: str, *, student_id: str | None = None):
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
