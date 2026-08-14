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

from app.main import app
from app.tracing.modules import MODULE_NAMES

EXECUTE_KEYS = {"status", "error", "response", "steps"}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


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


def test_agent_info_module_names_match_the_shared_registry(client: TestClient) -> None:
    """The spec requires one vocabulary across diagram, steps and descriptions."""
    body = client.get("/api/agent_info").json()

    assert {module["module"] for module in body["modules"]} == MODULE_NAMES


def test_execute_returns_exactly_the_four_required_fields(client: TestClient) -> None:
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


def test_execute_accepts_a_bare_prompt_without_a_student(client: TestClient) -> None:
    """The spec's input contract is `{prompt}` alone; it must remain sufficient."""
    response = client.post("/api/execute", json={"prompt": "what is my GPA?"})

    assert response.status_code == 200
    assert set(response.json()) == EXECUTE_KEYS


def test_health_reports_configuration_readiness(client: TestClient) -> None:
    body = client.get("/api/health").json()

    assert {"status", "llm", "supabase", "pinecone"} == set(body)
