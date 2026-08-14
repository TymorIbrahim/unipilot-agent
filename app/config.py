"""Runtime configuration.

Every external dependency the agent has is named here and nowhere else, so the
deployment story is readable in one file: which model provider, which database,
which vector index, and how long a request may run.

Defaults are the SPEC's values, not our development conveniences. A missing env
var should leave the agent pointing at the models and services the course
requires, so a fresh deploy is correct before anyone tunes it.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# The spec's hard ceiling: Vercel kills any function call at 300s. We finish
# well before it and ship whatever the agent has grounded so far, because a
# partial honest answer scores and a timeout does not.
VERCEL_HARD_LIMIT_S = 300.0
DEFAULT_TIME_BUDGET_S = 240.0


class Settings(BaseSettings):
    # --- LLM provider (LLMod.ai) -------------------------------------------
    llm_api_key: str = ""
    llm_base_url: str = "https://api.llmod.ai/v1"
    llm_chat_model: str = "MB5R2CF-azure/gpt-5.4-mini"
    llm_embedding_model: str = "MB5R2CF-azure/text-embedding-3-small"

    # Reasoning is the point of this agent, so thinking is on by default; the
    # effort level stays tunable because it trades directly against the budget.
    llm_thinking_enabled: bool = True
    llm_reasoning_effort: str | None = "medium"
    llm_timeout_s: float = 120.0

    # --- Supabase (primary database) ---------------------------------------
    # Use the POOLED connection (Supavisor). A serverless function opens a
    # connection per invocation; direct connections exhaust Postgres' limit
    # under even light concurrency.
    supabase_url: str = ""
    supabase_key: str = ""

    # --- Pinecone (vector database) ----------------------------------------
    pinecone_api_key: str = ""
    pinecone_index_name: str = "unipilot-wiki"

    # --- Budgets ------------------------------------------------------------
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    max_turns: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    def supabase_configured(self) -> bool:
        return bool(self.supabase_url.strip() and self.supabase_key.strip())

    def pinecone_configured(self) -> bool:
        return bool(self.pinecone_api_key.strip())

    def effective_time_budget_s(self) -> float:
        """Never allow a configured budget to exceed the platform's own limit.

        A budget above 300s cannot be honoured -- Vercel terminates the call
        first -- so a misconfiguration here would silently reintroduce exactly
        the timeout this budget exists to prevent.
        """
        return min(self.time_budget_s, VERCEL_HARD_LIMIT_S - 30.0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


__all__ = ["DEFAULT_TIME_BUDGET_S", "VERCEL_HARD_LIMIT_S", "Settings", "get_settings"]
