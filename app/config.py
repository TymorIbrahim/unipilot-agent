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

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The spec's hard ceiling: Vercel kills any function call at 300s. We finish
# well before it and ship whatever the agent has grounded so far, because a
# partial honest answer scores and a timeout does not.
VERCEL_HARD_LIMIT_S = 300.0
DEFAULT_TIME_BUDGET_S = 240.0


LLMOD_HOST = "api.llmod.ai"


class Settings(BaseSettings):
    # --- Chat model ---------------------------------------------------------
    # Defaults are the SUBMISSION values. Development overrides them to OpenAI
    # directly, on the same model, so iteration does not consume the $13 course
    # budget -- see `.env`. `chat_provider` reports which is live.
    llm_api_key: str = Field(default="", repr=False)
    llm_base_url: str = "https://api.llmod.ai/v1"
    llm_chat_model: str = "MB5R2CF-azure/gpt-5.4-mini"

    # --- Embedding model ----------------------------------------------------
    # Configured SEPARATELY from chat and pinned to LLMod on purpose: the
    # Pinecone index was built with this exact model, so pointing embeddings at
    # a different one would silently mismatch every stored vector -- queries
    # would still return results, just meaningless ones. This must not follow
    # the chat provider when chat is swapped for development.
    llm_embedding_api_key: str = Field(default="", repr=False)
    llm_embedding_base_url: str = "https://api.llmod.ai/v1"
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
    supabase_key: str = Field(default="", repr=False)

    # --- Pinecone (vector database) ----------------------------------------
    pinecone_api_key: str = Field(default="", repr=False)
    pinecone_index_name: str = "unipilot-wiki"

    # --- Budgets ------------------------------------------------------------
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    max_turns: int = 8

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    def chat_provider(self) -> str:
        """Which provider the CHAT model is actually pointed at right now.

        Surfaced through `/api/health` because the dev/submission swap is a
        credential change with no visible symptom: the agent answers just as
        well on either provider, so a run against the wrong one looks entirely
        healthy and would only be caught by whoever reads the billing. Naming
        the live provider makes the mistake visible before submission instead
        of after grading.
        """
        host = self.llm_base_url.split("//")[-1].split("/")[0].lower()
        if host == LLMOD_HOST:
            return "llmod"
        return host or "unknown"

    def submission_ready(self) -> bool:
        """True only when the graded configuration is the live one."""
        return self.chat_provider() == "llmod" and self.llm_configured()

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
