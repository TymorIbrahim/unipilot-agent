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

    # The Postgres DSN, which is a SEPARATE credential from the anon key above:
    # the agent reads through the Postgres wire protocol, not PostgREST, because
    # the predicate grammar includes field-to-field comparison and PostgREST
    # cannot express it. Its HOST is rewritten to the pooler at connect time --
    # `db.<ref>.supabase.co` is IPv6-only and unroutable from Vercel. See
    # `app/db/postgres.py`.
    supabase_db_url: str = Field(default="", repr=False)
    supabase_pooler_host: str = "aws-0-us-east-1.pooler.supabase.com"
    supabase_pooler_port: int = 6543

    # --- Pinecone (vector database) ----------------------------------------
    # The semantic half of retrieval. It carries 60% of the ranking weight
    # (`hybridVectorWeight` in the tuned profile), so a missing key is not a
    # small degradation -- it drops the majority of the signal and leaves
    # keyword-only ranking, which demonstrably surfaces the wrong page for
    # verbose natural-language questions.
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
        """True when the agent can actually READ. The anon key is not enough --
        it reaches PostgREST, which the data layer does not use."""
        return bool(self.supabase_db_url.strip())

    def pinecone_configured(self) -> bool:
        return bool(self.pinecone_api_key.strip())

    def embeddings_available(self) -> bool:
        """Whether a query can be embedded at all.

        Separate from `pinecone_configured`: the index and the embedder are two
        different credentials, and having one without the other still means no
        semantic search.
        """
        return bool(self.llm_embedding_api_key.strip() and self.llm_embedding_model.strip())

    def vector_index_enabled(self) -> bool:
        """The semantic half is only live when BOTH halves are.

        Checked in one place because the failure is silent either way: a query
        that cannot be embedded, or an index that cannot be reached, both return
        zero candidates and leave a keyword-only ranking that still looks like a
        working hybrid search.
        """
        return self.pinecone_configured() and self.embeddings_available()

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
