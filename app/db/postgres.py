"""The Postgres connection -- and the one place the pooler decision is written down.

Supabase publishes two ways in, and only one of them works here:

  db.<ref>.supabase.co:5432          the DIRECT host. Publishes an AAAA record
                                     and nothing else, so it is unroutable from
                                     any IPv4-only network -- Vercel's runtime
                                     included. Its DSN is still what the Supabase
                                     dashboard hands you, which is why this
                                     module rewrites it rather than using it.

  aws-0-<region>.pooler.supabase.com the SUPAVISOR POOLER. IPv4, and a pooler --
                                     which matters twice over, because a
                                     serverless function opens a connection per
                                     invocation and direct connections exhaust
                                     Postgres' limit under even light load.

Transaction mode (6543) is the serverless default: connections are handed back
between statements rather than held for a whole session. It does not support
prepared statements, hence `statement_cache_size=0` -- without it asyncpg
prepares by name and the pooler hands the next statement to a different backend
that has never heard of it.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_POOLER_HOST = "aws-0-us-east-1.pooler.supabase.com"
DEFAULT_POOLER_PORT = 6543

_pool: Any = None


class ConnectionError_(RuntimeError):
    """The database could not be reached. Raised rather than swallowed: an
    unreachable database that returns no rows is indistinguishable from a
    student with no records, and this layer exists to never make that mistake."""


def pooler_connect_kwargs(settings: Any) -> dict[str, Any]:
    """asyncpg connect arguments, derived from `SUPABASE_DB_URL`.

    The DSN carries the password and the project ref; the HOST is replaced,
    because the one it names cannot be reached. Rewriting rather than asking for
    a second DSN keeps a single credential in `.env` and makes it impossible to
    have the two drift apart.
    """
    raw = getattr(settings, "supabase_db_url", "") or ""
    if not raw:
        raise ConnectionError_(
            "SUPABASE_DB_URL is not set. It carries both the database password and the "
            "project ref, and there is no other route to Postgres."
        )
    parsed = urllib.parse.urlparse(raw)
    project_ref = (parsed.hostname or "").removeprefix("db.").split(".")[0]
    if not project_ref:
        raise ConnectionError_(f"no Supabase project ref in host {parsed.hostname!r}")

    return {
        "host": getattr(settings, "supabase_pooler_host", DEFAULT_POOLER_HOST),
        "port": int(getattr(settings, "supabase_pooler_port", DEFAULT_POOLER_PORT)),
        "user": f"postgres.{project_ref}",
        "password": urllib.parse.unquote(parsed.password or ""),
        "database": parsed.path.lstrip("/") or "postgres",
        "ssl": "require",
        # See the module docstring: mandatory on the transaction pooler.
        "statement_cache_size": 0,
    }


async def _configure(connection: Any) -> None:
    """Decode jsonb into Python objects on the way out.

    asyncpg hands back jsonb as a STRING by default. `find._convert` expects a
    list of mappings for a declared `ArrayOf(Sub(...))`, and a string is not one
    -- it would fail the isinstance check and the field would come back ABSENT,
    which is the silent-omission failure rather than an error.
    """
    await connection.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )
    await connection.set_type_codec(
        "json", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
    )


class Database:
    """What `find` talks to. One method, because one is all it needs.

    Deliberately not a general query interface: everything that reaches Postgres
    is built by `compile_to_sql` from a validated predicate, and a broader
    surface here would be somewhere else for a hand-written string to creep in.
    """

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def fetch(self, sql: str, *parameters: Any) -> list[Mapping[str, Any]]:
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(sql, *parameters)
        return [dict(row) for row in rows]

    async def execute(self, sql: str, *parameters: Any) -> str:
        async with self._pool.acquire() as connection:
            return await connection.execute(sql, *parameters)

    async def fetchval(self, sql: str, *parameters: Any) -> Any:
        async with self._pool.acquire() as connection:
            return await connection.fetchval(sql, *parameters)


CONNECT_ATTEMPTS = 3
CONNECT_TIMEOUT_S = 10.0
_RETRY_BACKOFF_S = (0.5, 1.5)

_pool_lock: Any = None


async def get_pool(settings: Any | None = None) -> Any:
    """The process-wide connection pool, created once, with a bounded retry.

    Small on purpose. A Vercel invocation serves one request, so a large pool
    buys nothing and costs connections that the pooler shares across every
    concurrently-warm instance.

    **Retried, because the first connection is the fragile one.** Opening a
    Supavisor connection means TCP, TLS and a SCRAM handshake against a shared
    pooler, and a cold instance occasionally times out doing it -- observed in
    production as `TimeoutError` with an empty message, which failed the entire
    request. A request that dies because one handshake was slow is the wrong
    trade when the budget is 240s and a second attempt costs under a second.

    Three attempts with an explicit 10s ceiling each, so the worst case is ~32s
    rather than asyncpg's much longer default: a genuinely unreachable database
    should be reported while there is still time to say so, not discovered when
    the platform kills the call.
    """
    global _pool, _pool_lock
    if _pool is not None:
        return _pool

    import asyncio

    import asyncpg

    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    if _pool_lock is None:
        _pool_lock = asyncio.Lock()

    # The loop issues tool calls concurrently, so two of them can race here on a
    # cold instance. Without the lock both build a pool and one is silently
    # orphaned, holding pooler connections nothing will ever close.
    async with _pool_lock:
        if _pool is not None:
            return _pool

        last_error: Exception | None = None
        for attempt in range(CONNECT_ATTEMPTS):
            try:
                _pool = await asyncpg.create_pool(
                    **pooler_connect_kwargs(settings),
                    min_size=1,
                    max_size=4,
                    init=_configure,
                    command_timeout=30,
                    timeout=CONNECT_TIMEOUT_S,
                )
                if attempt:
                    logger.info("postgres pool opened on attempt %d", attempt + 1)
                return _pool
            except ConnectionError_:
                # A missing DSN is a configuration fault. Retrying it just
                # spends the budget arriving at the same answer.
                raise
            except Exception as error:  # noqa: BLE001 -- retried, then surfaced
                last_error = error
                if attempt < CONNECT_ATTEMPTS - 1:
                    logger.warning(
                        "postgres pool attempt %d/%d failed (%s); retrying",
                        attempt + 1,
                        CONNECT_ATTEMPTS,
                        type(error).__name__,
                    )
                    await asyncio.sleep(_RETRY_BACKOFF_S[attempt])

    raise ConnectionError_(
        f"could not reach Postgres after {CONNECT_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


async def get_database(settings: Any | None = None) -> Database:
    return Database(await get_pool(settings))


async def close_pool() -> None:
    """Release the pool. Used by tests; a serverless instance simply dies."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


__all__ = [
    "DEFAULT_POOLER_HOST",
    "DEFAULT_POOLER_PORT",
    "ConnectionError_",
    "Database",
    "close_pool",
    "get_database",
    "get_pool",
    "pooler_connect_kwargs",
]
