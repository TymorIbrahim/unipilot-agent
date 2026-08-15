"""A fresh connection pool per test, for the same reason motor needed one.

`get_pool` memoises the pool process-wide, which is right in production -- a
Vercel invocation should not rebuild it -- and wrong under pytest-asyncio, which
gives every test its own event loop. The second test to ask for the database
gets a pool bound to the first test's closed loop and fails with `RuntimeError`
or `InterfaceError`, and the `database` fixture reads that as "no database" and
SKIPS.

That is the failure mode this suite exists to prevent, wearing a different hat:
eleven reachability tests reporting "NOT VERIFIED" while Supabase sat there
answering queries. A skip is not a neutral outcome -- it is a claim that
something could not be checked, and a wrong one is indistinguishable from a pass.
"""

from __future__ import annotations

import pytest

from app.db import postgres


@pytest.fixture(autouse=True)
async def _fresh_pool_per_test():
    """Drop any pool from an earlier test's loop, and clean up after this one."""
    await postgres.close_pool()
    yield
    await postgres.close_pool()
