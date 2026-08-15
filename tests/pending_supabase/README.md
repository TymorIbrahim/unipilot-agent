# Held back until the Supabase port

These suites exercise the agent's DATA layer, which still speaks MongoDB
(`motor`, `compile_to_mongo`, collection-shaped source schemas). They are kept
verbatim rather than deleted because they encode the behaviour the Supabase
backend has to reproduce -- notably `test_predicate`'s operator semantics.

Port them alongside `compile_to_sql`; a Supabase backend that cannot satisfy
these is not finished.

## Ported

`test_sources.py` is split in two. Its claims about REAL DATA are
`tests/reachability/test_sources_against_data.py` (marked `supabase`), and its
pure checks -- the semi-join and the nested declarations, neither of which had
any coverage in the active suite -- are `tests/agent_core/facts/test_semi_join.py`.
Its `TestObjectIdFilters` was deliberately NOT ported: it exercised the pass that
turned a model's string filter into a BSON ObjectId, and Postgres stores those
ids as text, so the pass was deleted rather than translated. One check is new
and could not exist against Mongo -- every declared field must be a real COLUMN,
which is knowable up front in a database that has a schema.

`test_predicate.py` is retired, split in two. Its cross-engine MATRIX -- the
part this README called the reason to keep the file -- is
`tests/reachability/test_predicate_engines.py`, run through `matches` and
`compile_to_sql` against a real database. Its engine-free rules are
`tests/agent_core/facts/test_predicate_semantics.py` in the fast suite: path
resolution, the type rules, and the constant-true predicate. Nothing else
asserted that a course code cannot be ORDERED, which a SQL engine will happily
do byte-wise rather than refuse.

`TestMongoCompilation` was deliberately NOT ported: it pinned `$nor` against
field-level `$not` on a compiler that has been deleted.

`test_find.py` is `tests/reachability/test_find_against_data.py`, marked
`supabase` and green -- the Phase 5 gate (a truncated fetch reports
`complete=false`, and an `aggregate` over it fails closed) now holds against
Postgres.

The port needed a real table. There is no throwaway database, and the dirty
fixture data is the whole point: every column is TEXT, because "3.5" and
"00940224" are the same shape of string and only the DECLARED schema separates a
quantity from an identifier. Letting Postgres type `credits` as double precision
would have tested nothing. So it creates `pytest_find_courses`, fills it with the
same four rows, and drops it -- never `courses`, which is 2,613 rows the whole
agent reads. One difference stands: Mongo let a document omit a field, a table
cannot, so absence is NULL here and the distinction that matters (absent is not
zero) is asserted the same way.

`test_ledger.py` is `tests/reachability/test_ledger_against_data.py`, marked
`supabase` and green. `MongoLedger` -> `SupabaseLedger` and `MongoConversations`
-> `SupabaseConversations`; the guarantee is unchanged because it was never
Mongo's -- the token is the PRIMARY KEY, so the database arbitrates a
double-submit and `insert ... on conflict do nothing` reports who won. Before
this ran, `spent_confirmations` held 0 rows: the one-write guarantee behind
`propose`, the only tool that can change anything, had never been exercised
against Postgres at all.

What the port needed beyond the mechanical rename: there is no throwaway
database. `spent_confirmations` and `agent_conversations` are LIVE runtime state
that `scripts/seed.py` deliberately never touches, and the latter holds real
exchanges from the deployed GUI, so `delete_many({})` has no equivalent here.
Rows are namespaced and diffed out instead -- and a prefix alone was not enough:
the end-to-end `propose` tests spend tokens minted by `confirm()`, which knows
nothing about the fixture, and the first run left two of them behind in a live
table.

`test_reachability.py` now lives at `tests/reachability/`, marked `supabase`
and running green against the real database. **Run it before submitting**
(`pytest -m supabase`): it is the only check that every advertised tool can
actually be fed by a route the model can walk, which is the failure that hid
`plan_term` and three other tools.

What the port needed, if the rest follow:

- `app.db.mongo.get_database` -> `app.db.postgres.get_database`, and
  `db.command("ping")` -> `db.fetchval("select 1")`.
- A `conftest.py` resetting the connection pool per test. `get_pool` memoises
  process-wide and pytest-asyncio gives each test its own event loop, so the
  second test to touch the database got a pool bound to a closed loop, raised,
  and was reported as "no database" -- eleven reachability tests SKIPPED while
  Supabase sat there answering queries.
- Mongo probes rewritten as SQL: `find_one({"semesters.0.plannedCourses.0":
  {"$exists": True}})` becomes
  `jsonb_path_exists("semesters", '$[*].plannedCourses[*]')`.
- `prerequisite_edges_source(...)` dropped -- edges are an ordinary REGISTRY
  table now, not a graph-derived schema.
