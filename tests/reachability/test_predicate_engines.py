"""One grammar, two engines -- and they must not drift.

`select` evaluates a predicate in memory via `matches`; `find` pushes the same
predicate to Postgres via `compile_to_sql`. When the two disagree, the agent
sees different rows depending on which path a question happened to take, and
nothing says so: both return a plausible collection.

`compile_to_sql` had NO tests. It is the boundary that decides which rows the
agent can see at all, and the Mongo suite that used to pin these semantics
(`tests/pending_supabase/test_predicate.py`) tests an engine this deployment no
longer has. This is its design ported: the same matrix, run through both
engines, compared row for row against a real database.

The NULL handling is the part worth proving. SQL comparisons against NULL are
NULL rather than false, so without `coalesce(..., false)` at every leaf,
`not ("grade" > 80)` DROPS a record that has no grade while `matches` KEEPS it.
A sparse record is the common case here -- 28% of transcript rows reference a
catalog course that no longer exists -- so the two engines would disagree on
real data, not on a contrived edge.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.predicate import (
    Always,
    And,
    Comparison,
    Not,
    Op,
    Or,
    Path,
    compile_to_sql,
    matches,
)
from app.agent_core.facts.types import Basis, Record, Scalar, ScalarKind
from app.db.postgres import get_database

pytestmark = pytest.mark.supabase

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER

TABLE = "predicate_engine_matrix"


def _record(**fields: object) -> Record:
    typed: dict = {}
    for name, value in fields.items():
        if isinstance(value, bool):
            typed[name] = Scalar(ScalarKind.BOOL, value)
        elif isinstance(value, (int, float)):
            typed[name] = Scalar(Q, value)
        else:
            typed[name] = Scalar(I, value)
    return Record(fields=typed, basis=Basis.OFFICIAL_RECORD)


# The shared matrix. Both engines see exactly this. The fourth row has NO grade,
# which is the row every negation disagrees about when NULL is mishandled.
MATRIX_RECORDS = (
    _record(id="00940224", grade=95, credits=3.5, passing=60),
    _record(id="00960211", grade=60, credits=3.0, passing=60),
    _record(id="00970800", grade=88, credits=4.0, passing=70),
    _record(id="00940594", credits=2.5, passing=60),
)

MATRIX_PREDICATES = (
    Always(),
    Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)),
    Comparison(Path.parse("grade"), Op.GE, Scalar(Q, 88)),
    Comparison(Path.parse("grade"), Op.LT, Scalar(Q, 88)),
    Comparison(Path.parse("id"), Op.EQ, Scalar(I, "00940224")),
    Comparison(Path.parse("id"), Op.NE, Scalar(I, "00940224")),
    Comparison(Path.parse("id"), Op.IN, (Scalar(I, "00940224"), Scalar(I, "00970800"))),
    Comparison(Path.parse("grade"), Op.GT, Path.parse("passing")),
    And((
        Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80)),
        Comparison(Path.parse("credits"), Op.GE, Scalar(Q, 3.5)),
    )),
    Or((
        Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)),
        Comparison(Path.parse("credits"), Op.LT, Scalar(Q, 3.0)),
    )),
    # The negations. Each of these keeps the grade-less row under `matches`, and
    # each would drop it under naive SQL.
    Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80))),
    Not(And((
        Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80)),
        Comparison(Path.parse("credits"), Op.GE, Scalar(Q, 3.5)),
    ))),
    Not(Or((
        Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)),
        Comparison(Path.parse("credits"), Op.LT, Scalar(Q, 3.0)),
    ))),
)


@pytest.fixture
async def matrix_table():
    """A real table holding the matrix, dropped afterwards.

    A temporary table cannot be used: the transaction pooler hands each
    statement to whichever backend is free, and a TEMP table exists only in the
    session that made it.
    """
    try:
        db = await get_database()
        await db.fetchval("select 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NOT VERIFIED: no database ({type(exc).__name__}). Engine equivalence UNCHECKED.")

    await db.execute(f'drop table if exists {TABLE}')
    await db.execute(
        f'create table {TABLE} ('
        '"id" text primary key, "grade" double precision, '
        '"credits" double precision, "passing" double precision)'
    )
    for record in MATRIX_RECORDS:
        grade = record.fields.get("grade")
        await db.execute(
            f'insert into {TABLE} ("id","grade","credits","passing") values ($1,$2,$3,$4)',
            record.fields["id"].value,
            grade.value if grade is not None else None,
            record.fields["credits"].value,
            record.fields["passing"].value,
        )
    try:
        yield db
    finally:
        await db.execute(f'drop table if exists {TABLE}')


@pytest.mark.parametrize("predicate", MATRIX_PREDICATES, ids=lambda p: type(p).__name__)
async def test_both_engines_select_the_same_rows(matrix_table, predicate) -> None:
    """The only proof that matters: same predicate, same rows, both engines."""
    in_memory = sorted(
        record.fields["id"].value for record in MATRIX_RECORDS if matches(predicate, record)
    )

    where, parameters = compile_to_sql(predicate)
    rows = await matrix_table.fetch(
        f'select "id" from {TABLE} where {where} order by "id"', *parameters
    )
    in_database = sorted(str(row["id"]) for row in rows)

    assert in_memory == in_database, (
        f"engines disagree on {predicate}: matches() -> {in_memory}, SQL -> {in_database}. "
        f"SQL was: where {where} with {parameters}"
    )


async def test_a_missing_field_survives_negation_in_both_engines(matrix_table) -> None:
    """Pinned on its own because it is the disagreement that would actually
    happen: `not (grade > 80)` must KEEP the row that has no grade."""
    predicate = Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80)))

    where, parameters = compile_to_sql(predicate)
    rows = await matrix_table.fetch(f'select "id" from {TABLE} where {where}', *parameters)

    assert "00940594" in {str(row["id"]) for row in rows}, "the grade-less row must survive"
    assert matches(predicate, MATRIX_RECORDS[3]) is True, "and matches() must agree"


async def test_no_value_is_ever_spliced_into_the_statement(matrix_table) -> None:
    """Every value leaves as a bound parameter, so nothing the model writes can
    become SQL. Checked with a string that would break out if it were spliced."""
    hostile = Comparison(Path.parse("id"), Op.EQ, Scalar(I, "x'; drop table " + TABLE + "; --"))
    where, parameters = compile_to_sql(hostile)

    assert "drop table" not in where.lower(), "the value must not appear in the statement text"
    assert parameters and "drop table" in str(parameters[0]).lower()

    rows = await matrix_table.fetch(f'select "id" from {TABLE} where {where}', *parameters)
    assert rows == []
    still_there = await matrix_table.fetchval(f'select count(*) from {TABLE}')
    assert still_there == len(MATRIX_RECORDS), "the table must still be standing"
