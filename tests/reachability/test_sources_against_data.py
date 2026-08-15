"""The source registry is a CLAIM about real tables. This checks the claim.

Ported from `tests/pending_supabase/test_sources.py`, which asked Mongo the same
questions. A registry that only agrees with itself is a way of being confidently
wrong -- and this repo has already paid for that twice: `courses.academicYear`
was declared and exists on 0 of 2,613 rows, and `completed_courses.courseNumber`
was declared from an API input model rather than the stored document.

What did NOT survive the port: `TestObjectIdFilters`. It exercised the binding
pass that turned a model's string filter into a BSON ObjectId, and Postgres
stores those ids as text, so the pass was deleted rather than ported. Testing it
here would be testing code that no longer exists.

Skips per table when a table is empty -- an empty table cannot contradict a
schema, and pretending otherwise turns "nothing seeded" into a green tick.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from app.agent_core.facts.find import ArrayOf, Sub, _coerce
from app.agent_core.facts.sources import COMPLETED_COURSES, REGISTRY
from app.agent_core.facts.types import ScalarKind
from app.db.postgres import get_database

pytestmark = pytest.mark.supabase

KNOWN_ORPHANED_COMPLETED_COURSES = 155
"""Completed-course rows whose `courseId` matches no `courses` row.

Measured against the real database: 155 of 554 (28%). The records carry no
course number, title or offering id, so which course they represent is
unrecoverable -- the defect cannot be repaired, only contained.

A regression guard, deliberately not a skip: a skip is invisible in a green run,
and the entire value here is noticing the day a fifty-sixth appears.
"""


@pytest.fixture
async def database():
    try:
        db = await get_database()
        await db.fetchval("select 1")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"NOT VERIFIED: no database ({type(exc).__name__}). Schema claims UNCHECKED.")
    return db


def _scalar_leaves(row: Mapping, declared: Mapping, prefix: str = ""):
    """Every declared SCALAR in a row, nested ones included.

    Walks arrays and sub-documents because the schemas declare them: a version
    that only looked at top-level fields would silently stop checking
    `semesters[].goalCredits`, and an unchecked declaration is exactly the lie
    this exists to catch.
    """
    if not isinstance(row, Mapping):
        return
    for name, spec in declared.items():
        value = row.get(name)
        if value is None:
            continue
        path = f"{prefix}{name}"
        if isinstance(spec, ScalarKind):
            yield path, value, spec
        elif isinstance(spec, Sub):
            yield from _scalar_leaves(value, spec.fields, f"{path}.")
        elif isinstance(spec, ArrayOf) and isinstance(value, list):
            for element in value:
                if isinstance(spec.element, ScalarKind):
                    if element is not None:
                        yield path, element, spec.element
                else:
                    yield from _scalar_leaves(element, spec.element.fields, f"{path}.")


class TestTheSchemaMatchesTheTables:
    @pytest.mark.parametrize("name", sorted(REGISTRY), ids=sorted(REGISTRY))
    async def test_every_declared_field_coerces_on_real_rows(self, database, name) -> None:
        """A declared kind that will not coerce is a schema lie, and it surfaces
        as a SILENTLY ABSENT field at query time rather than as an error."""
        schema = REGISTRY[name]
        rows = await database.fetch(f'select * from {schema.collection} limit 50')
        if not rows:
            pytest.skip(f"'{schema.collection}' is empty -- nothing to check the schema against")

        wrong: list[str] = []
        for row in rows:
            for path, value, kind in _scalar_leaves(dict(row), schema.fields):
                if _coerce(value, kind) is None:
                    wrong.append(f"{path}={value!r} is not a {kind.value}")
        assert not wrong, f"{schema.collection}: {sorted(set(wrong))[:5]}"

    @pytest.mark.parametrize("name", sorted(REGISTRY), ids=sorted(REGISTRY))
    async def test_the_key_is_present_on_real_rows(self, database, name) -> None:
        """A key absent in practice makes `find` refuse the whole fetch."""
        schema = REGISTRY[name]
        rows = await database.fetch(f'select * from {schema.collection} limit 50')
        if not rows:
            pytest.skip(f"'{schema.collection}' is empty")
        missing = sum(1 for row in rows if dict(row).get(schema.key) is None)
        assert not missing, (
            f"{schema.collection}: '{schema.key}' absent on {missing}/{len(rows)} rows"
        )

    @pytest.mark.parametrize("name", sorted(REGISTRY), ids=sorted(REGISTRY))
    async def test_every_declared_field_is_a_real_column(self, database, name) -> None:
        """The failure the Mongo version could not have: Postgres has a schema,
        so a declared field with no column is knowable up front instead of
        showing up as absent on every row. `courses.academicYear` was exactly
        this -- declared, and on 0 of 2,613 documents."""
        schema = REGISTRY[name]
        columns = {
            row["column_name"]
            for row in await database.fetch(
                "select column_name from information_schema.columns where table_name = $1",
                schema.collection,
            )
        }
        if not columns:
            pytest.skip(f"'{schema.collection}' does not exist")
        declared = {n for n, spec in schema.fields.items() if isinstance(spec, ScalarKind)}
        assert declared <= columns, (
            f"{schema.collection} declares fields with no column: {sorted(declared - columns)}"
        )


class TestTheOrphanedReferencesAreContained:
    async def test_the_orphan_count_has_not_grown(self, database) -> None:
        total = await database.fetchval("select count(*) from completed_courses")
        if not total:
            pytest.skip("'completed_courses' is empty")
        found = await database.fetchval(
            'select count(*) from completed_courses cc '
            'left join courses c on c."_id" = cc."courseId" where c."_id" is null'
        )
        assert found <= KNOWN_ORPHANED_COMPLETED_COURSES, (
            f"orphaned completed-course references grew from "
            f"{KNOWN_ORPHANED_COMPLETED_COURSES} to {found} of {total}. Something is writing "
            "completed_courses with a courseId that matches no catalog row. The records carry no "
            "course identity, so they cannot be repaired after the fact -- find the writer."
        )

    async def test_the_quantities_survive_on_orphaned_rows(self, database) -> None:
        """The impact is narrower than "broken records" suggests: an unresolvable
        `courseId` costs course IDENTITY, not the quantities. A credit total over
        the whole transcript is still right; only the join to the catalog fails,
        and it fails closed."""
        orphans = await database.fetch(
            'select cc."grade", cc."creditsEarned", cc."semesterCode" from completed_courses cc '
            'left join courses c on c."_id" = cc."courseId" where c."_id" is null limit 50'
        )
        if not orphans:
            pytest.skip("no orphaned references present")
        missing = [
            field
            for field in ("grade", "creditsEarned", "semesterCode")
            for row in orphans
            if dict(row).get(field) is None
        ]
        assert not missing, f"orphaned rows are also missing quantities: {sorted(set(missing))}"


class TestTheShapeClaims:
    """Pure checks -- they need no database and would be true of any deployment."""

    def test_every_schema_declares_its_key_as_a_field(self) -> None:
        for name, schema in REGISTRY.items():
            assert schema.key in schema.fields, (
                f"{name} keys on '{schema.key}' but never declares it"
            )

    def test_completed_courses_does_not_claim_a_course_number(self) -> None:
        """It is not stored there. An earlier registry declared it, having been
        derived from the API's INPUT model rather than the stored row."""
        assert COMPLETED_COURSES.key == "courseId"
        assert "courseNumber" not in COMPLETED_COURSES.fields

    def test_a_course_code_is_an_identifier_not_a_number(self) -> None:
        """`00940224` is a code with a leading zero, not the quantity 940224."""
        assert REGISTRY["courses"].fields["courseNumber"] is ScalarKind.IDENTIFIER
