"""The source registry -- phase 11 of docs/agent/tools_implementation_plan.md.

These schemas are a CLAIM about real collections, so the tests check the claim
against the real documents rather than against the schema file. A registry that
only agrees with itself is a way of being confidently wrong.

Skips loudly without Mongo, and skips per-collection when a collection is empty:
an empty collection cannot contradict a schema, and pretending otherwise would
turn "no data loaded" into a green tick.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from tests.agent_core.ise_student_fixture import (  # noqa: F401 -- autouse fixture
    _fresh_mongo_client_per_test,
)
from app.agent_core.facts.find import _coerce
from app.agent_core.facts.sources import COMPLETED_COURSES, REGISTRY
from app.agent_core.facts.types import ScalarKind


class TestShape:
    def test_every_schema_declares_its_key_as_a_field(self) -> None:
        for name, schema in REGISTRY.items():
            assert schema.key in schema.fields, f"{name} keys on '{schema.key}' but never declares it"

    def test_completed_courses_does_not_claim_a_courseNumber(self) -> None:
        """It is not stored there -- 0 of 93 real documents carry one.

        An earlier version of this registry declared it, having been derived
        from the API's INPUT model rather than the stored document. A course
        code is only reachable by joining to `courses` on the ObjectId.
        """
        assert COMPLETED_COURSES.key == "courseId"
        assert "courseNumber" not in COMPLETED_COURSES.fields

    def test_grades_and_credits_are_quantities(self) -> None:
        fields = COMPLETED_COURSES.fields
        assert fields["grade"] is ScalarKind.QUANTITY
        assert fields["creditsEarned"] is ScalarKind.QUANTITY

    def test_a_course_code_is_an_identifier_not_a_number(self) -> None:
        from app.agent_core.facts.sources import COURSES

        assert COURSES.fields["courseNumber"] is ScalarKind.IDENTIFIER

    def test_an_objectid_coerces_to_an_identifier(self) -> None:
        """`completed_courses.courseId` is an ObjectId. Refusing it would make
        every transcript fetch fail at admission, since the key itself is one."""
        from bson import ObjectId

        assert _coerce(ObjectId("507f1f77bcf86cd799439011"), ScalarKind.IDENTIFIER) == "507f1f77bcf86cd799439011"

    def test_an_objectid_is_not_a_quantity(self) -> None:
        from bson import ObjectId

        assert _coerce(ObjectId("507f1f77bcf86cd799439011"), ScalarKind.QUANTITY) is None


class TestLiveMoodleSource:
    """`moodle_grades` is written by services/operator -- the first non-registrar source. It joins
    the catalog the same way `course_offerings` does, and carries the weaker LIVE_MOODLE basis."""

    def test_it_is_a_live_moodle_basis_joined_to_the_catalog_on_courseNumber(self) -> None:
        from app.agent_core.facts.sources import MOODLE_GRADES
        from app.agent_core.facts.types import Basis

        assert MOODLE_GRADES.basis is Basis.LIVE_MOODLE
        assert ("courseNumber", "courses.courseNumber") in MOODLE_GRADES.joins
        assert MOODLE_GRADES.fields["grade"] is ScalarKind.QUANTITY

    def test_userId_is_an_objectid_filter_target(self) -> None:
        """Per-student data: a string `userId` filter must bind to an ObjectId, or it silently
        matches nothing -- the confident-silence failure `TestObjectIdFilters` exists to prevent."""
        from app.agent_core.facts.sources import MOODLE_GRADES

        assert "userId" in MOODLE_GRADES.object_id_fields


class TestObjectIdFilters:
    """A predicate arrives with STRING values, because that is what a model can
    write. The fields it filters on are often stored as ObjectId.

    Found on the first live run: filtering `userId = "6a5cfb..."` matched nothing
    and returned an empty collection marked COMPLETE -- a student with no
    transcript, reported with full confidence. Silence is the worst failure this
    layer can produce, and it was in the layer built to prevent it.
    """

    def test_a_string_filter_binds_to_an_objectid_on_a_declared_field(self) -> None:
        from bson import ObjectId

        from app.agent_core.facts.find import _bind_object_ids
        from app.agent_core.facts.sources import COMPLETED_COURSES

        bound = _bind_object_ids({"userId": {"$eq": "507f1f77bcf86cd799439011"}}, COMPLETED_COURSES)
        assert bound["userId"]["$eq"] == ObjectId("507f1f77bcf86cd799439011")

    def test_undeclared_fields_are_left_alone(self) -> None:
        from app.agent_core.facts.find import _bind_object_ids
        from app.agent_core.facts.sources import COMPLETED_COURSES

        bound = _bind_object_ids({"semesterCode": {"$eq": "2025-2"}}, COMPLETED_COURSES)
        assert bound["semesterCode"]["$eq"] == "2025-2"

    def test_it_reaches_inside_in_and_boolean_operators(self) -> None:
        """The rewrite walks the COMPILED filter, so it covers everything the
        predicate grammar compiles into, not just top-level equality."""
        from bson import ObjectId

        from app.agent_core.facts.find import _bind_object_ids
        from app.agent_core.facts.sources import COMPLETED_COURSES

        oid = "507f1f77bcf86cd799439011"
        bound = _bind_object_ids(
            {"$and": [{"userId": {"$in": [oid]}}, {"semesterCode": {"$eq": "2025-2"}}]},
            COMPLETED_COURSES,
        )
        assert bound["$and"][0]["userId"]["$in"] == [ObjectId(oid)]
        assert bound["$and"][1]["semesterCode"]["$eq"] == "2025-2"

    def test_a_non_objectid_string_is_not_forced(self) -> None:
        """A field may hold both forms during a migration. A value that is not
        ObjectId-shaped should return nothing, not raise."""
        from app.agent_core.facts.find import _bind_object_ids
        from app.agent_core.facts.sources import COMPLETED_COURSES

        bound = _bind_object_ids({"userId": {"$eq": "not-an-objectid"}}, COMPLETED_COURSES)
        assert bound["userId"]["$eq"] == "not-an-objectid"

    def test_a_schema_declaring_none_is_untouched(self) -> None:
        from app.agent_core.facts.find import SourceSchema, _bind_object_ids
        from app.agent_core.facts.types import Basis

        plain = SourceSchema("x", "id", {"id": ScalarKind.IDENTIFIER}, Basis.OFFICIAL_RECORD)
        query = {"id": {"$eq": "507f1f77bcf86cd799439011"}}
        assert _bind_object_ids(query, plain) == query


@pytest.fixture
async def database():
    # Through the agent's own settings, not a bespoke env var: an earlier
    # version read MONGODB_URI, which this project does not set, so these
    # skipped on every machine while looking like passes.
    from app.db.mongo import get_database

    try:
        database_handle = await get_database()
        await database_handle.command("ping")
        client = database_handle.client
    except Exception as exc:  # noqa: BLE001
        pytest.skip(
            f"NOT VERIFIED: no database ({type(exc).__name__}). "
            "The source schemas are UNCHECKED against real documents in this run."
        )
    yield client[os.environ.get("MONGO_DB", "unipilot_python")]
    client.close()


def _scalar_leaves(document, declared, prefix: str = ""):
    """Every declared SCALAR in a document, nested ones included.

    Walks arrays and sub-documents because the schemas now declare them: a
    version of this that only looked at top-level fields would silently stop
    checking `semesters[].goalCredits` -- and an unchecked declaration is
    exactly the schema lie this test exists to catch.
    """
    from app.agent_core.facts.find import ArrayOf, Sub

    if not isinstance(document, Mapping):
        return
    for name, spec in declared.items():
        value = document.get(name)
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


class TestNestedDeclarations:
    """`semester_plans` is the only source declaring nested structure, and it is
    the one `optimize` depends on. These pin the shape the route relies on."""

    def test_semesters_is_declared_as_an_array_of_sub_documents(self) -> None:
        from app.agent_core.facts.find import ArrayOf, Sub
        from app.agent_core.facts.sources import SEMESTER_PLANS

        semesters = SEMESTER_PLANS.fields["semesters"]
        assert isinstance(semesters, ArrayOf) and isinstance(semesters.element, Sub)
        # `order` is the slot index and `goalCredits` the capacity. Without both,
        # a plan unnests into slots that cannot be placed into.
        assert {"order", "goalCredits", "semesterCode"} <= set(semesters.element.fields)

    def test_nested_paths_are_reachable_by_name(self) -> None:
        """The unknown-field message lists these, so a model can find them."""
        from app.agent_core.facts.find import declared_paths
        from app.agent_core.facts.sources import SEMESTER_PLANS

        paths = declared_paths(SEMESTER_PLANS)
        assert "semesters.order" in paths
        assert "semesters.plannedCourses.courseNumber" in paths


class TestAgainstRealDocuments:
    @pytest.mark.parametrize("name", sorted(REGISTRY), ids=sorted(REGISTRY))
    async def test_declared_fields_coerce_on_real_documents(self, database, name) -> None:
        """Every declared field must be readable as its declared kind wherever
        it is present. A `QUANTITY` that will not coerce is a schema lie that
        would surface as a silently absent field at query time."""
        schema = REGISTRY[name]
        documents = [doc async for doc in database[schema.collection].find({}).limit(50)]
        if not documents:
            pytest.skip(f"'{schema.collection}' is empty -- nothing to check the schema against")

        wrong: list[str] = []
        for document in documents:
            for path, value, kind in _scalar_leaves(document, schema.fields):
                if _coerce(value, kind) is None:
                    wrong.append(f"{path}={value!r} is not a {kind.value}")

        assert not wrong, f"{schema.collection}: {sorted(set(wrong))[:5]}"

    @pytest.mark.parametrize("name", sorted(REGISTRY), ids=sorted(REGISTRY))
    async def test_the_key_is_present_on_real_documents(self, database, name) -> None:
        """A key absent in practice means `find` refuses the whole fetch, which
        is a very loud way to discover the wrong key was chosen."""
        schema = REGISTRY[name]
        documents = [doc async for doc in database[schema.collection].find({}).limit(50)]
        if not documents:
            pytest.skip(f"'{schema.collection}' is empty")

        missing = sum(1 for doc in documents if doc.get(schema.key) is None)
        assert not missing, f"{schema.collection}: '{schema.key}' absent on {missing}/{len(documents)} documents"

KNOWN_ORPHANED_COMPLETED_COURSES = 155
"""Completed-course records whose `courseId` matches no `courses` document.

Measured 2026-07-19 against the real database: 155 of 554 (28%), across 138
phantom ObjectIds grouped into 29 process-runs on 2026-07-13. An earlier figure
of "14 of 93" came from the local `mongo` CONTAINER, which holds a stale partial
copy -- auditing the wrong database returned a number that looked like an answer. Both were minted in the same window as the catalog seed (2026-06-23/26)
and the records carry NO metadata -- no course number, title, or offering id --
so which course they represent is unrecoverable from the data.

Neither known write path produced them: the completed-course route has validated
`courseId` against the catalog since 2026-06-20, and transcript import both
validates and stamps `metadata.importSource`, which these lack. The writer is
unidentified.

The number is pinned rather than tolerated. Growth means the unknown writer ran
again, which is the only thing anyone can act on without knowing what those two
courses were.

THE ENFORCING CHECK IS NOT HERE. Dev Mongo publishes no host port and the service
image ships no tests, so this case skips on every machine -- and a check that
always skips is a green tick meaning nothing. The real guard is
`scripts/audit_orphaned_course_references.py`, which reaches the database through
the running container and exits non-zero when the count grows. This stays as
documentation, and as a live check for anyone who does have a reachable database.
"""


class TestOrphanedCourseReferences:
    async def test_the_orphan_count_has_not_grown(self, database) -> None:
        """A regression guard on a defect that cannot be repaired, only contained.

        Deliberately not a skip. A skip is invisible in a green run, and the
        entire value here is noticing the day a fifteenth appears.
        """
        pipeline = [
            {"$lookup": {"from": "courses", "localField": "courseId", "foreignField": "_id", "as": "c"}},
            {"$match": {"c": {"$size": 0}}},
            {"$count": "n"},
        ]
        rows = [row async for row in database["completed_courses"].aggregate(pipeline)]
        found = rows[0]["n"] if rows else 0

        total = await database["completed_courses"].count_documents({})
        if total == 0:
            pytest.skip("'completed_courses' is empty")

        assert found <= KNOWN_ORPHANED_COMPLETED_COURSES, (
            f"orphaned completed-course references grew from {KNOWN_ORPHANED_COMPLETED_COURSES} "
            f"to {found} of {total}. Something is still writing completed_courses with a courseId "
            "that matches no catalog document, and neither known write path can do that -- both "
            "validate. Find the writer before the count grows again; the records carry no course "
            "identity, so they cannot be repaired after the fact."
        )

    async def test_grades_and_credits_survive_on_orphaned_records(self, database) -> None:
        """The impact is narrower than 'broken records' suggests.

        An unresolvable `courseId` costs course IDENTITY, not the quantities. A
        credit total over the whole transcript is still correct; only a join to
        the catalog fails -- and it fails closed, which is why the wrong answer
        never reaches anyone.
        """
        pipeline = [
            {"$lookup": {"from": "courses", "localField": "courseId", "foreignField": "_id", "as": "c"}},
            {"$match": {"c": {"$size": 0}}},
            {"$limit": 50},
        ]
        orphans = [row async for row in database["completed_courses"].aggregate(pipeline)]
        if not orphans:
            pytest.skip("no orphaned references present")

        missing = [
            field
            for field in ("grade", "creditsEarned", "semesterCode")
            for row in orphans
            if row.get(field) is None
        ]
        assert not missing, f"orphaned records are also missing quantities: {sorted(set(missing))}"


class TestSemiJoin:
    """`_id IN {"fact": collection, "field": f}` -- fetch the records a held
    collection points at, without pulling a whole catalog to join in memory.

    Two live evals stalled here: the model held 17 course ids and had no way to
    fetch the courses they referenced except `find(courses, limit=3000)` then a
    join, which it never discovered. The natural spelling did not parse.
    """

    def _completed(self):
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        records = tuple(
            Record(fields={"courseId": Scalar(ScalarKind.IDENTIFIER, cid)}, basis=Basis.OFFICIAL_RECORD)
            for cid in ("507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012", "507f1f77bcf86cd799439011")
        )
        return HeldFact(
            value=Collection(records=records, completeness=Completeness(complete=True, total=3)),
            basis=Basis.OFFICIAL_RECORD,
        )

    def _resolve(self, predicate_json, facts):
        from app.agent_core.facts.codec import parse_predicate
        from app.agent_core.facts.dispatch import DispatchContext, _resolve_fact_refs

        context = DispatchContext(facts=facts)
        return _resolve_fact_refs(parse_predicate(predicate_json), context)

    def test_it_resolves_to_the_distinct_set_of_field_values(self) -> None:
        from app.agent_core.facts.predicate import Comparison, Op

        resolved = self._resolve(
            {"path": "_id", "op": "in", "value": {"fact": "completed", "field": "courseId"}},
            {"completed": self._completed()},
        )
        assert isinstance(resolved, Comparison) and resolved.op is Op.IN
        # Three records, two distinct ids -- the duplicate collapses.
        assert sorted(s.value for s in resolved.value) == ["507f1f77bcf86cd799439011", "507f1f77bcf86cd799439012"]

    def test_in_against_a_held_fact_without_a_field_is_a_repairable_error(self) -> None:
        from app.agent_core.facts.codec import ParseError

        with pytest.raises(ParseError) as caught:
            self._resolve({"path": "_id", "op": "in", "value": {"fact": "completed"}}, {"completed": self._completed()})
        assert "field" in str(caught.value)

    def test_a_scalar_op_on_a_single_record_extracts_the_one_value(self) -> None:
        """The shape the model reached for turn after turn: `course = {"fact":
        "next_course", "field": "courseNumber"}` where next_course holds one
        record. This is one-record extraction, like `only`."""
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.predicate import Comparison
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        one = HeldFact(
            value=Collection(
                records=(Record(fields={"courseNumber": Scalar(ScalarKind.IDENTIFIER, "00960211")}, basis=Basis.OFFICIAL_RECORD),),
                completeness=Completeness(complete=True, total=1),
            ),
            basis=Basis.OFFICIAL_RECORD,
        )
        resolved = self._resolve(
            {"path": "course", "op": "=", "value": {"fact": "next_course", "field": "courseNumber"}},
            {"next_course": one},
        )
        assert isinstance(resolved, Comparison)
        assert resolved.value.value == "00960211"

    def test_a_scalar_op_on_a_multi_record_fact_is_refused_with_guidance(self) -> None:
        from app.agent_core.facts.operators import ExpressionDefect

        resolved = self._resolve(
            {"path": "_id", "op": "=", "value": {"fact": "completed", "field": "courseId"}},
            {"completed": self._completed()},  # two distinct ids
        )
        assert isinstance(resolved, ExpressionDefect)
        assert "in" in resolved.message and "one" in resolved.message

    def test_a_missing_field_fails_closed_rather_than_shrinking_the_set(self) -> None:
        from app.agent_core.facts.answer import HeldFact
        from app.agent_core.facts.operators import ExpressionDefect
        from app.agent_core.facts.types import Basis, Collection, Completeness, Record, Scalar, ScalarKind

        held = HeldFact(
            value=Collection(
                records=(Record(fields={"other": Scalar(ScalarKind.IDENTIFIER, "x")}, basis=Basis.OFFICIAL_RECORD),),
                completeness=Completeness(complete=True, total=1),
            ),
            basis=Basis.OFFICIAL_RECORD,
        )
        resolved = self._resolve(
            {"path": "_id", "op": "in", "value": {"fact": "c", "field": "courseId"}}, {"c": held}
        )
        assert isinstance(resolved, ExpressionDefect)
        assert "courseId" in resolved.message
