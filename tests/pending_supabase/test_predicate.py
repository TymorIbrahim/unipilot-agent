"""Phase 2 gate for docs/agent/tools_implementation_plan.md.

One grammar, two engines: `select` evaluates it in memory, `find` pushes it down
to Mongo. Two implementations of one grammar drift, and the drift is silent --
so the same MATRIX below feeds every test in this file:

  - in-memory semantics are pinned unconditionally
  - the compiled filter document is checked structurally, unconditionally
  - true cross-engine equivalence runs against a real Mongo when one is
    reachable, and skips LOUDLY when it is not

The last one is the only real proof, but it cannot be the only test, or the
semantics go unchecked on every machine without a database.
"""

from __future__ import annotations

import pytest

from tests.agent_core.ise_student_fixture import (  # noqa: F401 -- autouse fixture
    _fresh_mongo_client_per_test,
)

from app.agent_core.facts.predicate import (
    Always,
    And,
    Comparison,
    Not,
    Op,
    Or,
    Path,
    PredicateTypeError,
    compile_to_mongo,
    matches,
    validate,
)
from app.agent_core.facts.types import Basis, Record, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER
T = ScalarKind.TEXT


def _record(**fields: object) -> Record:
    typed = {}
    for name, value in fields.items():
        if isinstance(value, bool):
            typed[name] = Scalar(ScalarKind.BOOL, value)
        elif isinstance(value, (int, float)):
            typed[name] = Scalar(Q, value)
        elif isinstance(value, tuple):
            typed[name] = tuple(Scalar(I, item) for item in value)
        else:
            typed[name] = Scalar(I, value)
    return Record(fields=typed, basis=Basis.OFFICIAL_RECORD)


# The shared matrix. Every engine sees exactly this.
MATRIX_RECORDS = (
    _record(id="00940224", grade=95, credits=3.5, passing=60, tags=("core", "cs")),
    _record(id="00960211", grade=60, credits=3.0, passing=60, tags=("elective",)),
    _record(id="00970800", grade=88, credits=4.0, passing=70, tags=("core",)),
    _record(id="00940594", credits=2.5, passing=60, tags=()),  # no `grade` at all
)

MATRIX_PREDICATES = (
    Always(),
    Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)),
    Comparison(Path.parse("grade"), Op.GE, Scalar(Q, 88)),
    Comparison(Path.parse("grade"), Op.LT, Scalar(Q, 88)),
    Comparison(Path.parse("id"), Op.EQ, Scalar(I, "00940224")),
    Comparison(Path.parse("id"), Op.NE, Scalar(I, "00940224")),
    Comparison(Path.parse("id"), Op.IN, (Scalar(I, "00940224"), Scalar(I, "00970800"))),
    Comparison(Path.parse("tags"), Op.CONTAINS, Scalar(I, "core")),
    Comparison(Path.parse("grade"), Op.GT, Path.parse("passing")),
    And((Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80)), Comparison(Path.parse("tags"), Op.CONTAINS, Scalar(I, "core")))),
    Or((Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)), Comparison(Path.parse("credits"), Op.LT, Scalar(Q, 3.0)))),
    Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80))),
    # Composite negation. Added after a mutation test showed the matrix could not
    # distinguish `$nor` from Mongo's field-level `$not`: over a SINGLE comparison
    # the two agree (both match documents missing the field), so a wrong compiler
    # passed unnoticed. They diverge only here -- field-level `$not` cannot negate
    # a composite at all.
    Not(And((Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80)), Comparison(Path.parse("tags"), Op.CONTAINS, Scalar(I, "core"))))),
    Not(Or((Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)), Comparison(Path.parse("credits"), Op.LT, Scalar(Q, 3.0))))),
)


class TestPathResolution:
    def test_a_dotted_path_reaches_a_nested_field(self) -> None:
        nested = Record(
            fields={"profile": Record(fields={"year": Scalar(Q, 3)}, basis=Basis.OFFICIAL_RECORD)},
            basis=Basis.OFFICIAL_RECORD,
        )
        assert matches(Comparison(Path.parse("profile.year"), Op.EQ, Scalar(Q, 3)), nested)

    def test_a_missing_field_does_not_match_rather_than_raising(self) -> None:
        """Mongo's behaviour: a comparison against an absent field simply fails.
        The in-memory engine must agree or the two diverge on every sparse record."""
        absent = Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 50))
        assert matches(absent, MATRIX_RECORDS[3]) is False

    def test_a_missing_field_does_not_match_a_negated_comparison_either(self) -> None:
        """The subtle half: NOT(grade > 50) over a record with no grade.

        Mongo's $nor excludes documents the inner expression did not match, so a
        record missing the field DOES come back. Pinning it here because getting
        this wrong flips a whole result set."""
        assert matches(Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 50))), MATRIX_RECORDS[3]) is True


class TestTypeRules:
    def test_ordering_an_identifier_is_a_type_error(self) -> None:
        """Course codes have no order. Permitting `<` on them invites summing
        them next, and Mongo would happily compare them bytewise."""
        with pytest.raises(PredicateTypeError):
            validate(Comparison(Path.parse("id"), Op.LT, Scalar(I, "00940224")))

    def test_ordering_against_text_is_a_type_error(self) -> None:
        """Mongo orders ACROSS BSON types rather than refusing, so `grade > "ninety"`
        silently returns something there. We refuse to emit one."""
        with pytest.raises(PredicateTypeError):
            validate(Comparison(Path.parse("grade"), Op.GT, Scalar(T, "ninety")))

    def test_equality_is_allowed_on_unorderable_kinds(self) -> None:
        validate(Comparison(Path.parse("id"), Op.EQ, Scalar(I, "00940224")))

    def test_contains_against_a_non_collection_field_does_not_match(self) -> None:
        """`validate` sees no data, so it cannot know a field holds a collection --
        that check belongs to the pipeline type checker (phase 3), where the
        collection's field types are known. At runtime it simply fails to match,
        which is what Mongo does."""
        assert matches(Comparison(Path.parse("id"), Op.CONTAINS, Scalar(I, "core")), MATRIX_RECORDS[0]) is False


class TestAlways:
    def test_always_matches_every_record(self) -> None:
        """Required by §3.3: `join` expresses Cartesian product only if the
        grammar admits a constant-true predicate. Without this the basis is not
        relationally complete."""
        assert all(matches(Always(), record) for record in MATRIX_RECORDS)

    def test_always_compiles_to_an_empty_filter(self) -> None:
        assert compile_to_mongo(Always()) == {}


class TestInMemorySemantics:
    """Pinned expectations for the shared matrix. These run everywhere."""

    @pytest.mark.parametrize(
        ("predicate_index", "expected_ids"),
        [
            (0, ["00940224", "00960211", "00970800", "00940594"]),  # Always
            (1, ["00940224"]),                                       # grade > 90
            (2, ["00940224", "00970800"]),                           # grade >= 88
            (3, ["00960211"]),                                       # grade < 88
            (4, ["00940224"]),                                       # id == ...
            (5, ["00960211", "00970800", "00940594"]),               # id != ...
            (6, ["00940224", "00970800"]),                           # id in {...}
            (7, ["00940224", "00970800"]),                           # tags contains core
            (8, ["00940224", "00970800"]),                           # grade > passing (field vs field)
            (9, ["00940224", "00970800"]),                           # and
            (10, ["00940224", "00940594"]),                          # or (00960211 has credits == 3.0, not < 3.0)
            (11, ["00960211", "00940594"]),                          # not
            (12, ["00960211", "00940594"]),                          # not(and(...))
            (13, ["00960211", "00970800"]),                          # not(or(...))
        ],
    )
    def test_matrix(self, predicate_index: int, expected_ids: list[str]) -> None:
        predicate = MATRIX_PREDICATES[predicate_index]
        got = [r.fields["id"].value for r in MATRIX_RECORDS if matches(predicate, r)]
        assert got == expected_ids


class TestMongoCompilation:
    def test_every_matrix_predicate_compiles(self) -> None:
        for predicate in MATRIX_PREDICATES:
            assert isinstance(compile_to_mongo(predicate), dict)

    def test_comparison_compiles_to_an_operator_document(self) -> None:
        compiled = compile_to_mongo(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 90)))
        assert compiled == {"grade": {"$gt": 90}}

    def test_nested_path_compiles_to_dotted_notation(self) -> None:
        compiled = compile_to_mongo(Comparison(Path.parse("profile.year"), Op.EQ, Scalar(Q, 3)))
        assert compiled == {"profile.year": {"$eq": 3}}

    def test_field_to_field_comparison_uses_expr(self) -> None:
        """A literal cannot express `grade > passing`; Mongo needs $expr."""
        compiled = compile_to_mongo(Comparison(Path.parse("grade"), Op.GT, Path.parse("passing")))
        assert compiled == {"$expr": {"$gt": ["$grade", "$passing"]}}

    def test_negation_uses_nor_not_field_level_not(self) -> None:
        """Mongo's $not is field-scoped and cannot negate a composite; $nor can."""
        compiled = compile_to_mongo(Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 80))))
        assert compiled == {"$nor": [{"grade": {"$gt": 80}}]}

    def test_compilation_never_emits_a_regex(self) -> None:
        """`contains` is array-membership only. Text-contains would mean
        compiling user-supplied text into a regex -- an injection surface and a
        semantics divergence from the in-memory engine. Excluded by design."""
        for predicate in MATRIX_PREDICATES:
            assert "$regex" not in repr(compile_to_mongo(predicate))


def _as_document(record: Record) -> dict:
    """The same record as a plain Mongo document. Absent fields stay absent --
    the sparse record is the whole point of the missing-field cases."""
    document = {}
    for name, value in record.fields.items():
        document[name] = [item.value for item in value] if isinstance(value, tuple) else value.value
    return document


class TestCrossEngineEquivalence:
    """THE phase-2 gate: one grammar, two engines, identical results.

    Skips when no Mongo is reachable -- and says so rather than passing quietly,
    because a green suite that never ran this proves nothing about the property
    it exists to protect.
    """

    async def test_every_matrix_predicate_agrees_across_engines(self) -> None:
        # Through the agent's own settings, not a bespoke env var. This read
        # `MONGODB_URI`, which the project does not set, so it skipped on every
        # machine while looking like a pass -- the same wrong-env-var bug that
        # had already been fixed in the other database-backed files, left here.
        #
        # It matters more than it used to. `find` now runs the IN-MEMORY engine
        # against derived sources the model can query, so the two engines
        # agreeing is load-bearing on a production path rather than a property
        # only `select` relied on.
        from app.db.mongo import get_database

        try:
            handle = await get_database()
            await handle.command("ping")
            client = handle.client
        except Exception as exc:  # noqa: BLE001 -- any connection failure is a skip
            pytest.skip(
                f"NOT VERIFIED: no database ({type(exc).__name__}). "
                "Push-down and in-memory predicate semantics are UNCHECKED against each "
                "other in this run -- bring dev Mongo up to close the gate."
            )

        collection = client["unipilot_predicate_equivalence"]["matrix"]
        try:
            await collection.delete_many({})
            await collection.insert_many([_as_document(record) for record in MATRIX_RECORDS])

            divergences = []
            for predicate in MATRIX_PREDICATES:
                in_memory = sorted(
                    r.fields["id"].value for r in MATRIX_RECORDS if matches(predicate, r)
                )
                cursor = collection.find(compile_to_mongo(predicate), {"_id": 0, "id": 1})
                pushed_down = sorted([doc["id"] async for doc in cursor])
                if in_memory != pushed_down:
                    divergences.append(
                        f"\n  {predicate}\n    in-memory:  {in_memory}\n    pushed down: {pushed_down}"
                    )

            assert not divergences, "the two engines disagree:" + "".join(divergences)
        finally:
            await client.drop_database("unipilot_predicate_equivalence")
            client.close()
