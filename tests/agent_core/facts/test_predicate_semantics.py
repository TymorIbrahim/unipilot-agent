"""What the predicate grammar means, before any engine gets hold of it.

The other half of `tests/pending_supabase/test_predicate.py`. Its cross-engine
matrix is already ported -- `tests/reachability/test_predicate_engines.py` runs
it through `matches` and `compile_to_sql` against a real database. These are the
rules that need no engine and no database at all: path resolution, the type
rules, and the constant-true predicate.

They were the part in most danger of being lost. The matrix is obviously
valuable and was ported first; a handful of `validate` tests look like plumbing
until you notice that nothing else in the suite asserts a course code cannot be
ordered, and that `<` on identifiers is exactly the kind of thing a SQL engine
does happily and byte-wise rather than refusing.

`TestMongoCompilation` was NOT ported. It pinned `$nor` versus field-level
`$not` on a compiler that has been deleted; its equivalent for the SQL engine
lives with the matrix, where a real database can disagree with `matches`.
"""

from __future__ import annotations

import pytest

from app.agent_core.facts.predicate import (
    Always,
    Comparison,
    Not,
    Op,
    Path,
    PredicateTypeError,
    matches,
    validate,
)
from app.agent_core.facts.types import Basis, Record, Scalar, ScalarKind

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER
T = ScalarKind.TEXT


def _record(**fields: object) -> Record:
    typed: dict = {}
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


WITH_GRADE = _record(id="00940224", grade=95, credits=3.5, passing=60, tags=("core", "cs"))
NO_GRADE = _record(id="00940594", credits=2.5, passing=60, tags=())


class TestPathResolution:
    def test_a_dotted_path_reaches_a_nested_field(self) -> None:
        nested = Record(
            fields={"profile": Record(fields={"year": Scalar(Q, 3)}, basis=Basis.OFFICIAL_RECORD)},
            basis=Basis.OFFICIAL_RECORD,
        )
        assert matches(Comparison(Path.parse("profile.year"), Op.EQ, Scalar(Q, 3)), nested)

    def test_a_missing_field_does_not_match_rather_than_raising(self) -> None:
        """A comparison against an absent field simply fails. Raising instead
        would make one sparse record end a whole fetch."""
        assert matches(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 50)), NO_GRADE) is False

    def test_a_missing_field_does_not_match_a_negated_comparison_either(self) -> None:
        """The subtle half: NOT(grade > 50) over a record with no grade KEEPS it.

        Pinned because getting it wrong flips a whole result set, and because it
        is precisely where the SQL engine wants to disagree -- a comparison
        against NULL is NULL, not false, so without `coalesce(..., false)` at
        every leaf the negation drops this record while `matches` keeps it.
        """
        assert matches(Not(Comparison(Path.parse("grade"), Op.GT, Scalar(Q, 50))), NO_GRADE) is True


class TestTypeRules:
    def test_ordering_an_identifier_is_a_type_error(self) -> None:
        """Course codes have no order. Permitting `<` on them invites summing
        them next, and a SQL engine would compare them byte-wise without
        complaint."""
        with pytest.raises(PredicateTypeError):
            validate(Comparison(Path.parse("id"), Op.LT, Scalar(I, "00940224")))

    def test_ordering_against_text_is_a_type_error(self) -> None:
        """`grade > "ninety"` must not be expressible. A database asked this
        returns SOMETHING rather than refusing, and something is the problem."""
        with pytest.raises(PredicateTypeError):
            validate(Comparison(Path.parse("grade"), Op.GT, Scalar(T, "ninety")))

    def test_equality_is_allowed_on_unorderable_kinds(self) -> None:
        """The rule is about ORDER, not about identifiers being untouchable."""
        validate(Comparison(Path.parse("id"), Op.EQ, Scalar(I, "00940224")))

    def test_contains_against_a_non_collection_field_does_not_match(self) -> None:
        """`validate` sees no data, so it cannot know whether a field holds a
        collection -- that check belongs to the pipeline type checker, where the
        field types are known. At runtime it simply fails to match."""
        assert matches(Comparison(Path.parse("id"), Op.CONTAINS, Scalar(I, "core")), WITH_GRADE) is False


class TestAlways:
    def test_always_matches_every_record(self) -> None:
        """`join` expresses a Cartesian product only if the grammar admits a
        constant-true predicate. Without it the basis is not relationally
        complete."""
        assert all(matches(Always(), record) for record in (WITH_GRADE, NO_GRADE))

    def test_always_survives_validation(self) -> None:
        validate(Always())
