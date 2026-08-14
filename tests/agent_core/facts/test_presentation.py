"""Fact presentation -- phase 9c of docs/agent/tools_implementation_plan.md.

The model writes pipelines against what it believes it holds, so a wrong or thin
rendering causes failures that look like model mistakes and are not.

The tests that matter most are about what the rendering PREVENTS: a pipeline
against a field that does not exist, an aggregate that was always going to be
refused, and a fact that reads as empty when it is not.
"""

from __future__ import annotations

from app.agent_core.facts.presentation import render_facts
from app.agent_core.facts.types import (
    Basis,
    Collection,
    Completeness,
    Record,
    Scalar,
    ScalarKind,
)

Q = ScalarKind.QUANTITY
I = ScalarKind.IDENTIFIER


def _rec(basis: Basis = Basis.OFFICIAL_RECORD, **fields) -> Record:
    typed = {
        name: Scalar(Q, value) if isinstance(value, (int, float)) else Scalar(I, str(value))
        for name, value in fields.items()
    }
    return Record(fields=typed, basis=basis)


COMPLETE = Collection(
    records=(
        _rec(id="00940224", grade=95, credits=3.5),
        _rec(id="00960211", grade=60, credits=3.0),
    ),
    completeness=Completeness(complete=True, total=2),
)


class TestFieldsAndKinds:
    def test_field_names_and_kinds_are_shown(self) -> None:
        """Without kinds the model cannot tell which fields `arith` may touch,
        and a course code looks exactly like a number."""
        rendered = render_facts({"courses": COMPLETE})
        assert "id: identifier" in rendered
        assert "credits: quantity" in rendered

    def test_fields_are_the_UNION_across_records_not_a_sample(self) -> None:
        """Sampling the first record under-reports the moment data is uneven --
        and uneven is normal, because a missing field is what dirty upstream
        data produces."""
        uneven = Collection(
            records=(_rec(id="a", grade=90), _rec(id="b", credits=3.0)),
            completeness=Completeness(complete=True, total=2),
        )
        rendered = render_facts({"c": uneven})
        assert "grade" in rendered and "credits" in rendered

    def test_partial_coverage_warns_that_aggregates_will_refuse(self) -> None:
        """The whole point: predict the refusal instead of discovering it. A
        model told this fetches the missing values; a model not told burns a
        turn finding out."""
        uneven = Collection(
            records=(_rec(id="a", grade=90), _rec(id="b")),
            completeness=Completeness(complete=True, total=2),
        )
        rendered = render_facts({"c": uneven})
        assert "on 1 of 2" in rendered
        assert "refuse" in rendered

    def test_a_field_with_two_kinds_is_flagged_rather_than_resolved(self) -> None:
        """An operator will succeed on some rows and fail on others, which is
        the most confusing failure there is. Better to show it."""
        mixed = Collection(
            records=(
                Record(fields={"x": Scalar(Q, 1)}, basis=Basis.OFFICIAL_RECORD),
                Record(fields={"x": Scalar(I, "one")}, basis=Basis.OFFICIAL_RECORD),
            ),
            completeness=Completeness(complete=True, total=2),
        )
        assert "quantity|identifier" in render_facts({"c": mixed})


class TestCompleteness:
    def test_a_complete_collection_says_so_plainly(self) -> None:
        assert "[2 records]" in render_facts({"courses": COMPLETE})

    def test_a_truncated_collection_states_the_CONSEQUENCE(self) -> None:
        """'Truncated' means nothing to a caller who does not know what it
        forbids. The refusal is what changes behaviour."""
        page = Collection(records=COMPLETE.records, completeness=Completeness(complete=False, total=73))
        rendered = render_facts({"plans": page})
        assert "of 73" in rendered
        assert "refused" in rendered

    def test_an_empty_collection_is_distinguishable_from_a_missing_one(self) -> None:
        empty = Collection(records=(), completeness=Completeness(complete=True, total=0))
        rendered = render_facts({"none": empty})
        assert "0 records" in rendered
        assert "yields nothing" in rendered

    def test_no_facts_at_all_renders_as_none_rather_than_blank(self) -> None:
        assert "none yet" in render_facts({})


class TestProvenance:
    def test_the_basis_is_visible(self) -> None:
        assert "official_record" in render_facts({"c": COMPLETE})

    def test_mixed_provenance_shows_both(self) -> None:
        mixed = Collection(
            records=(_rec(id="a"), _rec(basis=Basis.SIMULATED, id="b")),
            completeness=Completeness(complete=True, total=2),
        )
        rendered = render_facts({"c": mixed})
        assert "official_record" in rendered and "simulated" in rendered


class TestScalars:
    def test_a_scalar_shows_its_value_and_kind(self) -> None:
        rendered = render_facts({"total": Scalar(Q, 62.5)})
        assert "62.5" in rendered and "quantity" in rendered


class TestContextDiscipline:
    def test_the_payload_is_never_inlined(self) -> None:
        """A working set that inlines its data stops being a working set. This
        renders shapes and counts; the rows stay where they are."""
        big = Collection(
            records=tuple(_rec(id=f"course-{n}", grade=n) for n in range(500)),
            completeness=Completeness(complete=True, total=500),
        )
        rendered = render_facts({"everything": big})
        assert len(rendered) < 400, f"rendering grew with the data ({len(rendered)} chars)"
        assert "500 records" in rendered

    def test_a_sample_value_is_shown_so_the_fact_does_not_read_as_empty(self) -> None:
        """Measured previously: a bare '[list of N items]' reads as 'nothing
        here', and a sub-loop gave up on a list it was actually holding."""
        rendered = render_facts({"courses": COMPLETE})
        assert "00940224" in rendered

    def test_a_wide_record_does_not_print_every_field(self) -> None:
        wide = Collection(
            records=(_rec(**{f"f{n}": n for n in range(40)}),),
            completeness=Completeness(complete=True, total=1),
        )
        rendered = render_facts({"wide": wide})
        assert "more fields" in rendered
