"""The most common wasted turn was a rendering the model never chose.

Across 38 measured runs the single most frequent refusal, twelve of them, was:

    the answer shows 00960211->00940224, 00960211->00940226, which are
    prerequisite EDGE ids, not courses

`check_no_edge_identifiers` is right to refuse it, and the loop cannot repair
it, because the model did not write those tokens. It slotted `{prereqs}` and the
renderer picked the record's FIRST field -- which for a `prerequisite_edges` row
is `edge`, the internal key.

"First" is an accident of the schema. The same accident was already fixed once
for `_id` and ObjectIds, by elimination; this fixes it by PREFERENCE, which also
picks the field the refusal asks for. A prerequisite edge is interesting for
what it REQUIRES.
"""

from __future__ import annotations

from app.agent_core.facts.answer import _readable_field
from app.agent_core.facts.types import Basis, Record, Scalar, ScalarKind

I = ScalarKind.IDENTIFIER
Q = ScalarKind.QUANTITY


def _record(**fields) -> Record:
    return Record(
        fields={k: Scalar(Q if isinstance(v, float) else I, v) for k, v in fields.items()},
        basis=Basis.OFFICIAL_RECORD,
    )


class TestTheRefusedRenderingIsGone:
    def test_a_prerequisite_edge_shows_what_it_requires(self) -> None:
        edge = _record(edge="00960211->00940224", course="00960211",
                       group="00960211.0", requires="00940224")
        assert _readable_field(edge) == "00940224"

    def test_the_result_would_not_be_refused(self) -> None:
        from app.agent_core.facts.postconditions import (
            check_no_edge_identifiers,
            check_no_group_identifiers,
        )

        edge = _record(edge="00960211->00940224", course="00960211",
                       group="00960211.0", requires="00940224")
        rendered = _readable_field(edge)
        assert check_no_edge_identifiers(rendered) == []
        assert check_no_group_identifiers(rendered) == []


class TestTheOtherShapesStillReadWell:
    def test_a_catalog_row_shows_its_course_number(self) -> None:
        assert _readable_field(
            _record(_id="6a3db0e382df7b7cb04552e8", courseNumber="00960211", title="x")
        ) == "00960211"

    def test_a_remaining_course_shows_its_number_not_the_user_id(self) -> None:
        assert _readable_field(
            _record(userId="6a578a2da43a2cfe1bcc791c", courseNumber="00940704", title="y")
        ) == "00940704"

    def test_an_offering_shows_the_course_not_the_key(self) -> None:
        assert _readable_field(
            _record(_id="6a3d0e382df7b7cb04552e8a", courseNumber="00940412",
                    semesterName="winter")
        ) == "00940412"

    def test_an_unfamiliar_record_still_renders_something(self) -> None:
        """The preference is an improvement, not a requirement -- a record shape
        nobody anticipated must not render as blank."""
        assert _readable_field(_record(zebra="something readable")) == "something readable"


class TestTheOldEliminationSurvives:
    def test_an_object_id_is_never_chosen(self) -> None:
        assert _readable_field(
            _record(_id="6a3db0e382df7b7cb04552e8", ref="6a3db0e382df7b7cb04552e9",
                    label="visible")
        ) == "visible"

    def test_an_edge_shaped_value_is_never_chosen(self) -> None:
        """Even under a field name the preference list does not know."""
        assert _readable_field(_record(link="00960211->00940224", label="visible")) == "visible"

    def test_a_group_shaped_value_is_never_chosen(self) -> None:
        assert _readable_field(_record(bucket="00970800.1", label="visible")) == "visible"

    def test_an_empty_record_renders_empty(self) -> None:
        assert _readable_field(Record(fields={}, basis=Basis.OFFICIAL_RECORD)) == ""
