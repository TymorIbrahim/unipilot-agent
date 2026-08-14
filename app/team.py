"""Team identity for `GET /api/team_info`.

Hard-coded rather than configured: this is submission metadata, it never varies
by environment, and a missing env var must not be able to turn the graded
identity endpoint into a blank response.

`group_batch_order_number` follows the spec's `"{batch#}_{order#}"` format.
"""

from __future__ import annotations

from typing import Any

GROUP_BATCH = 3
GROUP_ORDER = 3

TEAM_NAME = "UniPilot"

STUDENTS: tuple[dict[str, str], ...] = (
    {"name": "Tymor Ibrahim", "email": "tymoribrahim@campus.technion.ac.il"},
    {"name": "Kinan Ibrahim", "email": "kinani@campus.technion.ac.il"},
)


def team_info() -> dict[str, Any]:
    """The exact response body the spec requires."""
    return {
        "group_batch_order_number": f"{GROUP_BATCH}_{GROUP_ORDER}",
        "team_name": TEAM_NAME,
        "students": [dict(student) for student in STUDENTS],
    }


__all__ = ["GROUP_BATCH", "GROUP_ORDER", "STUDENTS", "TEAM_NAME", "team_info"]
