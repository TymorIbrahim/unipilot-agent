"""An empty fetch about a named thing is absence of knowledge, not a zero.

Asked "am I eligible to take 00999999" -- a course code that is in no catalog
row -- the agent answered:

    "yes -- this course has 0 prerequisite groups, and you meet 0 of them."

Every step was sound. `find` on `prerequisite_edges` returned nothing, so the
obligation count was 0, the met count was 0, and 0 >= 0 is true. A course that
does not exist got a confident yes.

Comparison operators are what made it reachable: before they existed the model
could not evaluate `met >= required` at all, so the vacuous path had no way to
produce a claim. Adding a capability opened a hole that had never been reachable.

The rule is in the system prompt because it is a REASONING rule, not a shape one
-- no post-condition can tell a genuinely prerequisite-free course from one that
is absent, since both legitimately have zero edges. Only checking the catalog
first distinguishes them. These tests pin the rule against accidental deletion;
the measured behaviour is in the commit that added it (0/3 false claims after,
against 1/2 before).
"""

from __future__ import annotations

import re

from app.agent_core.facts.adapter import SYSTEM_PROMPT

# The prompt is wrapped prose, so a phrase can straddle a line break. Matching
# against a whitespace-collapsed copy keeps these tests about WHAT it says
# rather than where the lines happen to end.
PROMPT = re.sub(r"\s+", " ", SYSTEM_PROMPT)


class TestTheRuleIsTaught:
    def test_the_prompt_requires_confirming_a_named_course_exists(self) -> None:
        assert "CONFIRM IT EXISTS FIRST" in PROMPT

    def test_the_prompt_names_the_failure_it_prevents(self) -> None:
        """The worked example is the point: an abstract rule about empty results
        did not stop this, a concrete "0 >= 0 is true" walkthrough did."""
        assert "00999999" in PROMPT
        assert "0 >= 0" in PROMPT

    def test_the_rule_generalises_beyond_courses(self) -> None:
        """A student, a term and a track fail the same way."""
        assert "ABSENCE OF KNOWLEDGE" in PROMPT
        assert "never a finding of zero" in PROMPT

    def test_an_already_passed_course_is_surfaced(self) -> None:
        """Eligibility for a course already passed is technically yes and
        practically useless -- re-taking it earns no further credit."""
        assert "already on the transcript" in PROMPT
        assert "no further credit" in PROMPT
