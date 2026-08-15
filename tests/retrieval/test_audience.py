"""The corpus must not answer an undergraduate out of the graduate rulebook.

Measured, not hypothesised: for the query "English language requirement to
graduate", `concepts/regulations-graduate.md` scores 0.591 and the correct
`concepts/regulations-undergraduate.md` scores 0.581, so the wrong document wins
on merit. A live BSc student was told "All graduate students must demonstrate
English proficiency" -- faithfully quoted from a real citation, and wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.corpus import _addresses


@dataclass
class _Chunk:
    tags: tuple[str, ...] = field(default_factory=tuple)


class TestAudienceFilter:
    def test_the_graduate_rulebook_is_hidden_from_an_undergraduate(self) -> None:
        graduate = _Chunk(tags=("regulations", "graduate", "msc", "phd", "technion"))
        assert not _addresses(graduate, "undergraduate")

    def test_the_undergraduate_rulebook_reaches_an_undergraduate(self) -> None:
        undergraduate = _Chunk(tags=("regulations", "undergraduate", "bsc", "technion"))
        assert _addresses(undergraduate, "undergraduate")

    def test_the_undergraduate_rulebook_is_hidden_from_a_graduate(self) -> None:
        undergraduate = _Chunk(tags=("regulations", "undergraduate", "bsc", "technion"))
        assert not _addresses(undergraduate, "graduate")

    def test_a_chunk_with_no_level_addresses_everyone(self) -> None:
        """1,988 of 4,895 chunks declare no level -- course pages, most faculty
        pages. Dropping them would empty the corpus for every ordinary question."""
        untagged = _Chunk(tags=("course", "technion"))
        assert _addresses(untagged, "undergraduate")
        assert _addresses(untagged, "graduate")

    def test_a_chunk_claiming_both_levels_is_kept(self) -> None:
        """A page that addresses both is not the wrong rulebook for either."""
        both = _Chunk(tags=("regulations", "undergraduate", "graduate"))
        assert _addresses(both, "undergraduate")
        assert _addresses(both, "graduate")

    def test_an_unknown_audience_filters_nothing(self) -> None:
        """An unreadable profile must leave the corpus whole rather than
        narrowing it to nothing -- no answer is worse than an unscoped one."""
        graduate = _Chunk(tags=("graduate", "msc"))
        assert _addresses(graduate, None)
        assert _addresses(graduate, "postdoctoral")


class TestAudienceResolution:
    """`programType` is the stored vocabulary; the corpus speaks in levels."""

    def test_bsc_resolves_to_undergraduate(self) -> None:
        from app.agent_core.facts.service import _audience_of_profile

        assert _audience_of_profile({"programType": "BSc"}) == "undergraduate"

    def test_msc_resolves_to_graduate(self) -> None:
        from app.agent_core.facts.service import _audience_of_profile

        assert _audience_of_profile({"programType": "MSc"}) == "graduate"

    def test_an_unrecognised_type_leaves_the_corpus_unfiltered(self) -> None:
        """Narrowing the corpus to nothing is worse than not narrowing it."""
        from app.agent_core.facts.service import _audience_of_profile

        assert _audience_of_profile({"programType": "Habilitation"}) == "undergraduate"
        assert _audience_of_profile({"programType": None}) is None
        assert _audience_of_profile(None) is None


class TestTheStudentMustExist:
    """An unknown student produced a confident zero.

    Every `find` returns an empty collection, `sum` over empty is 0, and the run
    answered "You have completed 0 credits" -- grounded in a real fetch, and
    about nobody. Measured in production with student_id=nonexistent-student:
    status ok, six steps, that answer.
    """

    async def test_a_known_student_is_returned(self) -> None:
        from app.agent_core.facts.service import _profile_of

        class _Db:
            async def fetch(self, *_: object) -> list:
                return [{"userId": "u1", "programType": "BSc"}]

        profile = await _profile_of(_Db(), "u1")
        assert profile["programType"] == "BSc"

    async def test_an_unknown_student_is_none(self) -> None:
        from app.agent_core.facts.service import _profile_of

        class _Db:
            async def fetch(self, *_: object) -> list:
                return []

        assert await _profile_of(_Db(), "ghost") is None

    async def test_a_failed_read_is_not_reported_as_no_such_student(self) -> None:
        """A database outage reported as "no such student" is the same
        confident-wrong-answer failure wearing a different hat, so the error
        propagates instead of being flattened into None."""
        import pytest

        from app.agent_core.facts.service import _profile_of

        class _Db:
            async def fetch(self, *_: object) -> list:
                raise RuntimeError("connection reset")

        with pytest.raises(RuntimeError):
            await _profile_of(_Db(), "u1")
