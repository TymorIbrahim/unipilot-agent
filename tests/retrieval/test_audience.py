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

    async def test_bsc_resolves_to_undergraduate(self) -> None:
        from app.agent_core.facts.service import _audience_of

        class _Db:
            async def fetchval(self, *_: object) -> str:
                return "BSc"

        assert await _audience_of(_Db(), "u1") == "undergraduate"

    async def test_msc_resolves_to_graduate(self) -> None:
        from app.agent_core.facts.service import _audience_of

        class _Db:
            async def fetchval(self, *_: object) -> str:
                return "MSc"

        assert await _audience_of(_Db(), "u1") == "graduate"

    async def test_a_missing_profile_leaves_the_corpus_unfiltered(self) -> None:
        from app.agent_core.facts.service import _audience_of

        class _Db:
            async def fetchval(self, *_: object) -> None:
                return None

        assert await _audience_of(_Db(), "u1") is None

    async def test_an_unreadable_profile_does_not_end_the_run(self) -> None:
        from app.agent_core.facts.service import _audience_of

        class _Db:
            async def fetchval(self, *_: object) -> str:
                raise RuntimeError("connection reset")

        assert await _audience_of(_Db(), "u1") is None
