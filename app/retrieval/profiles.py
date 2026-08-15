"""Retrieval profile configuration -- ported verbatim from UniPilot.

These weights and boosts are TUNED, not chosen: `hybridVectorWeight: 0.6` /
`hybridKeywordWeight: 0.4` and the `rerankBoosts` table came out of real
evaluation runs against this corpus. They are carried across with the numbers
intact, because a retriever with re-invented weights is a different retriever
wearing the same name.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROFILE_CONFIG_PATH = Path(__file__).with_name("profile_config.json")


class RerankBoosts(BaseModel):
    exactCourseNumberBoost: float = 8.0
    exactSemesterBoost: float = 6.0
    degreeProgramMatchBoost: float = 4.0
    trackMatchBoost: float = 3.0
    catalogYearMatchBoost: float = 2.0
    wrongSemesterPenalty: float = -10.0
    wrongTrackPenalty: float = -6.0
    wrongCatalogYearPenalty: float = -4.0
    linkRelevanceBoost: float = 1.5
    sourcePriorityBoost: float = 1.0


class RetrievalProfile(BaseModel):
    profileName: str
    exactLookupFirst: bool = True
    structuredPlannerFirst: bool = False
    sources: list[str] = Field(default_factory=list)
    vectorTopK: int = 30
    bm25TopK: int = 30
    hybridVectorWeight: float = 0.5
    hybridKeywordWeight: float = 0.5
    rerankCandidateLimit: int = 50
    finalTopN: int = 8
    wikiChunksFinal: int = 5
    linkExpansionDepth: int = 0
    maxLinkedChunks: int = 0
    maxContextTokens: int = 6000
    maxRetrievalAttempts: int = 2
    latencyBudgetMs: int = 2000
    minRetrievalConfidence: float = 0.25

    @property
    def hybrid_keyword_weight_normalized(self) -> float:
        total = self.hybridVectorWeight + self.hybridKeywordWeight
        if total <= 0:
            return 0.5
        return self.hybridKeywordWeight / total

    @property
    def hybrid_vector_weight_normalized(self) -> float:
        total = self.hybridVectorWeight + self.hybridKeywordWeight
        if total <= 0:
            return 0.5
        return self.hybridVectorWeight / total


class ProfileConfig(BaseModel):
    version: str = "1.0.0"
    lockedAt: str | None = None
    notes: str | None = None
    rerankBoosts: RerankBoosts = Field(default_factory=RerankBoosts)
    intentMapping: dict[str, list[str]] = Field(default_factory=dict)
    profiles: dict[str, RetrievalProfile] = Field(default_factory=dict)


def _load_profile_config_raw() -> dict[str, Any]:
    if not _PROFILE_CONFIG_PATH.is_file():
        return {"profiles": {}, "intentMapping": {}, "rerankBoosts": {}}
    return json.loads(_PROFILE_CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_profile_config() -> ProfileConfig:
    raw = _load_profile_config_raw()
    profiles: dict[str, RetrievalProfile] = {}
    for name, payload in (raw.get("profiles") or {}).items():
        data = dict(payload)
        data.pop("profileName", None)
        profiles[name] = RetrievalProfile(profileName=name, **data)
    return ProfileConfig(
        version=str(raw.get("version") or "1.0.0"),
        lockedAt=raw.get("lockedAt"),
        notes=raw.get("notes"),
        rerankBoosts=RerankBoosts.model_validate(raw.get("rerankBoosts") or {}),
        intentMapping=dict(raw.get("intentMapping") or {}),
        profiles=profiles,
    )


def reset_profile_config_cache() -> None:
    load_profile_config.cache_clear()


def get_profile(name: str) -> RetrievalProfile:
    config = load_profile_config()
    profile = config.profiles.get(name)
    if profile is None:
        return config.profiles.get(
            "fallback_academic_search",
            RetrievalProfile(profileName="fallback_academic_search"),
        )
    return profile


def get_rerank_boosts() -> RerankBoosts:
    return load_profile_config().rerankBoosts


def profile_allows_wiki(profile: RetrievalProfile) -> bool:
    return any(source in profile.sources for source in ("obsidian_wiki", "academic_graph"))


def profile_allows_structured_catalog(profile: RetrievalProfile) -> bool:
    return "structured_catalog" in profile.sources or "structured_requirements" in profile.sources


def profile_allows_structured_offerings(profile: RetrievalProfile) -> bool:
    return (
        "structured_offerings" in profile.sources
        or "offering_vector_index" in profile.sources
    )


def estimate_context_tokens(snippets: list[dict[str, Any]]) -> int:
    """Rough token estimate (~4 chars per token)."""
    total_chars = sum(len(str(item.get("content") or "")) for item in snippets)
    return max(1, total_chars // 4)
