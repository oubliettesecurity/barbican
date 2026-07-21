"""Shared record types for the BARBICAN detection pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Post:
    """A single normalized social post (synthetic or authentic)."""

    post_id: str
    text: str
    author_id: str  # persona_id for synthetic; account id for authentic
    campaign_id: str | None  # shared id for a coordinated set; None if standalone
    timestamp: str  # ISO 8601
    backend_model: str | None  # generator backend for synthetic; None for authentic
    label: str  # "synthetic" | "authentic"


@dataclass
class Dataset:
    """A collection of posts with grouping helpers."""

    posts: list[Post]

    def by_label(self, label: str) -> list[Post]:
        return [p for p in self.posts if p.label == label]

    def campaigns(self) -> dict[str, list[Post]]:
        groups: dict[str, list[Post]] = {}
        for p in self.posts:
            if p.campaign_id is not None:
                groups.setdefault(p.campaign_id, []).append(p)
        return groups

    def backends(self) -> set[str]:
        return {p.backend_model for p in self.posts if p.backend_model is not None}
