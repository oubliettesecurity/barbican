"""Coordination correlator (public/defensive; the Phase-2 novelty).

Discovers coordinated clusters in a set of unlabeled posts via pairwise
coordination signals -> thresholded graph -> union-find clustering ->
per-cluster coordination score. Does NOT import oubliette_shield.spectre.
"""

from dataclasses import dataclass
from datetime import datetime

from .artifact import extract_features
from .embed import EmbeddingFn, cosine
from .types import Post


@dataclass(frozen=True)
class CorrelatorConfig:
    edge_threshold: float = 0.6
    coord_threshold: float = 0.6
    time_window_seconds: float = 3600.0
    narrative_weight: float = 0.5
    temporal_weight: float = 0.25
    persona_weight: float = 0.25
    ngram_n: int = 3


def _char_ngrams(text: str, n: int) -> set[str]:
    t = " ".join(text.lower().split())
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def ngram_jaccard(a: str, b: str, n: int) -> float:
    sa, sb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def temporal_proximity(ts_a: str, ts_b: str, window: float) -> float:
    ta = datetime.fromisoformat(ts_a)
    tb = datetime.fromisoformat(ts_b)
    delta = abs((ta - tb).total_seconds())
    if delta >= window:
        return 0.0
    return 1.0 - (delta / window)


def _feature_vec(text: str) -> list[float]:
    feats = extract_features(text)
    return [feats[k] for k in sorted(feats)]


def persona_sim(a: str, b: str) -> float:
    return cosine(_feature_vec(a), _feature_vec(b))


def pairwise_signal(
    pi: Post, pj: Post, ei: list[float], ej: list[float], cfg: CorrelatorConfig
) -> float:
    # Real embeddings can yield negative cosine; clamp the narrative similarity
    # contribution to be non-negative so anti-correlated vectors cannot subtract
    # from the (non-negative) ngram/temporal/persona coordination evidence.
    narrative = 0.5 * max(0.0, cosine(ei, ej)) + 0.5 * ngram_jaccard(pi.text, pj.text, cfg.ngram_n)
    temporal = temporal_proximity(pi.timestamp, pj.timestamp, cfg.time_window_seconds)
    persona = persona_sim(pi.text, pj.text)
    total_w = cfg.narrative_weight + cfg.temporal_weight + cfg.persona_weight
    return (
        cfg.narrative_weight * narrative
        + cfg.temporal_weight * temporal
        + cfg.persona_weight * persona
    ) / total_w


@dataclass(frozen=True)
class Cluster:
    post_ids: tuple[str, ...]
    score: float
    flagged: bool


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    comps = [sorted(members) for members in groups.values()]
    comps.sort(key=lambda c: c[0])
    return comps


def discover(posts: list[Post], embed: EmbeddingFn, cfg: CorrelatorConfig) -> list[Cluster]:
    n = len(posts)
    embeddings = embed([p.text for p in posts])
    # pairwise signals + edges
    signal: dict[tuple[int, int], float] = {}
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s = pairwise_signal(posts[i], posts[j], embeddings[i], embeddings[j], cfg)
            signal[(i, j)] = s
            if s >= cfg.edge_threshold:
                edges.append((i, j))

    clusters: list[Cluster] = []
    for comp in connected_components(n, edges):
        if len(comp) < 2:
            continue
        pairs = [signal[(a, b)] for k, a in enumerate(comp) for b in comp[k + 1 :]]
        score = sum(pairs) / len(pairs) if pairs else 0.0
        flagged = score >= cfg.coord_threshold
        post_ids = tuple(sorted(posts[i].post_id for i in comp))
        clusters.append(Cluster(post_ids=post_ids, score=score, flagged=flagged))

    clusters.sort(key=lambda c: c.post_ids[0])
    return clusters
