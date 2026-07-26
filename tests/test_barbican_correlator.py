import pytest

from barbican.types import Post
from barbican.correlator import (
    CorrelatorConfig, ngram_jaccard, temporal_proximity, persona_sim, pairwise_signal,
)


def _post(pid, text, author, ts, camp=None):
    return Post(post_id=pid, text=text, author_id=author, campaign_id=camp,
                timestamp=ts, backend_model=None, label="synthetic")


def test_ngram_jaccard_identical_is_one():
    assert ngram_jaccard("the quick brown fox", "the quick brown fox", 3) == 1.0


def test_ngram_jaccard_disjoint_is_zero():
    assert ngram_jaccard("aaaa bbbb", "cccc dddd", 3) == 0.0


def test_temporal_proximity_same_time_is_one():
    assert temporal_proximity("2026-01-01T00:00:00", "2026-01-01T00:00:00", 3600.0) == 1.0


def test_temporal_proximity_beyond_window_is_zero():
    assert temporal_proximity("2026-01-01T00:00:00", "2026-01-01T02:00:00", 3600.0) == 0.0


def test_pairwise_signal_high_for_coordinated_pair():
    cfg = CorrelatorConfig()
    a = _post("A", "buy now this token moons soon", "u1", "2026-01-01T00:00:00")
    b = _post("B", "buy now this token moons soon", "u2", "2026-01-01T00:01:00")
    # identical embeddings, near-identical text, close time -> high signal
    s = pairwise_signal(a, b, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], cfg)
    assert s > 0.8


def test_narrative_embed_weight_is_configurable():
    # Default (0.5/0.5): embed cosine and ngram jaccard contribute equally.
    # A non-default weight must shift the narrative term toward whichever
    # sub-signal it favors, instead of the 0.5/0.5 split being hard-coded.
    a = _post("A", "completely different phrasing entirely", "u1", "2026-01-01T00:00:00")
    b = _post("B", "totally unrelated wording altogether", "u2", "2026-01-01T00:00:00")
    ei, ej = [1.0, 0.0], [1.0, 0.0]  # cosine = 1.0 (perfect embed match)
    # ngram_jaccard(a.text, b.text, 3) is low (near-disjoint character trigrams)
    low_ngram = ngram_jaccard(a.text, b.text, 3)
    assert low_ngram < 0.3

    cfg_default = CorrelatorConfig()
    cfg_embed_heavy = CorrelatorConfig(narrative_embed_weight=1.0)
    cfg_ngram_heavy = CorrelatorConfig(narrative_embed_weight=0.0)

    s_default = pairwise_signal(a, b, ei, ej, cfg_default)
    s_embed_heavy = pairwise_signal(a, b, ei, ej, cfg_embed_heavy)
    s_ngram_heavy = pairwise_signal(a, b, ei, ej, cfg_ngram_heavy)

    # Embed-heavy (weight=1.0) ignores the low ngram score -> higher signal.
    assert s_embed_heavy > s_default
    # Ngram-heavy (weight=0.0) ignores the perfect cosine -> lower signal.
    assert s_ngram_heavy < s_default


def test_narrative_embed_weight_default_preserves_prior_behavior():
    # Default must reproduce the old hard-coded 0.5*cosine + 0.5*ngram split.
    cfg = CorrelatorConfig()
    assert cfg.narrative_embed_weight == 0.5
    a = _post("A", "buy now this token moons soon", "u1", "2026-01-01T00:00:00")
    b = _post("B", "buy now this token moons soon", "u2", "2026-01-01T00:01:00")
    s = pairwise_signal(a, b, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], cfg)
    assert s > 0.8


def test_pairwise_signal_low_for_unrelated_pair():
    cfg = CorrelatorConfig()
    a = _post("A", "the weather is nice in spring", "u1", "2026-01-01T00:00:00")
    b = _post("B", "quarterly earnings exceeded forecasts", "u2", "2026-06-01T00:00:00")
    s = pairwise_signal(a, b, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], cfg)
    assert s < 0.3


from barbican.correlator import connected_components, discover, Cluster


def test_connected_components_basic():
    # 0-1 linked, 2-3 linked, 4 alone
    comps = connected_components(5, [(0, 1), (2, 3)])
    assert comps == [[0, 1], [2, 3], [4]]


def test_connected_components_transitive_chain():
    # 0-1 and 1-2 linked, but 0-2 has NO direct edge. Union-find must still
    # merge all three into one component via transitivity through node 1 -
    # a naive clique-requiring implementation would wrongly split this into
    # {0,1} and {1,2} (or fail to merge 0 with 2 at all).
    comps = connected_components(3, [(0, 1), (1, 2)])
    assert comps == [[0, 1, 2]]


def test_discover_flags_planted_cluster_not_noise():
    cfg = CorrelatorConfig()
    # 3 coordinated posts: identical text, same-ish time; embeddings identical.
    coordinated = [
        _post(f"C{i}", "vote yes on prop today it is urgent", f"u{i}",
              f"2026-01-01T00:0{i}:00", camp="CAMP-1")
        for i in range(3)
    ]
    # 3 noise posts: distinct topics, spread out in time; orthogonal embeddings.
    noise_texts = [
        "i had pasta for dinner last night",
        "the museum exhibit closes in march",
        "my train was delayed twenty minutes",
    ]
    noise = [
        _post(f"N{i}", noise_texts[i], f"v{i}", f"2026-03-0{i + 1}T12:00:00")
        for i in range(3)
    ]
    posts = coordinated + noise

    def fake_embed(texts):
        vecs = []
        for t in texts:
            if "vote yes on prop" in t:
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                # distinct orthogonal-ish vectors per noise post
                idx = noise_texts.index(t) if t in noise_texts else 0
                v = [0.0, 0.0, 0.0, 0.0]
                v[idx + 1] = 1.0
                vecs.append(v)
        return vecs

    clusters = discover(posts, fake_embed, cfg)
    flagged = [c for c in clusters if c.flagged]
    assert len(flagged) == 1
    assert set(flagged[0].post_ids) == {"C0", "C1", "C2"}


def test_discover_no_free_lunch_uncoordinated_yields_no_flag():
    cfg = CorrelatorConfig()
    texts = [
        "i had pasta for dinner last night",
        "the museum exhibit closes in march",
        "my train was delayed twenty minutes",
        "the quarterly report is due friday",
    ]
    posts = [
        _post(f"N{i}", texts[i], f"v{i}", f"2026-0{i + 1}-01T12:00:00")
        for i in range(4)
    ]

    def fake_embed(texts_):
        return [[1.0 if j == i else 0.0 for j in range(4)] for i in range(len(texts_))]

    clusters = discover(posts, fake_embed, cfg)
    assert [c for c in clusters if c.flagged] == []


def test_discover_is_deterministic():
    cfg = CorrelatorConfig()
    posts = [
        _post(f"C{i}", "same coordinated message here now", f"u{i}",
              f"2026-01-01T00:0{i}:00", camp="CAMP-1")
        for i in range(3)
    ]

    def fake_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    r1 = discover(posts, fake_embed, cfg)
    r2 = discover(posts, fake_embed, cfg)
    assert r1 == r2


def test_discover_transitive_non_clique_cluster():
    # A~B and B~C are strong edges (>= edge_threshold), but A~C is weak
    # (< edge_threshold) - no direct edge between A and C. A clique-requiring
    # implementation would either split this into two separate pairs or drop
    # the weakly-linked post; union-find must merge all three into ONE
    # cluster via transitivity through B, and the score must be the mean of
    # all THREE intra-cluster pairs (including the weak A~C pair), not just
    # the two strong edges.
    cfg = CorrelatorConfig()
    text = "same coordinated narrative text here"
    a = _post("A", text, "u1", "2026-01-01T00:00:00")
    b = _post("B", text, "u2", "2026-01-01T00:30:00")
    c = _post("C", text, "u3", "2026-01-01T01:00:00")
    posts = [a, b, c]

    # Identical text -> ngram_jaccard == persona_sim == 1.0 for every pair,
    # so narrative/persona contribute equally everywhere. The discriminator
    # is embedding cosine (A and C orthogonal, B similar to both) combined
    # with timestamps (A and C are exactly time_window_seconds apart, so
    # temporal_proximity(A, C) == 0.0, while A-B and B-C are 30 minutes
    # apart each, well inside the window).
    embeds = {"A": [1.0, 0.0], "B": [1.0, 1.0], "C": [0.0, 1.0]}

    def fake_embed(texts):
        assert texts == [text, text, text]
        return [embeds["A"], embeds["B"], embeds["C"]]

    # Verify the intended signal shape directly via pairwise_signal before
    # trusting discover() to reproduce it.
    s_ab = pairwise_signal(a, b, embeds["A"], embeds["B"], cfg)
    s_bc = pairwise_signal(b, c, embeds["B"], embeds["C"], cfg)
    s_ac = pairwise_signal(a, c, embeds["A"], embeds["C"], cfg)
    assert s_ab >= cfg.edge_threshold
    assert s_bc >= cfg.edge_threshold
    assert s_ac < cfg.edge_threshold

    clusters = discover(posts, fake_embed, cfg)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert set(cluster.post_ids) == {"A", "B", "C"}

    expected_score = (s_ab + s_bc + s_ac) / 3
    assert cluster.score == pytest.approx(expected_score)
