import json

from barbican.artifact import BaselineDetector
from barbican.correlator import Cluster, CorrelatorConfig, pairwise_signal
from barbican.experiment import (
    artifact_metrics,
    campaign_metrics_discovery,
    campaign_metrics_naive,
    prf,
    run_baseline,
    run_layered,
)
from barbican.types import Dataset, Post


def test_prf_handles_zero_denominators():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    r = prf(tp=2, fp=2, fn=0)
    assert r["precision"] == 0.5 and r["recall"] == 1.0


def _syn(pid, text, cid, backend="modelA"):
    return Post(
        post_id=pid,
        text=text,
        author_id="PER-1",
        campaign_id=cid,
        timestamp="2020-01-01T00:00:00+00:00",
        backend_model=backend,
        label="synthetic",
    )


def _auth(pid, text):
    return Post(
        post_id=pid,
        text=text,
        author_id="acct",
        campaign_id=None,
        timestamp="2020-01-01T00:00:00+00:00",
        backend_model=None,
        label="authentic",
    )


def _train():
    syn = [_syn(f"s{i}", "VOTE NOW #a #b rigged share http://x.io", "T") for i in range(6)]
    auth = [
        _auth(f"a{i}", t)
        for i, t in enumerate(
            ["lol monday", "my cat is cute", "taco place?", "great game!!", "ttyl", "nice weather"]
        )
    ]
    return Dataset(posts=syn + auth)


def test_single_surface_ceiling_on_coordination_only_campaign():
    # A coordinated campaign whose members individually look authentic (personal, messy),
    # only suspicious in aggregate (identical narrative). Baseline should MISS it at
    # the campaign level -> demonstrates the single-surface ceiling Phase 2 will lift.
    train = _train()
    det = BaselineDetector()
    det.fit(train.posts)
    stealthy = Dataset(
        posts=[
            _syn("c1", "had coffee, thinking the mayor really let us down today", "STEALTH"),
            _syn("c2", "grabbed lunch, honestly the mayor let us down today", "STEALTH"),
            _syn("c3", "walking home, the mayor really let us down today tbh", "STEALTH"),
        ]
    )
    m = campaign_metrics_naive(det, stealthy, threshold=0.5, vote=0.5)
    assert m["recall"] < 1.0  # naive per-artifact aggregation misses the coordinated campaign


def test_run_baseline_writes_artifacts_and_skips_none(tmp_path):
    train = _train()
    evalsets = {
        "held_in": Dataset(
            posts=[_syn("h1", "VOTE NOW #a #b rigged http://x.io", "H"), _auth("h2", "lol ok")]
        ),
        "real_world": None,
    }
    res = run_baseline(train, evalsets, seed=3, out_dir=str(tmp_path), spec_hash="abc123")
    assert res["real_world"] == {"status": "SKIPPED"}
    assert "artifact" in res["held_in"] and "campaign" in res["held_in"]
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["seed"] == 3 and saved["spec_hash"] == "abc123"
    assert (tmp_path / "RESULTS.md").exists()


def _p(pid, text, author, ts, camp, label="synthetic"):
    return Post(
        post_id=pid,
        text=text,
        author_id=author,
        campaign_id=camp,
        timestamp=ts,
        backend_model=None,
        label=label,
    )


def test_pairwise_signal_clamps_negative_cosine():
    # Anti-correlated embeddings (cosine = -1) must not drag the narrative term below
    # what orthogonal embeddings (cosine = 0) produce: both clamp to a 0 embedding term.
    cfg = CorrelatorConfig()
    a = _p("P0", "identical coordinated text here", "u0", "2026-01-01T00:00:00", "K")
    b = _p("P1", "identical coordinated text here", "u1", "2026-01-01T00:00:00", "K")
    opposite = pairwise_signal(a, b, [1.0, 0.0], [-1.0, 0.0], cfg)
    orthogonal = pairwise_signal(a, b, [1.0, 0.0], [0.0, 1.0], cfg)
    assert opposite == orthogonal


def test_campaign_metrics_discovery_recovers_campaign():
    ds = Dataset(
        posts=[
            _p("C0", "x", "u0", "2026-01-01T00:00:00", "CAMP-1"),
            _p("C1", "x", "u1", "2026-01-01T00:01:00", "CAMP-1"),
        ]
    )
    clusters = [Cluster(post_ids=("C0", "C1"), score=0.9, flagged=True)]
    m = campaign_metrics_discovery(clusters, ds, overlap=0.5)
    assert m["recall"] == 1.0
    assert m["n_campaigns"] == 1.0
    assert m["false_clusters"] == 0.0


# Coordinated stealthy campaign: each post individually reads as authentic personal
# chatter (so the per-artifact BaselineDetector, trained to spot spammy synthetics,
# scores every member < 0.5 and MISSES the campaign), yet the members are
# near-duplicate + time-synchronized so the correlator recovers them as one cluster.
_COORD_TEXTS = [
    "had coffee, thinking the mayor really let us down today",
    "grabbed lunch, honestly the mayor let us down today",
    "walking home, the mayor really let us down today tbh",
]
# Genuinely uncoordinated authentic posts: distinct topics, spread-out timestamps,
# and orthogonal stub embeddings -> must NOT form a flagged cluster (false_clusters==0).
_AUTHENTIC_EVAL = [
    (
        "A0",
        "finally fixed my bike chain after three tries",
        "v0",
        "2026-01-15T12:00:00",
        "bike chain",
        [0.0, 1.0, 0.0, 0.0],
    ),
    (
        "A1",
        "anyone know a good ramen spot downtown?",
        "v1",
        "2026-02-20T08:30:00",
        "ramen",
        [0.0, 0.0, 1.0, 0.0],
    ),
    (
        "A2",
        "the sunrise over the harbor was unreal this morning",
        "v2",
        "2026-03-25T18:45:00",
        "sunrise",
        [0.0, 0.0, 0.0, 1.0],
    ),
]


def _thesis_fixture():
    """train (obvious spam vs benign), stealthy coordinated evalset, and stub embed."""
    train = _train()  # 6 spammy synthetics + 6 benign authentics
    coord = [
        _p(f"C{i}", _COORD_TEXTS[i], f"u{i}", f"2026-03-01T09:0{i}:00", "CAMP-1") for i in range(3)
    ]
    authentic = [
        _p(pid, text, author, ts, None, label="authentic")
        for pid, text, author, ts, _key, _vec in _AUTHENTIC_EVAL
    ]
    evalset = Dataset(posts=coord + authentic)
    vecs = {key: vec for _pid, _t, _a, _ts, key, vec in _AUTHENTIC_EVAL}

    def fake_embed(texts):
        out = []
        for t in texts:
            if "mayor" in t:  # every coordinated post shares this near-identical vector
                out.append([1.0, 0.0, 0.0, 0.0])
            else:
                matched = next((v for k, v in vecs.items() if k in t), [0.0, 0.0, 0.0, 0.0])
                out.append(list(matched))
        return out

    return train, evalset, fake_embed


def test_layered_beats_baseline_when_only_coordination_separates(tmp_path):
    train, evalset, fake_embed = _thesis_fixture()
    cfg = CorrelatorConfig()
    r = run_layered(
        train,
        evalset,
        fake_embed,
        cfg,
        seed=1337,
        out_dir=str(tmp_path),
        spec_hash="deadbeef",
    )
    # The per-artifact baseline GENUINELY MISSES the coordinated campaign: this test
    # would fail (not silently pass) if discover() returned no flagged cluster.
    assert r["baseline_campaign_recall"] < 1.0
    assert r["layered_campaign_recall"] > r["baseline_campaign_recall"]
    assert r["lift"] > 0.0  # strict: coordination provides real lift
    assert r["layered_campaign_recall"] == 1.0
    # Honesty signal: the uncoordinated authentic posts must not be wrongly flagged.
    assert r["false_clusters"] == 0.0


def test_run_layered_persists_results_with_provenance(tmp_path):
    train, evalset, fake_embed = _thesis_fixture()
    cfg = CorrelatorConfig(edge_threshold=0.55, ngram_n=4)
    run_layered(
        train,
        evalset,
        fake_embed,
        cfg,
        seed=7,
        out_dir=str(tmp_path),
        spec_hash="cafe1234",
    )
    assert (tmp_path / "RESULTS.md").exists()
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["seed"] == 7.0
    assert saved["spec_hash"] == "cafe1234"
    assert saved["backends"] == ["modelA"]  # from train.backends()
    cc = saved["correlator_config"]
    assert cc["edge_threshold"] == 0.55
    assert cc["coord_threshold"] == cfg.coord_threshold
    assert cc["ngram_n"] == 4
    assert cc["narrative_weight"] == cfg.narrative_weight
    assert cc["temporal_weight"] == cfg.temporal_weight
    assert cc["persona_weight"] == cfg.persona_weight
    assert cc["time_window_seconds"] == cfg.time_window_seconds
    assert "baseline_campaign_recall" in saved and "lift" in saved
