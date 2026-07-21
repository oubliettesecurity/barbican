import json
from barbican.realworld import load_realworld
from barbican.types import Dataset


def _write_jsonl(tmp_path, rows):
    p = tmp_path / "rw.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(p)


def test_absent_path_returns_none():
    assert load_realworld(None) is None
    assert load_realworld("does/not/exist.jsonl") is None


def test_maps_labels_and_campaign(tmp_path):
    rows = [
        {"text": "vote now", "author_id": "a1", "timestamp": "2026-01-01T00:00:00",
         "label": "io", "campaign_id": "C1"},
        {"text": "nice weather", "author_id": "b1", "timestamp": "2026-01-02T00:00:00",
         "label": "control", "campaign_id": None},
    ]
    ds = load_realworld(_write_jsonl(tmp_path, rows))
    assert isinstance(ds, Dataset)
    io = ds.by_label("synthetic")
    control = ds.by_label("authentic")
    assert len(io) == 1 and len(control) == 1
    assert io[0].campaign_id == "C1"
    assert io[0].backend_model is None
    assert control[0].campaign_id is None
    assert io[0].post_id == "RW-000000"


def test_control_campaign_forced_none(tmp_path):
    rows = [{"text": "x", "author_id": "b", "timestamp": "2026-01-01T00:00:00",
             "label": "control", "campaign_id": "SHOULD_BE_DROPPED"}]
    ds = load_realworld(_write_jsonl(tmp_path, rows))
    assert ds.by_label("authentic")[0].campaign_id is None


def test_malformed_lines_skipped_not_silent(tmp_path, caplog):
    rows_text = "\n".join([
        json.dumps({"text": "ok", "author_id": "a", "timestamp": "2026-01-01T00:00:00",
                    "label": "io", "campaign_id": "C1"}),
        "{not json",
        json.dumps({"text": "bad label", "author_id": "a", "timestamp": "2026-01-01T00:00:00",
                    "label": "??", "campaign_id": None}),
    ])
    p = tmp_path / "rw.jsonl"
    p.write_text(rows_text, encoding="utf-8")
    import logging
    with caplog.at_level(logging.WARNING):
        ds = load_realworld(str(p))
    assert len(ds.posts) == 1  # only the valid row
    assert any("dropped" in r.message.lower() for r in caplog.records)


def test_deterministic(tmp_path):
    rows = [{"text": f"t{i}", "author_id": f"a{i}", "timestamp": "2026-01-01T00:00:00",
             "label": "io", "campaign_id": "C1"} for i in range(3)]
    path = _write_jsonl(tmp_path, rows)
    assert load_realworld(path).posts == load_realworld(path).posts


from barbican.types import Post, Dataset
from barbican.correlator import CorrelatorConfig
from barbican.experiment import run_realworld_split


def _p(pid, text, author, ts, camp, label="synthetic"):
    return Post(post_id=pid, text=text, author_id=author, campaign_id=camp,
                timestamp=ts, backend_model=None, label=label)


def test_realworld_split_skipped_when_absent(tmp_path):
    train = Dataset(posts=[_p("C0", "x", "u0", "2026-01-01T00:00:00", "C1")])
    out = run_realworld_split(train, None, lambda ts: [[1.0] for _ in ts],
                              CorrelatorConfig(), seed=1, out_dir=str(tmp_path),
                              spec_hash="h")
    assert out["status"] == "SKIPPED"


def test_realworld_split_runs_when_present(tmp_path):
    coord = [_p(f"C{i}", "same msg here", f"u{i}", f"2026-01-01T00:0{i}:00", "C1")
             for i in range(3)]
    authentic = [_p(f"A{i}", f"benign {i}", f"v{i}", f"2026-0{i+1}-01T00:00:00", None,
                    label="authentic") for i in range(3)]
    train = Dataset(posts=coord + authentic)
    rw = Dataset(posts=coord + authentic)
    emb = lambda ts: [[1.0, 0.0] if "same msg" in t else [0.0, 1.0] for t in ts]
    out = run_realworld_split(train, rw, emb, CorrelatorConfig(), seed=1,
                              out_dir=str(tmp_path), spec_hash="h")
    assert out["status"] == "OK"
    assert "lift" in out
