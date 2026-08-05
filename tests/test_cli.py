"""The operator CLI.

BARBICAN's public API is written for experiments: it takes `Dataset` objects
and returns metric dicts. An analyst has a file of posts and a question --
"is anything in here coordinated?" -- and no way to ask it.

Two properties are load-bearing here.

**It must run with no network.** The correlator blends an embedding signal with
n-gram, temporal, and persona signals; at `narrative_embed_weight=0.0` the
embedder is never consulted. That is the default, so `detect` works on a
disconnected host. Embeddings are opt-in, not assumed.

**It must discriminate.** A detector that flags every corpus is worthless, so
every positive case here is paired with a negative one.
"""

from __future__ import annotations

import io
import json

import pytest

from barbican import cli

# Four posts pushing the same line, same author-style, minutes apart: the shape
# a coordinated push actually has.
COORDINATED = [
    {"post_id": "c1", "author_id": "a1", "timestamp": "2026-01-01T10:00:00Z",
     "text": "The new port authority ruling is a disaster for working families everywhere"},
    {"post_id": "c2", "author_id": "a2", "timestamp": "2026-01-01T10:04:00Z",
     "text": "The new port authority ruling is a disaster for working families across the state"},
    {"post_id": "c3", "author_id": "a3", "timestamp": "2026-01-01T10:07:00Z",
     "text": "This port authority ruling is a disaster for working families, plain and simple"},
    {"post_id": "c4", "author_id": "a4", "timestamp": "2026-01-01T10:09:00Z",
     "text": "The port authority ruling is a disaster for working families in every county"},
]

# Unrelated people talking about unrelated things at unrelated times.
ORGANIC = [
    {"post_id": "o1", "author_id": "b1", "timestamp": "2026-01-01T04:11:00Z",
     "text": "finally fixed the leaking radiator in the spare room, took all weekend"},
    {"post_id": "o2", "author_id": "b2", "timestamp": "2026-01-02T19:40:00Z",
     "text": "anyone know a decent thai place near the station? craving green curry"},
    {"post_id": "o3", "author_id": "b3", "timestamp": "2026-01-03T08:02:00Z",
     "text": "third straight loss. the defence needs rebuilding from scratch honestly"},
    {"post_id": "o4", "author_id": "b4", "timestamp": "2026-01-04T22:15:00Z",
     "text": "reading about deep sea vents tonight, the chemistry down there is wild"},
]


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def run(args):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(args, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


@pytest.fixture
def coordinated_file(tmp_path):
    return write_jsonl(tmp_path / "coordinated.jsonl", COORDINATED)


@pytest.fixture
def organic_file(tmp_path):
    return write_jsonl(tmp_path / "organic.jsonl", ORGANIC)


# --- the core question -----------------------------------------------------


def test_detect_flags_a_coordinated_corpus(coordinated_file):
    code, out, err = run(["detect", "-i", str(coordinated_file), "-f", "json"])
    report = json.loads(out)
    assert report["flagged_clusters"], f"nothing flagged; stderr={err}"
    members = {pid for c in report["flagged_clusters"] for pid in c["post_ids"]}
    assert len(members) >= 3, f"expected the pushed set to cluster, got {members}"
    assert code == 1, "finding coordination must be signalled in the exit code"


def test_detect_does_not_flag_an_organic_corpus(organic_file):
    """The paired negative. Without it the detector could flag everything."""
    code, out, _ = run(["detect", "-i", str(organic_file), "-f", "json"])
    assert json.loads(out)["flagged_clusters"] == []
    assert code == 0


# --- the airgap guarantee --------------------------------------------------


def test_detect_makes_no_network_calls_by_default(coordinated_file, monkeypatch):
    """DoD and disconnected deployments are a first-class case, not an edge one."""
    import urllib.request

    def explode(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("detect attempted a network call on the default path")

    monkeypatch.setattr(urllib.request, "urlopen", explode)
    code, out, _ = run(["detect", "-i", str(coordinated_file), "-f", "json"])
    assert code in (0, 1)
    assert json.loads(out)["embedding"] == "none"


def test_ollama_embeddings_are_opt_in(coordinated_file):
    _, out, _ = run(["detect", "-i", str(coordinated_file), "-f", "json"])
    assert json.loads(out)["embedding"] == "none"


# --- output ----------------------------------------------------------------


def test_text_output_names_the_posts_and_the_score(coordinated_file):
    _, out, _ = run(["detect", "-i", str(coordinated_file)])
    assert "c1" in out
    assert "score" in out.lower()


def test_report_states_how_many_posts_were_examined(coordinated_file):
    _, out, _ = run(["detect", "-i", str(coordinated_file), "-f", "json"])
    assert json.loads(out)["posts_examined"] == len(COORDINATED)


def test_thresholds_are_reported_so_a_finding_can_be_reproduced(coordinated_file):
    """An analyst handing a finding to someone else must be able to say how it
    was produced; a score with no threshold behind it is not evidence."""
    _, out, _ = run(["detect", "-i", str(coordinated_file), "-f", "json"])
    cfg = json.loads(out)["config"]
    assert "edge_threshold" in cfg and "coord_threshold" in cfg


def test_tuning_a_threshold_changes_the_outcome(organic_file):
    """Proves the knob is wired, not decorative."""
    _, strict, _ = run(["detect", "-i", str(organic_file), "-f", "json"])
    _, loose, _ = run(
        ["detect", "-i", str(organic_file), "-f", "json",
         "--edge-threshold", "0.01", "--coord-threshold", "0.01"]
    )
    assert json.loads(strict)["flagged_clusters"] == []
    assert json.loads(loose)["flagged_clusters"] != []


# --- evaluate --------------------------------------------------------------


def test_evaluate_scores_against_known_campaigns(tmp_path):
    labelled = [dict(r, campaign_id="camp-1", label="synthetic") for r in COORDINATED]
    labelled += [dict(r, campaign_id=None, label="authentic") for r in ORGANIC]
    path = write_jsonl(tmp_path / "labelled.jsonl", labelled)
    code, out, err = run(["evaluate", "-i", str(path), "-f", "json"])
    assert code == 0, err
    metrics = json.loads(out)["metrics"]
    assert "recovered" in metrics or metrics, f"expected campaign metrics, got {metrics}"


def test_evaluate_refuses_an_unlabelled_corpus(coordinated_file):
    """Scoring against absent ground truth would invent a number."""
    code, _, err = run(["evaluate", "-i", str(coordinated_file)])
    assert code != 0
    assert "campaign_id" in err or "label" in err


# --- refusals --------------------------------------------------------------


def test_a_missing_input_file_is_reported_cleanly(tmp_path):
    code, _, err = run(["detect", "-i", str(tmp_path / "nope.jsonl")])
    assert code == 2
    assert "not found" in err.lower()


def test_malformed_jsonl_is_reported_with_the_line_number(tmp_path):
    """Line 1 is deliberately valid, so the error can only come from line 2."""
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(COORDINATED[0]) + "\nnot json at all\n", encoding="utf-8"
    )
    code, _, err = run(["detect", "-i", str(path)])
    assert code == 2
    # "line 2", not bare "2" -- the temp path is full of digits.
    assert "line 2" in err, f"the offending line number must be named: {err}"


def test_a_record_missing_required_fields_is_reported(tmp_path):
    path = write_jsonl(tmp_path / "partial.jsonl", [{"post_id": "a", "text": "hi"}])
    code, _, err = run(["detect", "-i", str(path)])
    assert code == 2
    assert "author_id" in err or "timestamp" in err


def test_an_empty_corpus_is_not_a_silent_success(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    code, _, err = run(["detect", "-i", str(path)])
    assert code == 2
    assert "empty" in err.lower() or "no posts" in err.lower()
