"""The bundled demo corpus must keep demonstrating what the README claims.

A fixture quoted in documentation is a promise. If a threshold or a signal
weight changes and the demo silently stops recovering its own campaigns, the
README becomes false and the first thing a new user runs stops working.

These also pin the generator as deterministic, so the committed JSONL can be
regenerated and diffed rather than trusted.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from barbican import cli

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
REPO = Path(__file__).resolve().parent.parent
CORPUS = REPO / "src" / "barbican" / "data" / "demo_corpus.jsonl"
GENERATOR = EXAMPLES / "build_demo_corpus.py"


def run(args):
    import io

    out, err = io.StringIO(), io.StringIO()
    code = cli.main(args, stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def _load_generator():
    spec = importlib.util.spec_from_file_location("build_demo_corpus", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _corpus_records():
    return [
        json.loads(line)
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_the_corpus_is_committed():
    assert CORPUS.is_file(), "the README tells users to run this file"


def test_detect_recovers_both_constructed_campaigns():
    code, out, err = run(["detect", "-i", str(CORPUS), "-f", "json"])
    assert code == 1, err
    report = json.loads(out)
    assert len(report["flagged_clusters"]) == 2, report["flagged_clusters"]


def test_no_organic_post_is_flagged():
    """The claim that carries the demo: it separates, rather than flags all."""
    _, out, _ = run(["detect", "-i", str(CORPUS), "-f", "json"])
    flagged = {pid for c in json.loads(out)["flagged_clusters"] for pid in c["post_ids"]}
    organic = {r["post_id"] for r in _corpus_records() if r["campaign_id"] is None}
    assert organic, "fixture must contain organic controls"
    assert not (flagged & organic), f"organic posts were flagged: {flagged & organic}"


def test_evaluate_reports_the_numbers_the_readme_quotes():
    _, out, _ = run(["evaluate", "-i", str(CORPUS), "-f", "json"])
    metrics = json.loads(out)["metrics"]
    assert metrics["recall"] == 1.0, metrics
    assert metrics["false_clusters"] == 0.0, metrics


def test_the_generator_reproduces_the_committed_corpus():
    """Regenerating must yield exactly what is committed."""
    module = _load_generator()
    assert module.build() == _corpus_records(), (
        "the committed corpus does not match what the generator produces"
    )


def test_no_third_party_text_is_bundled():
    """Provenance guard: the fixture is CC0 text written for it.

    A test cannot prove authorship. It can pin the claim that every line comes
    from the committed generator -- so text sourced from elsewhere would have to
    be added there, in the open, rather than dropped into an opaque data blob.
    """
    source = GENERATOR.read_text(encoding="utf-8")
    missing = [r["text"] for r in _corpus_records() if r["text"] not in source]
    assert not missing, f"corpus text not present in the generator: {missing[:2]}"


@pytest.mark.parametrize("doc", ["README.md"])
def test_examples_readme_states_the_limits(doc):
    """The fixture must not be readable as a performance benchmark."""
    text = (EXAMPLES / doc).read_text(encoding="utf-8").lower()
    assert "does not" in text
    assert "public domain" in text or "cc0" in text


def test_the_corpus_resolves_from_the_installed_package():
    """`barbican demo` must find it without knowing the repo layout."""
    from barbican.cli import bundled_corpus

    assert bundled_corpus().is_file(), "the bundled corpus is not importable"


def test_demo_runs_against_the_bundled_corpus():
    code, out, err = run(["demo", "-f", "json"])
    assert code == 1, err
    # `demo` prints a provenance banner before the report; parse from the JSON.
    report = json.loads(out[out.index("{") :])
    assert len(report["flagged_clusters"]) == 2


def test_the_corpus_ships_in_the_built_wheel():
    """The failure this file exists to prevent.

    The README tells a `pip install` user to run the demo. An `examples/`
    directory reaches only the sdist, so the corpus lived nowhere a wheel user
    could reach it -- the command was documented and impossible. Config alone
    would not have shown that; only looking inside the artifact does.
    """
    import zipfile

    wheels = sorted((REPO / "dist").glob("*.whl"))
    if not wheels:
        pytest.skip("no built wheel in dist/ to inspect")
    names = zipfile.ZipFile(wheels[-1]).namelist()
    assert any(n.endswith("barbican/data/demo_corpus.jsonl") for n in names), (
        f"demo corpus absent from {wheels[-1].name}; a pip-install user cannot run `barbican demo`"
    )
