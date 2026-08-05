# BARBICAN — synthetic influence-operation detection

Public defensive detector for coordinated synthetic social-post content. The
offensive twin (SPECTRE) is private and federal-only and is **not** part of this
package.

This is a self-contained, **zero runtime dependency** Python package (standard
library only). The content-scanner signals it relies on are vendored into
`barbican._scanners`, so importing `barbican` never loads any other Oubliette
package.

## Quickstart

Point it at a file of posts and ask whether anything in it is coordinated:

```bash
pip install oubliette-barbican
barbican detect -i examples/demo_corpus.jsonl
```

A ready-made corpus ships in `examples/` so there is nothing to source first —
21 posts, two constructed coordinated pushes and twelve organic controls. It is
CC0 text written for the fixture, so it carries no licence encumbrance. See
[`examples/README.md`](examples/README.md) for provenance and for what it does
and does not demonstrate.

```
Examined 7 posts (embeddings: none).
Thresholds: edge=0.6 coord=0.6 window=3600s

1 coordinated cluster(s):

  [1] 4 posts, score 0.795
      c1         a1         2026-01-01T10:00:00Z
                 The new port authority ruling is a disaster for working families ever...
      ...

Score is coordination evidence, not proof of inauthenticity: quotation,
syndication and genuine consensus also cluster.
```

**No network by default.** The correlator blends an embedding signal with
n-gram, temporal and persona signals; with embeddings off the embedder is never
consulted, so `detect` runs on a disconnected host using the standard library
alone. `--embed ollama` opts in to a local model server and buys paraphrase
robustness.

Input is JSONL, one post per line. `post_id`, `text`, `author_id` and
`timestamp` are required; `campaign_id`, `label` and `backend_model` are
evaluation metadata and optional.

```json
{"post_id": "c1", "author_id": "a1", "timestamp": "2026-01-01T10:00:00Z", "text": "..."}
```

Exit codes suit pipelines: `0` nothing flagged, `1` coordination found, `2` bad
input.

With ground truth available, score the discovered clusters against the known
campaigns rather than eyeballing them:

```bash
barbican evaluate -i labelled.jsonl -f json
```

`evaluate` refuses a corpus with no `campaign_id`, rather than reporting a
number computed against nothing.

## What it does

- **Operator CLI (`cli.py`).** `detect` and `evaluate` over a JSONL corpus —
  the analyst-facing surface over the library below.
- **Per-artifact baseline (`artifact.py`).** A deterministic single-surface
  detector: short-text lexical/statistical features plus two vendored scanner
  signals (invisible-text hits, AI-generation hits), combined by a linear
  baseline. Demonstrates — and bounds — what per-post scoring alone can achieve.
- **Coordination correlator (`correlator.py`).** The novelty: pairwise
  coordination signals (narrative n-gram/embedding similarity, temporal
  proximity, persona similarity) → thresholded graph → union-find clustering →
  per-cluster coordination score. Recovers coordinated campaigns that look
  individually authentic and only become suspicious in aggregate.
- **Embeddings (`embed.py`).** An injectable `EmbeddingFn` seam plus a local
  Ollama client that uses only stdlib `urllib` (no third-party HTTP dependency).
- **Evaluation harness (`experiment.py`).** Baseline, layered (baseline +
  correlator), and real-world-split conditions with precision/recall/F1 and
  campaign-level metrics; writes `results.json` + `RESULTS.md`.
- **Real-world loader (`realworld.py`).** Normalizes a documented intermediate
  JSONL into the `Post`/`Dataset` schema for a held-out generalization check.

## Boundary

The offensive SPECTRE framework and the eval-only corpus generator that imports
it (`corpus.py` in the private monorepo) are deliberately **excluded** from this
package. A regression test (`tests/test_barbican_smoke.py`) enforces the
invariant that `import barbican` succeeds standalone and pulls in **no**
`oubliette_shield` module.

## Real-world validation

The loader reads a documented **intermediate JSONL** — one object per line:

```json
{"text": "...", "author_id": "...", "timestamp": "ISO-8601", "label": "io|control", "campaign_id": "... | null"}
```

`label` maps `io`→`synthetic`, `control`→`authentic`; `campaign_id` is kept for
`io` posts (the coordinated group) and forced to null for controls. Producing
that JSONL from a specific licensed release is a one-time local preprocessing
step you run — no source dataset is bundled or committed. When the JSONL is
absent, `run_realworld_split` returns `status="SKIPPED"`; the real-world number
is never fabricated.

Suggested held-out corpus: ISI/USC "Labeled Datasets for Research on Information
Operations" (Zenodo 10.5281/zenodo.14141549) — labeled IO posts plus matched
organic control, per-post schema. Its license is CC BY-NC-ND 4.0 (NonCommercial):
usable for research/validation, **not** for bundling into a shipped product.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
ruff format --check src/
mypy --strict src/
```
