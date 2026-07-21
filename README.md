# BARBICAN — synthetic influence-operation detection

Public defensive detector for coordinated synthetic social-post content. The
offensive twin (SPECTRE) is private and federal-only and is **not** part of this
package.

This is a self-contained, **zero runtime dependency** Python package (standard
library only). The content-scanner signals it relies on are vendored into
`barbican._scanners`, so importing `barbican` never loads any other Oubliette
package.

## What it does

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
