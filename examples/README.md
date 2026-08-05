# Demo corpus

A small, self-contained corpus for trying BARBICAN without sourcing data first.

```bash
barbican detect   -i examples/demo_corpus.jsonl
barbican evaluate -i examples/demo_corpus.jsonl
```

21 posts: two constructed coordinated pushes (5 and 4 posts, distinct personas,
minutes apart) and 12 unrelated organic posts spread across days. `detect`
recovers both pushes and flags none of the organic posts; `evaluate` scores
recall 1.0 with 0 false clusters against the embedded ground truth.

Regenerate it with `python examples/build_demo_corpus.py` — the generator is
deterministic, so the committed file can be audited rather than trusted.

## Provenance

Every sentence was **written for this fixture and dedicated to the public
domain (CC0)**. No third-party corpus is used, quoted, or redistributed.

That is a deliberate choice rather than a convenience. The public datasets that
contain real influence operations carry licences that restrict what you may do
with them:

| Source | Licence | Consequence |
|---|---|---|
| ISI/USC "Labeled Datasets for Research on Information Operations" ([Zenodo 14141549](https://doi.org/10.5281/zenodo.14141549)) | CC BY-**NC-ND** 4.0 | No commercial use; no redistribution of derivatives — so no preprocessed JSONL may be shipped |
| [FiveThirtyEight IRA tweets](https://github.com/fivethirtyeight/russian-troll-tweets) | **None declared** | No licence means no grant; default is all rights reserved |
| [X/Twitter Information Operations Archive](https://blog.x.com/en_us/topics/company/2019/information-ops-on-twitter) | Platform terms; availability changed after 2021 | Verify current terms before relying on it |

Bundling any of those would place a licence encumbrance inside a shipped
product. This fixture carries none.

## What this demonstrates — and what it does not

**Does:** that a coordinated push is structurally distinguishable from organic
conversation, and that the detector separates them on narrative, temporal and
persona signals with no network access.

**Does not:** real-world detection performance. The posts are invented, the
municipal dispute is fictional, and no real person, organisation or campaign is
depicted. Ground truth is exact *because* the campaigns were constructed, which
is precisely why the numbers here are a demonstration and not a benchmark.

Performance claims require evaluation against attributed real-world influence
operations. That is a separate exercise, run locally against a licensed source
under its own terms, and is deliberately not bundled here — see the
"Real-world validation" section of the top-level README.
