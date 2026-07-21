"""BARBICAN Phase 1 evaluation harness: baseline condition, splits, metrics, reports."""

import json
from pathlib import Path
from typing import Any

from .artifact import BaselineDetector
from .correlator import Cluster, CorrelatorConfig, discover
from .embed import EmbeddingFn
from .types import Dataset, Post


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def artifact_metrics(
    det: BaselineDetector, posts: list[Post], threshold: float = 0.5
) -> dict[str, float]:
    tp = fp = fn = 0
    for p in posts:
        pred = det.predict(p.text, threshold)
        is_pos = p.label == "synthetic"
        if pred and is_pos:
            tp += 1
        elif pred and not is_pos:
            fp += 1
        elif not pred and is_pos:
            fn += 1
    return prf(tp, fp, fn)


def campaign_metrics_naive(
    det: BaselineDetector, ds: Dataset, threshold: float = 0.5, vote: float = 0.5
) -> dict[str, float]:
    campaigns = ds.campaigns()  # true synthetic campaigns
    tp = fn = 0
    for _cid, members in campaigns.items():
        flagged = sum(1 for p in members if det.predict(p.text, threshold))
        predicted_synthetic = (flagged / len(members)) >= vote if members else False
        if predicted_synthetic:
            tp += 1
        else:
            fn += 1
    out = prf(tp, fp=0, fn=fn)
    out["n_campaigns"] = float(len(campaigns))
    return out


def run_baseline(
    train: Dataset,
    evalsets: "dict[str, Dataset | None]",
    *,
    seed: int,
    out_dir: str,
    spec_hash: str,
) -> dict[str, Any]:
    det = BaselineDetector()
    det.fit(train.posts)
    results: dict[str, Any] = {
        "seed": seed,
        "spec_hash": spec_hash,
        "backends": sorted(train.backends()),
    }
    for name, ds in evalsets.items():
        if ds is None:
            results[name] = {"status": "SKIPPED"}
            continue
        results[name] = {
            "artifact": artifact_metrics(det, ds.posts),
            "campaign": campaign_metrics_naive(det, ds),
        }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    lines = [
        "# BARBICAN Phase 1 — baseline results",
        "",
        f"seed={seed} spec_hash={spec_hash} backends={results['backends']}",
        "",
    ]
    for name, r in results.items():
        if name in ("seed", "spec_hash", "backends"):
            continue
        if r.get("status") == "SKIPPED":
            lines.append(f"## {name}: SKIPPED (dataset not present)")
            continue
        a, c = r["artifact"], r["campaign"]
        lines.append(f"## {name}")
        lines.append(f"- artifact F1={a['f1']:.3f} (P={a['precision']:.3f} R={a['recall']:.3f})")
        lines.append(
            f"- campaign  R={c['recall']:.3f} over {int(c['n_campaigns'])} campaigns "
            f"(precision N/A in naive view)"
        )
        lines.append("")
    (out / "RESULTS.md").write_text("\n".join(lines))
    return results


def campaign_metrics_discovery(
    clusters: list[Cluster], ds: Dataset, *, overlap: float = 0.5
) -> dict[str, float]:
    campaigns = ds.campaigns()  # {campaign_id: [Post, ...]} for synthetic campaigns
    flagged = [set(c.post_ids) for c in clusters if c.flagged]
    recovered = 0
    matched_flagged = set()
    for _cid, members in campaigns.items():
        truth = {p.post_id for p in members}
        best = -1
        best_i = None
        for i, fset in enumerate(flagged):
            hit = len(truth & fset)
            if hit > best:
                best, best_i = hit, i
        if best_i is not None and best / len(truth) >= overlap:
            recovered += 1
            matched_flagged.add(best_i)
    false_clusters = len(flagged) - len(matched_flagged)
    recall = recovered / len(campaigns) if campaigns else 0.0
    return {
        "recall": recall,
        "n_campaigns": float(len(campaigns)),
        "false_clusters": float(false_clusters),
    }


def run_layered(
    train: Dataset,
    evalset: Dataset,
    embed: EmbeddingFn,
    cfg: CorrelatorConfig,
    *,
    seed: int,
    out_dir: str,
    spec_hash: str,
) -> dict[str, Any]:
    det = BaselineDetector()
    det.fit(train.posts)
    baseline = campaign_metrics_naive(det, evalset)
    clusters = discover(evalset.posts, embed, cfg)
    discovery = campaign_metrics_discovery(clusters, evalset)
    layered_recall = max(baseline["recall"], discovery["recall"])
    # Echo the correlator configuration so results carry their own provenance
    # (spec §5.3/§9/§11): no threshold is silently hard-coded downstream.
    correlator_config = {
        "edge_threshold": cfg.edge_threshold,
        "coord_threshold": cfg.coord_threshold,
        "time_window_seconds": cfg.time_window_seconds,
        "narrative_weight": cfg.narrative_weight,
        "temporal_weight": cfg.temporal_weight,
        "persona_weight": cfg.persona_weight,
        "ngram_n": cfg.ngram_n,
    }
    results: dict[str, Any] = {
        "seed": float(seed),
        "spec_hash": spec_hash,
        "backends": sorted(train.backends()),
        "correlator_config": correlator_config,
        "baseline_campaign_recall": baseline["recall"],
        "layered_campaign_recall": layered_recall,
        "lift": layered_recall - baseline["recall"],
        "false_clusters": discovery["false_clusters"],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    lines = [
        "# BARBICAN Phase 2 — layered (baseline + correlator) results",
        "",
        f"seed={seed} spec_hash={spec_hash} backends={results['backends']}",
        "",
        "## correlator config",
        f"- edge_threshold={cfg.edge_threshold} coord_threshold={cfg.coord_threshold}",
        f"- time_window_seconds={cfg.time_window_seconds} ngram_n={cfg.ngram_n}",
        f"- weights: narrative={cfg.narrative_weight} temporal={cfg.temporal_weight} "
        f"persona={cfg.persona_weight}",
        "",
        "## campaign recall",
        f"- baseline (per-artifact vote): {baseline['recall']:.3f}",
        f"- layered  (+ coordination):   {layered_recall:.3f}",
        f"- lift:                         {results['lift']:.3f}",
        f"- false_clusters:               {int(discovery['false_clusters'])}",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))
    return results


def run_realworld_split(
    train: Dataset,
    realworld: Dataset | None,
    embed: EmbeddingFn,
    cfg: CorrelatorConfig,
    *,
    seed: int,
    out_dir: str,
    spec_hash: str,
) -> dict[str, Any]:
    if realworld is None:
        return {"status": "SKIPPED", "reason": "real-world dataset not present"}
    result = run_layered(
        train, realworld, embed, cfg, seed=seed, out_dir=out_dir, spec_hash=spec_hash
    )
    return {"status": "OK", **result}
