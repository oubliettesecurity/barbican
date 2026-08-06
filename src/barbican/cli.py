"""``barbican`` -- ask a corpus whether anything in it is coordinated.

The package's public API is written for experiments: it takes ``Dataset``
objects and returns metric dicts. An analyst has a file of posts and one
question, and no way to ask it. This is that question.

    barbican detect   -i posts.jsonl      # is anything here coordinated?
    barbican evaluate -i labelled.jsonl   # how well did we do against known truth?

**No network by default.** The correlator blends an embedding signal with
n-gram, temporal, and persona signals. At ``narrative_embed_weight=0.0`` the
embedder is never consulted, so ``detect`` runs on a disconnected host with the
standard library alone. Embeddings are opt-in via ``--embed ollama``, and buy
paraphrase robustness at the cost of a local model server.

Exit codes are meant for pipelines: ``0`` nothing flagged, ``1`` coordination
found, ``2`` the input or the request was bad.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any

from .correlator import Cluster, CorrelatorConfig, discover
from .embed import EmbeddingFn, OllamaEmbedder
from .types import Dataset, Post

#: Fields every record must carry. `campaign_id`, `label` and `backend_model`
#: are evaluation metadata and stay optional -- an operator's corpus has none.
REQUIRED_FIELDS = ("post_id", "text", "author_id", "timestamp")

EXIT_CLEAN = 0
EXIT_FOUND = 1
EXIT_BAD_INPUT = 2


class CLIError(Exception):
    """Something the operator can fix, reported without a traceback."""


def _zero_embedder(texts: list[str]) -> list[list[float]]:
    """Embedding stub used when embeddings are disabled.

    Never consulted for a score: at ``narrative_embed_weight=0.0`` the cosine
    term is multiplied by zero. It exists so ``discover`` keeps one code path
    whether or not a model server is available.
    """
    return [[0.0] for _ in texts]


def bundled_corpus() -> Path:
    """Path to the demo corpus shipped inside the package.

    Resolved through importlib.resources rather than __file__ arithmetic so it
    works from an installed wheel, not just a source checkout.
    """
    from importlib.resources import files

    return Path(str(files("barbican").joinpath("data/demo_corpus.jsonl")))


def load_posts(path: Path) -> list[Post]:
    """Read operator JSONL. Strict about structure, quiet about extra keys."""
    if not path.is_file():
        raise CLIError(f"input file not found: {path}")

    posts: list[Post] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CLIError(f"{path} line {lineno}: not valid JSON ({exc.msg})") from exc
        if not isinstance(rec, dict):
            raise CLIError(f"{path} line {lineno}: expected a JSON object")
        missing = [f for f in REQUIRED_FIELDS if f not in rec]
        if missing:
            raise CLIError(
                f"{path} line {lineno}: missing required field(s) {', '.join(missing)}. "
                f"Every record needs: {', '.join(REQUIRED_FIELDS)}."
            )
        posts.append(
            Post(
                post_id=str(rec["post_id"]),
                text=str(rec["text"]),
                author_id=str(rec["author_id"]),
                timestamp=str(rec["timestamp"]),
                campaign_id=rec.get("campaign_id"),
                backend_model=rec.get("backend_model"),
                label=str(rec.get("label", "unknown")),
            )
        )
    if not posts:
        raise CLIError(f"{path} contains no posts -- nothing to analyse")
    return posts


def config_from(args: argparse.Namespace) -> CorrelatorConfig:
    return CorrelatorConfig(
        edge_threshold=args.edge_threshold,
        coord_threshold=args.coord_threshold,
        time_window_seconds=args.window,
        # Embeddings off means the cosine term contributes nothing; the
        # narrative signal falls back to character n-gram overlap.
        narrative_embed_weight=0.5 if args.embed == "ollama" else 0.0,
    )


def _embedder(args: argparse.Namespace) -> EmbeddingFn:
    if args.embed == "ollama":
        return OllamaEmbedder(model=args.embed_model, host=args.embed_host)
    return _zero_embedder


def _cluster_dict(cluster: Cluster) -> dict[str, Any]:
    return {
        "post_ids": list(cluster.post_ids),
        "size": len(cluster.post_ids),
        "score": round(cluster.score, 4),
    }


def analyse(args: argparse.Namespace) -> tuple[list[Post], list[Cluster], CorrelatorConfig]:
    posts = load_posts(Path(args.input))
    cfg = config_from(args)
    return posts, discover(posts, _embedder(args), cfg), cfg


def cmd_detect(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    posts, clusters, cfg = analyse(args)
    flagged = [c for c in clusters if c.flagged]

    if args.format == "json":
        print(
            json.dumps(
                {
                    "schema": "barbican.detect/1",
                    "posts_examined": len(posts),
                    "embedding": args.embed,
                    "config": {
                        "edge_threshold": cfg.edge_threshold,
                        "coord_threshold": cfg.coord_threshold,
                        "time_window_seconds": cfg.time_window_seconds,
                        "narrative_embed_weight": cfg.narrative_embed_weight,
                    },
                    "clusters_found": len(clusters),
                    "flagged_clusters": [_cluster_dict(c) for c in flagged],
                },
                indent=2,
            ),
            file=out,
        )
    else:
        by_id = {p.post_id: p for p in posts}
        print(f"\nExamined {len(posts)} posts (embeddings: {args.embed}).", file=out)
        print(
            f"Thresholds: edge={cfg.edge_threshold} coord={cfg.coord_threshold} "
            f"window={cfg.time_window_seconds:g}s",
            file=out,
        )
        if not flagged:
            print("\nNo coordinated clusters found.", file=out)
        else:
            print(f"\n{len(flagged)} coordinated cluster(s):\n", file=out)
            for n, cluster in enumerate(flagged, 1):
                print(f"  [{n}] {len(cluster.post_ids)} posts, score {cluster.score:.3f}", file=out)
                for pid in cluster.post_ids:
                    post = by_id.get(pid)
                    if post is None:  # pragma: no cover - ids come from these posts
                        continue
                    text = post.text if len(post.text) <= 72 else post.text[:69] + "..."
                    print(f"      {pid:<10} {post.author_id:<10} {post.timestamp}", file=out)
                    print(f"                 {text}", file=out)
                print("", file=out)
        print(
            "Score is coordination evidence, not proof of inauthenticity: "
            "quotation, syndication and genuine consensus also cluster.",
            file=out,
        )
    return EXIT_FOUND if flagged else EXIT_CLEAN


def cmd_demo(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    """Run detect against the corpus bundled with the package."""
    corpus = bundled_corpus()
    if not corpus.is_file():  # pragma: no cover - packaging failure
        raise CLIError(
            f"the bundled demo corpus is missing from this install ({corpus}). "
            f"This is a packaging fault, not a usage error."
        )
    print(f"Demo corpus: {corpus.name} (bundled, CC0)", file=out)
    print(
        "Constructed fixture -- shows that coordinated pushes separate from "
        "organic conversation, NOT real-world detection performance.",
        file=out,
    )
    args.input = str(corpus)
    return cmd_detect(args, out, err)


def cmd_evaluate(args: argparse.Namespace, out: IO[str], err: IO[str]) -> int:
    from .experiment import campaign_metrics_discovery

    posts, clusters, cfg = analyse(args)
    if not any(p.campaign_id for p in posts):
        raise CLIError(
            "no campaign_id on any record -- evaluate scores discovered clusters "
            "against known campaigns, and cannot do that without ground truth. "
            "Use `detect` for an unlabelled corpus."
        )

    metrics = campaign_metrics_discovery(clusters, Dataset(posts), overlap=args.overlap)
    if args.format == "json":
        print(
            json.dumps(
                {
                    "schema": "barbican.evaluate/1",
                    "posts_examined": len(posts),
                    "campaigns": len(Dataset(posts).campaigns()),
                    "clusters_found": len(clusters),
                    "overlap": args.overlap,
                    "metrics": metrics,
                },
                indent=2,
            ),
            file=out,
        )
    else:
        print(
            f"\nExamined {len(posts)} posts against "
            f"{len(Dataset(posts).campaigns())} known campaign(s).",
            file=out,
        )
        for key in sorted(metrics):
            print(f"  {key:<24} {metrics[key]}", file=out)
    return EXIT_CLEAN


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-i", "--input", required=True, help="JSONL file of posts")
    parser.add_argument("-f", "--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--embed",
        choices=("none", "ollama"),
        default="none",
        help="embedding backend; 'none' (default) needs no network",
    )
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--embed-host", default="127.0.0.1:11434")
    parser.add_argument("--edge-threshold", type=float, default=CorrelatorConfig.edge_threshold)
    parser.add_argument("--coord-threshold", type=float, default=CorrelatorConfig.coord_threshold)
    parser.add_argument(
        "--window",
        type=float,
        default=CorrelatorConfig.time_window_seconds,
        help="temporal proximity window, seconds",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="barbican",
        description="Detect coordinated synthetic influence content in a corpus.",
        epilog=(
            "Exit codes: 0 nothing flagged, 1 coordination found, 2 bad input.\n"
            "Runs with no network unless --embed ollama is given."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p_detect = sub.add_parser("detect", help="find coordinated clusters in a corpus")
    _add_common(p_detect)
    p_detect.set_defaults(func=cmd_detect)

    p_eval = sub.add_parser("evaluate", help="score discovered clusters against known campaign_ids")
    _add_common(p_eval)
    p_eval.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="fraction of a campaign a cluster must cover to count as recovered",
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_demo = sub.add_parser("demo", help="run detect against the corpus bundled with the package")
    p_demo.add_argument("-f", "--format", choices=("text", "json"), default="text")
    p_demo.add_argument("--embed", choices=("none", "ollama"), default="none")
    p_demo.add_argument("--embed-model", default="nomic-embed-text")
    p_demo.add_argument("--embed-host", default="127.0.0.1:11434")
    p_demo.add_argument("--edge-threshold", type=float, default=CorrelatorConfig.edge_threshold)
    p_demo.add_argument("--coord-threshold", type=float, default=CorrelatorConfig.coord_threshold)
    p_demo.add_argument("--window", type=float, default=CorrelatorConfig.time_window_seconds)
    p_demo.set_defaults(func=cmd_demo)

    p_version = sub.add_parser("version", help="show the installed version")
    p_version.set_defaults(func=lambda a, o, e: _print_version(o))

    return parser


def _print_version(out: IO[str]) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("oubliette-barbican")
    except PackageNotFoundError:  # pragma: no cover - source checkout
        v = "unknown (not installed)"
    print(f"barbican {v} -- coordinated synthetic influence detection", file=out)
    return EXIT_CLEAN


def main(
    argv: Sequence[str] | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    if not getattr(args, "command", None):
        parser.print_help(out)
        return EXIT_CLEAN
    try:
        return int(args.func(args, out, err))
    except CLIError as exc:
        print(f"error: {exc}", file=err)
        return EXIT_BAD_INPUT


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
