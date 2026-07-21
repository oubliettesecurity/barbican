"""BARBICAN -- defensive detection of coordinated synthetic influence content.

Public defensive twin of the private SPECTRE offensive IO framework. This
standalone package ships zero runtime dependencies (stdlib only) and imports
nothing from oubliette_shield: the content-scanner signals it needs are
vendored in ``barbican._scanners``. The offensive twin (SPECTRE) and the
eval-only corpus generator that imports it are deliberately NOT part of this
package.
"""

from .artifact import BaselineDetector, extract_features
from .correlator import Cluster, CorrelatorConfig, discover
from .embed import EmbeddingFn, OllamaEmbedder, cosine
from .experiment import (
    campaign_metrics_discovery,
    run_baseline,
    run_layered,
    run_realworld_split,
)
from .realworld import load_realworld
from .types import Dataset, Post

__all__ = [
    "Post",
    "Dataset",
    "extract_features",
    "BaselineDetector",
    "run_baseline",
    "Cluster",
    "CorrelatorConfig",
    "discover",
    "EmbeddingFn",
    "OllamaEmbedder",
    "cosine",
    "campaign_metrics_discovery",
    "run_layered",
    "load_realworld",
    "run_realworld_split",
]
