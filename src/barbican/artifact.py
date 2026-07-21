"""Per-artifact authenticity features + single-surface baseline detector.

Short-text oriented: shield's scan_ai_generated is tuned for long injection
text (>=100 chars / >=3 sentences) and rarely fires on tweets, so we add
lexical/statistical short-text features and use the scanner only as one weak
signal among several.
"""

import math
import re

from . import _scanners as scanners
from .types import Post

_FEATURE_ORDER = [
    "len_chars",
    "len_words",
    "type_token_ratio",
    "hashtag_ratio",
    "url_count",
    "mention_ratio",
    "uppercase_ratio",
    "exclaim_count",
    "invisible_char_hits",
    "ai_scanner_hits",
]

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")


def extract_features(text: str) -> dict[str, float]:
    words = text.split()
    n_words = len(words) or 1
    letters = [c for c in text if c.isalpha()]
    uppercase_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    ttr = len({w.lower() for w in words}) / n_words
    invisible = scanners.scan_invisible_text(text)
    ai_hits = scanners.scan_ai_generated(text)
    return {
        "len_chars": float(len(text)),
        "len_words": float(len(words)),
        "type_token_ratio": ttr,
        "hashtag_ratio": len(_HASHTAG_RE.findall(text)) / n_words,
        "url_count": float(len(_URL_RE.findall(text))),
        "mention_ratio": len(_MENTION_RE.findall(text)) / n_words,
        "uppercase_ratio": uppercase_ratio,
        "exclaim_count": float(text.count("!")),
        "invisible_char_hits": float(len(invisible)),
        "ai_scanner_hits": float(len(ai_hits)),
    }


def _vec(text: str) -> list[float]:
    f = extract_features(text)
    return [f[k] for k in _FEATURE_ORDER]


class BaselineDetector:
    """Deterministic linear baseline: weight = mean(synthetic) - mean(authentic)."""

    def __init__(self) -> None:
        self._w: list[float] = []
        self._b: float = 0.0

    def fit(self, posts: list[Post]) -> None:
        syn = [_vec(p.text) for p in posts if p.label == "synthetic"]
        auth = [_vec(p.text) for p in posts if p.label == "authentic"]
        if not syn or not auth:
            raise ValueError("fit requires both synthetic and authentic posts")
        dim = len(_FEATURE_ORDER)

        def mean(rows: list[list[float]]) -> list[float]:
            return [sum(r[i] for r in rows) / len(rows) for i in range(dim)]

        m_syn, m_auth = mean(syn), mean(auth)
        allrows = syn + auth
        spread = [
            (max(r[i] for r in allrows) - min(r[i] for r in allrows)) or 1.0 for i in range(dim)
        ]
        # fold per-feature spread into the weights so no single feature dominates
        self._w = [(m_syn[i] - m_auth[i]) / (spread[i] ** 2) for i in range(dim)]
        midpoint = [(m_syn[i] + m_auth[i]) / 2 for i in range(dim)]
        self._b = -sum(self._w[i] * midpoint[i] for i in range(dim))

    def score(self, text: str) -> float:
        v = _vec(text)
        z = self._b + sum(self._w[i] * v[i] for i in range(len(self._w)))
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, text: str, threshold: float = 0.5) -> bool:
        return self.score(text) >= threshold
