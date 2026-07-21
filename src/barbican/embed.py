"""Embedding provider for the coordination correlator (public/defensive).

Injectable EmbeddingFn seam + a local Ollama client using stdlib urllib only.
Does NOT import oubliette_shield.spectre.
"""

import json
import math
import urllib.request
from collections.abc import Callable
from typing import Any

EmbeddingFn = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class OllamaEmbedder:
    """Calls a local Ollama /api/embeddings endpoint, one prompt per text."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "127.0.0.1:11434",
        *,
        timeout: float = 30.0,
        _opener: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self._opener = _opener or urllib.request.urlopen

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        url = f"http://{self.host}/api/embeddings"
        for text in texts:
            body = json.dumps({"model": self.model, "prompt": text}).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with self._opener(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
            out.append([float(x) for x in payload["embedding"]])
        return out
