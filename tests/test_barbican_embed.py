import json
from barbican.embed import cosine, OllamaEmbedder


def test_cosine_identical_is_one():
    assert abs(cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_ollama_embedder_uses_injected_opener_no_network():
    calls = []

    class FakeResp:
        def __init__(self, payload):
            self._p = payload.encode()
        def read(self):
            return self._p
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_opener(req, timeout=None):
        calls.append(json.loads(req.data.decode()))
        return FakeResp(json.dumps({"embedding": [0.1, 0.2, 0.3]}))

    emb = OllamaEmbedder(_opener=fake_opener)
    out = emb(["hello", "world"])
    assert out == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert [c["prompt"] for c in calls] == ["hello", "world"]
    assert calls[0]["model"] == "nomic-embed-text"
