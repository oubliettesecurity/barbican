import dataclasses
import pytest

from barbican.types import Post, Dataset


def _post(**kw):
    base = dict(
        post_id="p1", text="hello", author_id="a1", campaign_id=None,
        timestamp="2020-01-01T00:00:00+00:00", backend_model=None, label="authentic",
    )
    base.update(kw)
    return Post(**base)


def test_post_is_frozen():
    p = _post()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.text = "mutated"


def test_by_label_filters():
    ds = Dataset(posts=[_post(post_id="s1", label="synthetic"), _post(post_id="a1", label="authentic")])
    assert [p.post_id for p in ds.by_label("synthetic")] == ["s1"]
    assert [p.post_id for p in ds.by_label("authentic")] == ["a1"]


def test_campaigns_groups_only_nonnull_campaign_ids():
    ds = Dataset(posts=[
        _post(post_id="s1", label="synthetic", campaign_id="C1"),
        _post(post_id="s2", label="synthetic", campaign_id="C1"),
        _post(post_id="a1", label="authentic", campaign_id=None),
    ])
    groups = ds.campaigns()
    assert set(groups) == {"C1"}
    assert [p.post_id for p in groups["C1"]] == ["s1", "s2"]


def test_backends_collects_nonnull():
    ds = Dataset(posts=[
        _post(post_id="s1", backend_model="qwen2.5:14b"),
        _post(post_id="s2", backend_model="qwen2.5:7b"),
        _post(post_id="a1", backend_model=None),
    ])
    assert ds.backends() == {"qwen2.5:14b", "qwen2.5:7b"}
