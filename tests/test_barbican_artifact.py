from barbican.artifact import extract_features, BaselineDetector
from barbican.types import Post


def _post(text, label):
    return Post(post_id="p", text=text, author_id="a", campaign_id=None,
                timestamp="2020-01-01T00:00:00+00:00", backend_model=None, label=label)


def test_extract_features_is_deterministic():
    t = "Breaking!! The #election is RIGGED, share now http://x.io @friend"
    assert extract_features(t) == extract_features(t)


def test_extract_features_counts_signals():
    f = extract_features("#a #b hi http://x.io @you !!")
    assert f["hashtag_ratio"] > 0
    assert f["url_count"] == 1.0
    assert f["exclaim_count"] == 2.0


def test_baseline_separates_obvious_classes():
    syn = [_post(f"VOTE NOW #freedom #truth the system is rigged share http://x{i}.io", "synthetic")
           for i in range(6)]
    auth = [_post(t, "authentic") for t in [
        "ugh monday again lol",
        "my cat knocked over the plant :(",
        "anyone know a good taco place downtown?",
        "can't believe that game last night!!!",
        "running late, ttyl",
        "the weather is so nice today",
    ]]
    det = BaselineDetector()
    det.fit(syn + auth)
    syn_scores = [det.score(p.text) for p in syn]
    auth_scores = [det.score(p.text) for p in auth]
    assert min(syn_scores) > max(auth_scores)  # perfectly separable on this fixture


def test_baseline_predict_uses_threshold():
    det = BaselineDetector()
    det.fit([_post("VOTE NOW #a #b http://x.io rigged share", "synthetic"),
             _post("lol ok whatever", "authentic")])
    assert det.predict("VOTE NOW #a #b http://x.io rigged share") is True
    assert det.predict("lol ok whatever") is False
