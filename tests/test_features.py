import numpy as np

from pulsecheck.features import band_coherence, coherence_matrix, describe, feature_names, video_features

from .conftest import make_traces


def test_coherence_high_when_patches_share_a_heart_low_when_not():
    shared = coherence_matrix(make_traces([72] * 6))
    split = coherence_matrix(make_traces([72, 72, 95, 95, 95, 95], seed=1))  # forehead vs swapped inner face
    off = ~np.eye(6, dtype=bool)
    assert shared[off].mean() > 0.85
    assert split[:2, 2:].mean() < 0.4  # forehead disagrees with the rest
    assert split[0, 1] > 0.85  # but the two forehead halves still agree


def test_band_coherence_ignores_unrelated_noise():
    rng = np.random.default_rng(3)
    a, b = rng.standard_normal(480), rng.standard_normal(480)
    assert band_coherence(a, b, 30.0, 64) < 0.4


def test_video_features_shape_and_names():
    tr = make_traces([72] * 6)
    X, names, spans = video_features(tr)
    assert X.shape == (len(spans), len(names)) == (3, 91)
    assert names == feature_names(tr.roi_names)
    assert spans[0] == (0.0, 8.0)
    assert describe("coh_cheek_left__forehead_right") == "coherence ChL-FhR"
    assert describe("snr_pooled") == "pooled SNR"


def test_features_separate_shared_from_mismatched_pulse():
    names = feature_names(["forehead_left", "forehead_right", "cheek_left", "cheek_right", "nose", "chin"])
    i = names.index("coh_mean")
    real, _, _ = video_features(make_traces([70] * 6, seed=4))
    fake, _, _ = video_features(make_traces([70, 70, 88, 88, 88, 88], seed=5))
    assert real[:, i].min() > fake[:, i].max()


def test_too_short_clip_gives_no_windows():
    X, names, spans = video_features(make_traces([72] * 6, seconds=3.0))
    assert X.shape == (0, len(names)) and spans == []
