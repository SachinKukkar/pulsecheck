import numpy as np
import pytest

from pulsecheck.detector import video_labels, video_scores
from pulsecheck.metrics import best_threshold, detection_metrics, downsample_curve, equal_error_rate, hr_metrics


def test_hr_metrics_by_hand():
    m = hr_metrics(np.array([70.0, 82.0, 60.0, 101.0]), np.array([72.0, 80.0, 60.0, 90.0]))
    assert m["mae"] == pytest.approx((2 + 2 + 0 + 11) / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 0 + 121) / 4))
    assert m["bias"] == pytest.approx((-2 + 2 + 0 + 11) / 4)
    assert m["within_5bpm"] == pytest.approx(0.75)
    assert m["loa_low"] < m["bias"] < m["loa_high"]


def test_detection_metrics_perfect_and_chance():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = detection_metrics(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), 0.5, ci=False)
    assert m["auc"] == 1.0 and m["accuracy"] == 1.0 and m["eer"] == 0.0
    assert m["confusion"] == {"tn": 3, "fp": 0, "fn": 0, "tp": 3}
    rng = np.random.default_rng(0)
    yr = rng.integers(0, 2, 4000)
    assert abs(detection_metrics(yr, rng.random(4000), 0.5, curves=False, ci=False)["auc"] - 0.5) < 0.03


def test_threshold_and_eer():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    t = best_threshold(y, s)
    assert 0.4 <= t <= 0.7
    eer, _ = equal_error_rate(y, s)
    assert eer == pytest.approx(0.25)


def test_video_scores_average_windows_in_order():
    vids = np.array(["b", "b", "a", "a", "a"])
    ids, sc = video_scores(np.array([0.2, 0.4, 1.0, 0.5, 0.0]), vids)
    assert list(ids) == ["b", "a"] and np.allclose(sc, [0.3, 0.5])
    assert list(video_labels(np.array([1, 1, 0, 0, 0]), vids)) == [1, 0]


def test_downsample_keeps_ends():
    x = np.linspace(0, 1, 5000)
    d = downsample_curve(x, x**2, max_points=50)
    assert len(d["x"]) <= 50 and d["x"][0] == 0 and d["x"][-1] == 1
