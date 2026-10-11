"""The PPG-map CNN (skipped when PyTorch is not installed)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pulsecheck.deep import CROP, MapTable, WIN, build_model, ppg_maps, train_cnn  # noqa: E402

from .conftest import make_traces  # noqa: E402


def test_ppg_maps_shape_and_standardisation():
    X, spans = ppg_maps(make_traces([72] * 6, seconds=16.0))
    assert X.shape == (3, 2, 32, WIN) and len(spans) == 3
    assert np.allclose(X.mean(axis=-1), 0, atol=1e-4) and np.allclose(X.std(axis=-1), 1, atol=1e-3)


def test_traces_without_grid_are_refused():
    with pytest.raises(ValueError):
        ppg_maps(make_traces([72] * 6, grid=False))


def test_model_output_shape_and_size():
    m = build_model()
    assert m(torch.zeros(4, 2, 32, CROP)).shape == (4, 1)
    assert 50_000 < sum(p.numel() for p in m.parameters()) < 150_000


def _table(n, seed):
    # Learnable toy task: "fake" maps have their rows shuffled in time, "real" ones don't.
    rng = np.random.default_rng(seed)
    t = np.arange(WIN) / 30.0
    X, y = [], []
    for i in range(n):
        base = np.sin(2 * np.pi * rng.uniform(1, 2) * t)
        rows = np.stack([base + 0.3 * rng.standard_normal(WIN) for _ in range(32)])
        if i % 2:
            rows = np.stack([np.roll(r, rng.integers(10, 60)) for r in rows])
        X.append(np.stack([rows, rows]))
        y.append(i % 2)
    vids = np.array([f"v{i}" for i in range(n)])
    return MapTable(np.array(X, np.float16), np.array(y), vids, vids, np.array(["train"] * n), np.array(["m"] * n))


def test_training_learns_a_toy_task_on_cpu():
    model, history, info = train_cnn(_table(160, 0), _table(60, 1), epochs=6, batch=32, device="cpu", log=lambda *_: None)
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert max(h["val_video_auc"] for h in history) > 0.8
    assert info["parameters"] > 0 and info["best_epoch"] >= 1


def test_saved_model_reloads_in_safe_mode_and_predicts(tmp_path):
    from pulsecheck.deep import CNNDetector

    model, _, info = train_cnn(_table(64, 2), _table(32, 3), epochs=1, batch=32, device="cpu", log=lambda *_: None)
    CNNDetector(model, 0.42, {"rppg_method": "pos", **info}).save(tmp_path / "cnn.pt")
    det = CNNDetector.load(tmp_path / "cnn.pt")  # torch.load(weights_only=True) inside
    assert det.threshold == pytest.approx(0.42) and det.meta["torch"] == info["torch"]
    probs, spans, _ = det.predict_traces(make_traces([72] * 6, seconds=16.0))
    assert len(probs) == len(spans) == 3 and ((0 <= probs) & (probs <= 1)).all()
