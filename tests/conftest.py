"""Shared fixtures.

The end-to-end tests need a face video whose true heart rate is known, so they
render one with pulsecheck.synthetic (a still portrait with a pulse painted on).
That clip exists only to check the code against a known answer; no result
reported anywhere in the project comes from synthetic data.
"""

import numpy as np
import pytest

from pulsecheck.traces import Traces

PBV = np.array([0.33, 0.77, 0.53]) / 0.77


@pytest.fixture(scope="session")
def known_pulse_clip(tmp_path_factory):
    pytest.importorskip("skimage")
    pytest.importorskip("imageio_ffmpeg")
    from pulsecheck.synthetic import SynthParams, make_clip

    path = tmp_path_factory.mktemp("clips") / "known_78bpm.mp4"
    # An ordinary lossy H.264 encode, like a real upload (and small enough for the upload limit).
    truth = make_clip(path, SynthParams(kind="real", bpm=78.0, seconds=12.0, seed=11), crf=17)
    return path, truth


def make_traces(roi_bpm, fs=30.0, seconds=16.0, noise=0.03, seed=0, grid=True):
    """Synthetic Traces whose six patches pulse at the given rates (same rate = same heart)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    skin = np.array([180.0, 135.0, 115.0])
    rgb = []
    for bpm in roi_bpm:
        p = np.sin(2 * np.pi * bpm / 60 * t + (0 if bpm == roi_bpm[0] else rng.uniform(0, 6.3)))
        rgb.append(skin * (1 - 0.005 * p[:, None] * PBV) + noise * rng.standard_normal((len(t), 3)))
    rgb = np.stack(rgb, axis=1)
    names = ["forehead_left", "forehead_right", "cheek_left", "cheek_right", "nose", "chin"][: len(roi_bpm)]
    grid_rgb = None
    if grid:
        base = rgb[:, [k % len(roi_bpm) for k in range(32)]]
        grid_rgb = base + noise * rng.standard_normal(base.shape)
    return Traces(t=t, fs=fs, rgb=rgb, counts=np.full((len(t), len(roi_bpm)), 400), roi_names=names,
                  motion=np.zeros(len(t)), meta={"skin_rgb": list(skin), "face_fraction": 1.0}, grid_rgb=grid_rgb)
