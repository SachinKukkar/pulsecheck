import numpy as np
import pytest

from pulsecheck import rppg, signals

# Relative pulsatile strength of R, G, B for skin (normalised so green = 1):
# blood absorbs green most, then blue, then red.
PBV = np.array([0.33, 0.77, 0.53]) / 0.77
SKIN = np.array([180.0, 135.0, 115.0])


def synth_rgb(bpm=72.0, seconds=20.0, fs=30.0, flicker_hz=None, flicker_amp=0.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    p = np.sin(2 * np.pi * bpm / 60 * t)  # blood volume
    amp = 0.004  # 0.4% of skin colour, a realistic pulse depth
    rgb = SKIN * (1 - amp * p[:, None] * PBV)  # more blood -> less light reflected
    if flicker_hz:
        light = 1 + flicker_amp * np.sin(2 * np.pi * flicker_hz * t)  # brightness: same for R, G, B
        rgb = rgb * light[:, None]
    rgb = rgb + noise * rng.standard_normal(rgb.shape)
    return t, p, rgb


@pytest.mark.parametrize("method", rppg.METHODS)
def test_every_method_finds_clean_pulse_with_correct_sign(method):
    fs = 30.0
    _, p, rgb = synth_rgb(fs=fs)
    y = rppg.pulse(rgb, fs, method)
    assert signals.hr_bpm(y, fs) == pytest.approx(72.0, abs=1.0)
    core = slice(30, -30)
    assert np.corrcoef(y[core], p[core])[0, 1] > 0.9  # follows blood volume, not its inverse


@pytest.mark.parametrize("method", ["pos", "chrom"])
def test_pos_and_chrom_ignore_brightness_flicker(method):
    # A 1.8 Hz (108 bpm) lighting flicker twice as strong as the pulse.
    fs = 30.0
    _, _, rgb = synth_rgb(fs=fs, flicker_hz=1.8, flicker_amp=0.008)
    y = rppg.pulse(rgb, fs, method)
    assert signals.hr_bpm(y, fs) == pytest.approx(72.0, abs=1.5)


def test_green_is_fooled_by_the_same_flicker():
    fs = 30.0
    _, _, rgb = synth_rgb(fs=fs, flicker_hz=1.8, flicker_amp=0.008)
    y = rppg.pulse(rgb, fs, "green")
    assert signals.hr_bpm(y, fs) == pytest.approx(108.0, abs=1.5)


def test_pos_handles_short_and_low_fps_clips():
    fs = 15.0
    _, _, rgb = synth_rgb(fs=fs, seconds=6.0)
    y = rppg.pulse(rgb, fs, "pos")
    assert np.isfinite(y).all()
    assert signals.hr_bpm(y, fs) == pytest.approx(72.0, abs=3.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        rppg.raw_pulse(np.ones((100, 3)), 30.0, "nope")
