import numpy as np
import pytest

from pulsecheck import signals


def test_article_example_72_bpm_lands_on_step_12():
    # 10 s at 30 fps = 300 numbers; 72 bpm = 1.2 Hz = one beat every 25 frames.
    fs, n = 30.0, 300
    t = np.arange(n) / fs
    x = 142.5 + 0.4 * np.sin(2 * np.pi * 1.2 * t)
    spec = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    assert freqs[1] - freqs[0] == pytest.approx(0.1)  # steps are 0.1 Hz apart
    assert int(np.argmax(spec)) == 12
    assert freqs[12] * 60 == pytest.approx(72.0)


@pytest.mark.parametrize("bpm", [48.0, 63.7, 72.0, 95.3, 150.0])
def test_hr_bpm_recovers_off_bin_rates(bpm):
    fs = 30.0
    t = np.arange(int(10 * fs)) / fs
    rng = np.random.default_rng(0)
    x = np.sin(2 * np.pi * bpm / 60 * t) + 0.3 * rng.standard_normal(len(t))
    x = signals.bandpass(signals.detrend(x), fs, 0.7, 3.0)
    assert signals.hr_bpm(x, fs) == pytest.approx(bpm, abs=1.5)


def test_bandpass_removes_drift_and_flicker():
    fs = 30.0
    t = np.arange(600) / fs
    pulse = np.sin(2 * np.pi * 1.2 * t)
    x = pulse + 5 * t / t[-1] + 0.8 * np.sin(2 * np.pi * 8.0 * t)  # slow drift + 8 Hz flicker
    y = signals.bandpass(signals.detrend(x), fs, 0.7, 3.0)
    core = slice(60, -60)  # ignore filter edges
    assert np.corrcoef(y[core], pulse[core])[0, 1] > 0.98


def test_snr_high_for_clean_low_for_noise():
    fs = 30.0
    t = np.arange(600) / fs
    rng = np.random.default_rng(1)
    clean = np.sin(2 * np.pi * 1.2 * t) + 0.1 * rng.standard_normal(len(t))
    noise = rng.standard_normal(len(t))
    f, p = signals.power_spectrum(signals.bandpass(clean, fs, 0.7, 3.0), fs)
    assert signals.snr_db(f, p, 1.2) > 5
    f, p = signals.power_spectrum(signals.bandpass(noise, fs, 0.7, 3.0), fs)
    assert signals.snr_db(f, p, 1.2) < 0


def test_resample_uniform_handles_jittered_clock_and_gaps():
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(1 / 40, 1 / 20, size=400))  # ~30 fps with jitter
    x = np.sin(2 * np.pi * 1.0 * t)
    x[50:55] = np.nan  # a few dropped detections
    tu, xu = signals.resample_uniform(t, x, 30.0)
    assert np.allclose(np.diff(tu), 1 / 30)
    assert np.isfinite(xu).all()
    assert np.max(np.abs(xu - np.sin(2 * np.pi * 1.0 * (tu + t[0])))) < 0.2


def test_windows_cover_clip():
    w = signals.windows(n=300, fs=30, win_s=8, step_s=4)
    assert w[0] == slice(0, 240)
    assert w[-1].stop == 300
    assert signals.windows(n=100, fs=30, win_s=8, step_s=4) == [slice(0, 100)]


def test_spectral_entropy_orders_tone_below_noise():
    fs = 30.0
    t = np.arange(600) / fs
    f, p_tone = signals.power_spectrum(np.sin(2 * np.pi * 1.2 * t), fs)
    _, p_noise = signals.power_spectrum(np.random.default_rng(3).standard_normal(600), fs)
    assert signals.spectral_entropy(f, p_tone) < signals.spectral_entropy(f, p_noise)
