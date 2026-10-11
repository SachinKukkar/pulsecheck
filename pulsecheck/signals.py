"""Steps 3-4: separate the pulse from the junk, then read the beats.

* resample onto a uniform clock (webcams drop and jitter frames)
* detrend (remove slow drift) and band-pass (keep where a heartbeat can live)
* Fourier transform -> the strongest frequency in the band x 60 = beats a minute
* SNR: how much of the band's power sits on the heart rate and its harmonic
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from .config import HR_BAND


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs along axis 0 (each column separately).

    Leading/trailing NaNs are filled with the nearest valid value. Columns with
    no valid samples stay NaN.
    """
    x = np.array(x, dtype=float, copy=True)
    flat = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.all() or not ok.any():
            continue
        col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return flat.reshape(x.shape)


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample samples taken at times ``t`` onto a uniform ``fs`` grid.

    ``x`` may have any trailing shape; interpolation is linear along axis 0.
    """
    t = np.asarray(t, dtype=float)
    x = fill_nans(np.asarray(x, dtype=float))
    n = int(np.floor((t[-1] - t[0]) * fs)) + 1
    tu = t[0] + np.arange(n) / fs
    flat = x.reshape(len(x), -1)
    out = np.empty((n, flat.shape[1]))
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(tu, t, flat[:, j])
    return tu - tu[0], out.reshape((n,) + x.shape[1:])


def detrend(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove a straight-line drift (``scipy.signal.detrend``)."""
    return sps.detrend(np.asarray(x, dtype=float), axis=axis, type="linear")


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3, axis: int = 0) -> np.ndarray:
    """Zero-phase Butterworth band-pass. ``high`` is clipped below Nyquist."""
    high = min(high, 0.45 * fs)
    if high <= low:
        raise ValueError(f"band {low}-{high} Hz is empty at fs={fs}")
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    x = np.asarray(x, dtype=float)
    padlen = min(3 * (2 * len(sos) + 1), x.shape[axis] - 1)
    return sps.sosfiltfilt(sos, x, axis=axis, padlen=padlen)


def next_pow2(n: int) -> int:
    return 1 << max(int(n) - 1, 1).bit_length()


def power_spectrum(x: np.ndarray, fs: float, pad_factor: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram, zero-padded ``pad_factor`` x.

    Zero-padding does not add information - a 10 s clip still only *resolves*
    frequencies 0.1 Hz (6 bpm) apart - but it interpolates the spectrum so the
    peak of a single dominant pulse can be located between FFT bins.
    """
    x = np.asarray(x, dtype=float)
    nfft = next_pow2(len(x) * pad_factor)
    freqs, power = sps.periodogram(x, fs=fs, window="hann", nfft=nfft, detrend="constant", scaling="density")
    return freqs, power


def peak_frequency(freqs: np.ndarray, power: np.ndarray, band: tuple[float, float] = HR_BAND) -> float:
    """Frequency (Hz) of the largest spectral peak inside ``band``, refined by a parabola fit."""
    sel = np.flatnonzero((freqs >= band[0]) & (freqs <= band[1]))
    if len(sel) == 0:
        return float("nan")
    i = sel[np.argmax(power[sel])]
    if 0 < i < len(power) - 1:
        a, b, c = power[i - 1], power[i], power[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            shift = 0.5 * (a - c) / denom
            return float(freqs[i] + shift * (freqs[1] - freqs[0]))
    return float(freqs[i])


def hr_bpm(pulse: np.ndarray, fs: float, band: tuple[float, float] = HR_BAND) -> float:
    freqs, power = power_spectrum(pulse, fs)
    return 60.0 * peak_frequency(freqs, power, band)


def snr_db(
    freqs: np.ndarray,
    power: np.ndarray,
    f0: float,
    halfwidth: float = 0.1,
    band: tuple[float, float] = (0.5, 4.0),
) -> float:
    """de Haan & Jeanne (2013) SNR around a heart rate ``f0`` (Hz).

    Signal = power within +/- ``halfwidth`` of f0 and of its first harmonic 2*f0;
    noise = every other bin in ``band``. 0 dB means as much power off the pulse
    as on it.
    """
    if not np.isfinite(f0):
        return float("nan")
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    sig = (np.abs(freqs - f0) <= halfwidth) | (np.abs(freqs - 2 * f0) <= 2 * halfwidth)
    s = power[in_band & sig].sum()
    n = power[in_band & ~sig].sum()
    if s <= 0 or n <= 0:
        return float("nan")
    return float(10 * np.log10(s / n))


def spectral_entropy(freqs: np.ndarray, power: np.ndarray, band: tuple[float, float] = HR_BAND) -> float:
    """Normalised Shannon entropy of the in-band spectrum: 0 = one pure tone, 1 = flat noise."""
    sel = (freqs >= band[0]) & (freqs <= band[1])
    p = power[sel]
    total = p.sum()
    if total <= 0 or len(p) < 2:
        return float("nan")
    p = p / total
    return float(-(p * np.log(p + 1e-300)).sum() / np.log(len(p)))


def band_power(freqs: np.ndarray, power: np.ndarray, band: tuple[float, float]) -> float:
    sel = (freqs >= band[0]) & (freqs <= band[1])
    return float(power[sel].sum() * (freqs[1] - freqs[0])) if sel.any() else 0.0


def windows(n: int, fs: float, win_s: float, step_s: float) -> list[slice]:
    """Slices for sliding windows. A clip shorter than one window gets one window."""
    w = int(round(win_s * fs))
    s = max(int(round(step_s * fs)), 1)
    if n <= w:
        return [slice(0, n)]
    starts = list(range(0, n - w + 1, s))
    if n - (starts[-1] + w) >= s // 2:  # keep the tail when at least half a hop is left over
        starts.append(n - w)
    return [slice(a, a + w) for a in starts]


def hr_timeline(
    pulse: np.ndarray,
    fs: float,
    win_s: float,
    step_s: float,
    band: tuple[float, float] = HR_BAND,
) -> dict[str, list[float]]:
    """Heart rate and SNR on sliding windows of an already band-passed pulse."""
    t, bpm, snr = [], [], []
    for sl in windows(len(pulse), fs, win_s, step_s):
        seg = pulse[sl]
        if len(seg) < fs * 3:
            continue
        freqs, power = power_spectrum(seg, fs)
        f0 = peak_frequency(freqs, power, band)
        t.append((sl.start + sl.stop) / 2 / fs)
        bpm.append(60 * f0)
        snr.append(snr_db(freqs, power, f0))
    return {"t": t, "bpm": bpm, "snr_db": snr}
