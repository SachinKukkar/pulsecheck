"""Turn a patch's mean colour over time into a pulse waveform.

Three classic, training-free methods. All take ``rgb`` of shape (T, 3) (mean
R, G, B per frame) and return a pulse of shape (T,) whose sign follows blood
volume: up when there is more blood under the skin (skin reflects less green).

* GREEN - Verkruysse et al. 2008: the green channel alone, inverted.
* CHROM - de Haan & Jeanne 2013: two colour-difference signals chosen so that
  brightness changes (lighting, motion) cancel, combined with a tuned weight.
* POS   - Wang et al. 2017: projects colour onto a plane orthogonal to the
  skin tone, again cancelling brightness, then combines the two axes.

CHROM and POS work on short (1.6 s) windows that are overlap-added, so their
weights can follow changes in lighting over the clip.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import HR_BAND, RPPG_WINDOW_S
from .signals import bandpass, detrend

METHODS = ("pos", "chrom", "green")


def _safe_mean(x: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=keepdims)
    return np.where(np.abs(m) < 1e-9, 1e-9, m)


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    g = np.asarray(rgb, dtype=float)[:, 1]
    return -(g / _safe_mean(g, 0) - 1.0)


def _window_len(fs: float, win_s: float, n: int) -> int:
    w = int(np.ceil(win_s * fs))
    w += w % 2
    return int(min(max(w, 4), n))


def chrom(rgb: np.ndarray, fs: float, win_s: float = RPPG_WINDOW_S, band: tuple[float, float] = HR_BAND) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    n = len(rgb)
    w = _window_len(fs, win_s, n)
    hop = w // 2
    hann = np.hanning(w)
    out = np.zeros(n)
    starts = list(range(0, n - w + 1, hop))
    if starts and starts[-1] + w < n:
        starts.append(n - w)
    for s in starts:
        c = rgb[s : s + w]
        cn = c / _safe_mean(c, 0)
        x = 3 * cn[:, 0] - 2 * cn[:, 1]
        y = 1.5 * cn[:, 0] + cn[:, 1] - 1.5 * cn[:, 2]
        xf = bandpass(x, fs, *band)
        yf = bandpass(y, fs, *band)
        sy = yf.std()
        alpha = xf.std() / sy if sy > 1e-12 else 0.0
        out[s : s + w] += hann * (xf - alpha * yf)
    return out


def pos(rgb: np.ndarray, fs: float, win_s: float = RPPG_WINDOW_S) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    n = len(rgb)
    w = _window_len(fs, win_s, n)
    win = sliding_window_view(rgb, w, axis=0)  # (n - w + 1, 3, w)
    cn = win / _safe_mean(win, axis=2, keepdims=True)
    s0 = cn[:, 1] - cn[:, 2]  # G - B
    s1 = -2 * cn[:, 0] + cn[:, 1] + cn[:, 2]  # -2R + G + B
    sd1 = s1.std(axis=1)
    alpha = np.where(sd1 > 1e-12, s0.std(axis=1) / np.where(sd1 > 1e-12, sd1, 1.0), 0.0)
    h = s0 + alpha[:, None] * s1
    h -= h.mean(axis=1, keepdims=True)
    out = np.zeros(n)
    idx = np.arange(n - w + 1)[:, None] + np.arange(w)[None, :]
    np.add.at(out, idx, h)
    return -out  # POS's projection points against blood volume; flip to match GREEN/CHROM


def raw_pulse(rgb: np.ndarray, fs: float, method: str = "pos") -> np.ndarray:
    if method == "pos":
        return pos(rgb, fs)
    if method == "chrom":
        return chrom(rgb, fs)
    if method == "green":
        return green(rgb, fs)
    raise ValueError(f"unknown rPPG method {method!r}; choose from {METHODS}")


def pulse(rgb: np.ndarray, fs: float, method: str = "pos", band: tuple[float, float] = HR_BAND) -> np.ndarray:
    """Pulse waveform: method -> detrend -> zero-phase band-pass."""
    return bandpass(detrend(raw_pulse(rgb, fs, method)), fs, *band)
