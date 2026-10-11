"""Step 5: compare the patches.

On a real face every patch is lit by the same heart, so the patches' pulses
rise and fall together. A generated face has no heart of its own: its pulse is
copied, smeared, or missing, and it often disagrees with skin the generator
didn't touch (forehead edge, neck). Each 8 s window becomes a row of numbers
describing that agreement; the classifier learns which rows look fake.

Feature families (names are stable; the website shows them):
  global   - pooled SNR, spectral entropy, spread of per-patch heart rates,
             mean/min pairwise coherence and correlation, mean lag, HF ratio
  per-ROI  - SNR at the pooled rate, rate disagreement, leave-one-out
             correlation, high-band ratio, relative amplitude, high-band flicker
  per-pair - magnitude-squared coherence in the HR band, correlation, lag
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import signal as sps

from .config import DETECT_STEP_S, DETECT_WINDOW_S, HR_BAND, WIDE_BAND
from .rppg import raw_pulse
from .signals import band_power, bandpass, detrend, peak_frequency, power_spectrum, snr_db, spectral_entropy, windows
from .traces import Traces

MIN_WINDOW_S = 4.0
HIGH_BAND = (4.0, 12.0)
MAX_LAG_S = 0.33

SHORT = {
    "forehead_left": "FhL", "forehead_right": "FhR", "cheek_left": "ChL",
    "cheek_right": "ChR", "nose": "Nose", "chin": "Chin",
}


def feature_names(roi_names: list[str]) -> list[str]:
    names = [
        "snr_pooled", "entropy_pooled", "hr_spread_bpm", "hr_range_bpm", "coh_mean", "coh_min",
        "corr_mean", "corr_min", "lag_mean_abs_s", "hf_ratio_pooled",
    ]
    for r in roi_names:
        names += [f"snr_{r}", f"dhr_{r}", f"loo_corr_{r}", f"hf_ratio_{r}", f"rel_amp_{r}", f"flicker_{r}"]
    for a, b in combinations(roi_names, 2):
        names += [f"coh_{a}__{b}", f"corr_{a}__{b}", f"lag_{a}__{b}"]
    return names


def describe(name: str) -> str:
    """Short human label for charts, e.g. 'coh_cheek_left__forehead_right' -> 'coherence ChL-FhR'."""
    for key, label in (("coh_", "coherence"), ("corr_", "correlation"), ("lag_", "lag")):
        if name.startswith(key) and "__" in name:
            a, b = name[len(key):].split("__")
            return f"{label} {SHORT.get(a, a)}-{SHORT.get(b, b)}"
    for key, label in (("snr_", "SNR"), ("dhr_", "rate gap"), ("loo_corr_", "agreement"), ("hf_ratio_", "HF ratio"),
                       ("rel_amp_", "rel. amplitude"), ("flicker_", "flicker")):
        if name.startswith(key) and name[len(key):] in SHORT:
            return f"{label} {SHORT[name[len(key):]]}"
    return {
        "snr_pooled": "pooled SNR", "entropy_pooled": "spectral entropy", "hr_spread_bpm": "rate spread",
        "hr_range_bpm": "rate range", "coh_mean": "mean coherence", "coh_min": "min coherence",
        "corr_mean": "mean correlation", "corr_min": "min correlation", "lag_mean_abs_s": "mean lag",
        "hf_ratio_pooled": "pooled HF ratio",
    }.get(name, name)


def _std(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _nperseg(n: int) -> int:
    """Welch segment length: a power of two giving at least ~6 overlapping segments."""
    return int(min(n, max(32, 2 ** int(np.log2(n / 3)))))


def band_coherence(x: np.ndarray, y: np.ndarray, fs: float, nperseg: int, band=HR_BAND) -> float:
    """Magnitude-squared coherence over ``band``, weighted by sqrt(Pxx * Pyy).

    A plain average over the band is swamped by the many frequencies that hold
    only noise. Weighting by the two patches' own power concentrates it where
    both carry energy - on a real face, the heart rate - without favouring
    frequencies just because the patches happen to agree there.
    """
    f, pxy = sps.csd(x, y, fs=fs, nperseg=nperseg)
    _, pxx = sps.welch(x, fs=fs, nperseg=nperseg)
    _, pyy = sps.welch(y, fs=fs, nperseg=nperseg)
    sel = (f >= band[0]) & (f <= band[1])
    if not sel.any():
        return float("nan")
    coh = np.abs(pxy[sel]) ** 2 / (pxx[sel] * pyy[sel] + 1e-30)
    w = np.sqrt(pxx[sel] * pyy[sel])
    return float((coh * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def _max_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, int]:
    a, b = _std(a), _std(b)
    n = len(a)
    best, best_k = -np.inf, 0
    for k in range(-max_lag, max_lag + 1):
        c = np.dot(a[max(k, 0): n + min(k, 0)], b[max(-k, 0): n - max(k, 0)]) / (n - abs(k))
        if c > best:
            best, best_k = c, k
    return best, best_k


def clip_pulses(traces: Traces, method: str = "pos") -> dict[str, np.ndarray]:
    """Whole-clip pulses, computed once and sliced per window (no per-window filter edges)."""
    fs = traces.fs
    wide = (WIDE_BAND[0], min(WIDE_BAND[1], 0.45 * fs))
    roi_raw = np.stack([detrend(raw_pulse(traces.rgb[:, i], fs, method)) for i in range(len(traces.roi_names))], axis=1)
    g = traces.rgb[..., 1]
    g_norm = g / np.where(g.mean(axis=0) > 0, g.mean(axis=0), 1) - 1
    out = {
        "roi_hr": bandpass(roi_raw, fs, *HR_BAND),
        "roi_wide": bandpass(roi_raw, fs, *wide),
        "pooled_hr": bandpass(detrend(raw_pulse(traces.pooled(), fs, method)), fs, *HR_BAND),
        "pooled_wide": bandpass(detrend(raw_pulse(traces.pooled(), fs, method)), fs, *wide),
    }
    if wide[1] > HIGH_BAND[0] + 0.5:
        out["green_high"] = bandpass(detrend(g_norm), fs, HIGH_BAND[0], min(HIGH_BAND[1], 0.45 * fs))
    return out


def window_features(p: dict[str, np.ndarray], sl: slice, fs: float, roi_names: list[str]) -> dict[str, float]:
    R = len(roi_names)
    roi = p["roi_hr"][sl]
    wide = p["roi_wide"][sl]
    pooled = p["pooled_hr"][sl]
    n = len(pooled)
    hi = (HIGH_BAND[0], min(HIGH_BAND[1], 0.45 * fs))
    f: dict[str, float] = {}

    freqs, pw = power_spectrum(pooled, fs)
    f0 = peak_frequency(freqs, pw, HR_BAND)
    f["snr_pooled"] = snr_db(freqs, pw, f0)
    f["entropy_pooled"] = spectral_entropy(freqs, pw, HR_BAND)
    fw, pww = power_spectrum(p["pooled_wide"][sl], fs)
    f["hf_ratio_pooled"] = np.log10((band_power(fw, pww, hi) + 1e-20) / (band_power(fw, pww, HR_BAND) + 1e-20))

    stds = roi.std(axis=0) + 1e-12
    med_std = np.median(stds)
    rates = []
    for i, r in enumerate(roi_names):
        fr, pr = power_spectrum(roi[:, i], fs)
        fi = peak_frequency(fr, pr, HR_BAND)
        rates.append(60 * fi)
        f[f"snr_{r}"] = snr_db(fr, pr, f0)
        f[f"dhr_{r}"] = abs(60 * fi - 60 * f0)
        others = np.delete(roi, i, axis=1).mean(axis=1)
        f[f"loo_corr_{r}"] = float(np.corrcoef(roi[:, i], others)[0, 1])
        fwr, pwr = power_spectrum(wide[:, i], fs)
        f[f"hf_ratio_{r}"] = np.log10((band_power(fwr, pwr, hi) + 1e-20) / (band_power(fwr, pwr, HR_BAND) + 1e-20))
        f[f"rel_amp_{r}"] = float(np.log(stds[i] / med_std))
        f[f"flicker_{r}"] = float(np.log10(p["green_high"][sl, i].std() + 1e-9)) if "green_high" in p else np.nan
    rates = np.array(rates)
    f["hr_spread_bpm"] = float(np.std(rates))
    f["hr_range_bpm"] = float(np.ptp(rates))

    nperseg = _nperseg(n)
    max_lag = int(round(MAX_LAG_S * fs))
    cohs, corrs, lags = [], [], []
    for a, b in combinations(range(R), 2):
        coh = band_coherence(roi[:, a], roi[:, b], fs, nperseg)
        corr = float(np.corrcoef(roi[:, a], roi[:, b])[0, 1])
        _, k = _max_lag(roi[:, a], roi[:, b], max_lag)
        lag = abs(k) / fs
        key = f"{roi_names[a]}__{roi_names[b]}"
        f[f"coh_{key}"], f[f"corr_{key}"], f[f"lag_{key}"] = coh, corr, lag
        cohs.append(coh), corrs.append(corr), lags.append(lag)
    f["coh_mean"], f["coh_min"] = float(np.nanmean(cohs)), float(np.nanmin(cohs))
    f["corr_mean"], f["corr_min"] = float(np.nanmean(corrs)), float(np.nanmin(corrs))
    f["lag_mean_abs_s"] = float(np.mean(lags))
    return f


def video_features(
    traces: Traces,
    method: str = "pos",
    win_s: float = DETECT_WINDOW_S,
    step_s: float = DETECT_STEP_S,
) -> tuple[np.ndarray, list[str], list[tuple[float, float]]]:
    """Feature matrix (n_windows, n_features), feature names, and window (start, end) seconds."""
    names = feature_names(traces.roi_names)
    if traces.duration < MIN_WINDOW_S:
        return np.empty((0, len(names))), names, []
    p = clip_pulses(traces, method)
    rows, spans = [], []
    for sl in windows(len(traces.t), traces.fs, win_s, step_s):
        if (sl.stop - sl.start) / traces.fs < MIN_WINDOW_S:
            continue
        f = window_features(p, sl, traces.fs, traces.roi_names)
        rows.append([f.get(k, np.nan) for k in names])
        spans.append((sl.start / traces.fs, sl.stop / traces.fs))
    X = np.array(rows, dtype=float)
    X[~np.isfinite(X)] = np.nan
    return X, names, spans


def coherence_matrix(traces: Traces, method: str = "pos") -> np.ndarray:
    """Whole-clip HR-band coherence between every pair of patches (R x R, diagonal = 1)."""
    p = clip_pulses(traces, method)["roi_hr"]
    R = p.shape[1]
    nperseg = _nperseg(len(p))
    M = np.eye(R)
    for a, b in combinations(range(R), 2):
        M[a, b] = M[b, a] = band_coherence(p[:, a], p[:, b], traces.fs, nperseg)
    return M
