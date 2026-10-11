"""One video in, everything the website shows out (all JSON-serialisable)."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .config import HR_BAND, HR_STEP_S, HR_WINDOW_S, MAX_ANALYZE_S
from .features import coherence_matrix, describe
from .pipeline import Preview, extract_traces, ita_category, skin_ita
from .rppg import METHODS, pulse
from .signals import bandpass, detrend, hr_timeline, peak_frequency, power_spectrum, snr_db
from .traces import Traces

UNCERTAIN_MARGIN = 0.1  # within this of the threshold we say "uncertain"


def _r(x, nd=4):
    return [round(float(v), nd) if np.isfinite(v) else None for v in np.asarray(x, dtype=float)]


def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def _face_crop(img: np.ndarray, lmk: np.ndarray, margin: float = 0.4) -> np.ndarray:
    """Square crop around the face so the patches are visible, not a speck in a wide frame."""
    (x0, y0), (x1, y1) = lmk.min(axis=0), lmk.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * (0.5 + margin)
    H, W = img.shape[:2]
    return img[int(max(cy - half, 0)):int(min(cy + half, H)), int(max(cx - half, 0)):int(min(cx + half, W))]


def _jpeg_data_url(rgb: np.ndarray, max_side: int = 520, min_side: int = 360) -> str:
    h, w = rgb.shape[:2]
    s = min(max_side / max(h, w), max(1.0, min_side / max(h, w)))
    if s != 1:
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def quality_checks(traces: Traces, snr: float, file_bytes: int | None) -> tuple[dict, list[dict]]:
    m = traces.meta
    q = {
        "duration_s": round(traces.duration, 2),
        "face_fraction": round(m.get("face_fraction", 1.0), 3),
        "face_px": round(m.get("face_px", 0.0), 1),
        "motion": round(float(np.median(traces.motion)), 3),
        "effective_fps": round(m.get("effective_fps", traces.fs), 1),
        "snr_db": round(snr, 2) if np.isfinite(snr) else None,
        "bits_per_pixel": None,
    }
    w, h = m.get("original_size") or [0, 0]
    if file_bytes and w and h and q["effective_fps"]:
        q["bits_per_pixel"] = round(file_bytes * 8 / max(traces.duration, 0.1) / (w * h * q["effective_fps"]), 4)
    warn = []

    def add(level, text):
        warn.append({"level": level, "text": text})

    if q["duration_s"] < 10:
        add("warn", f"Only {q['duration_s']:.0f} s of video: heart-rate steps are {60 / max(q['duration_s'], 1):.0f} bpm apart. 15-30 s is better.")
    if q["face_fraction"] < 0.95:
        add("warn", f"The face was lost in {1 - q['face_fraction']:.0%} of frames.")
    if q["face_px"] and q["face_px"] < 70:
        add("warn", "The face is small in the frame; move closer for more skin pixels.")
    if q["motion"] > 0.35:
        add("warn", "A lot of head movement; movement is the main source of false alarms.")
    if q["effective_fps"] < 15:
        add("warn", f"Low frame rate ({q['effective_fps']:.0f} fps).")
    if q["bits_per_pixel"] is not None and q["bits_per_pixel"] < 0.03:
        add("warn", "Heavily compressed video: compression wipes out much of the faint colour change.")
    if q["snr_db"] is not None and q["snr_db"] < -2:
        add("bad", "Weak pulse signal (low SNR): treat the heart rate and the verdict as unreliable.")
    elif q["snr_db"] is not None and q["snr_db"] > 3:
        add("good", "Clear pulse signal.")
    return q, warn


def analyze_traces(traces: Traces, preview: Preview | None = None, *, method: str = "pos", detector=None,
                   file_bytes: int | None = None) -> dict:
    fs = traces.fs
    t = traces.t
    pooled = traces.pooled()
    g_raw = pooled[:, 1]
    g_detr = detrend(g_raw)
    g_filt = bandpass(g_detr, fs, *HR_BAND)
    pulses = {m: pulse(pooled, fs, m) for m in METHODS}
    main = pulses[method]

    freqs, power = power_spectrum(main, fs)
    f0 = peak_frequency(freqs, power, HR_BAND)
    snr = snr_db(freqs, power, f0)
    # The article's plain FFT: 1/duration Hz bins, no zero-padding.
    raw_f = np.fft.rfftfreq(len(main), 1 / fs)
    raw_p = np.abs(np.fft.rfft(main * np.hanning(len(main)))) ** 2
    show = freqs <= 4.0
    raw_show = raw_f <= 4.0

    per_method = {}
    for m, p in pulses.items():
        fm, pm = power_spectrum(p, fs)
        fm0 = peak_frequency(fm, pm, HR_BAND)
        per_method[m] = {"bpm": round(60 * fm0, 1), "snr_db": round(snr_db(fm, pm, fm0), 2)}

    roi_pulses = {name: _r(_z(pulse(traces.rgb[:, i], fs, method)), 3) for i, name in enumerate(traces.roi_names)}
    roi_hr = {}
    for i, name in enumerate(traces.roi_names):
        fr, pr = power_spectrum(pulse(traces.rgb[:, i], fs, method), fs)
        roi_hr[name] = round(60 * peak_frequency(fr, pr, HR_BAND), 1)

    quality, warnings = quality_checks(traces, snr, file_bytes)
    ita = skin_ita(traces.meta["skin_rgb"])

    result = {
        "video": {k: traces.meta.get(k) for k in ("original_size", "nominal_fps", "effective_fps", "clock", "frames_decoded")},
        "quality": quality,
        "warnings": warnings,
        "method": method,
        "hr": {
            "bpm": round(60 * f0, 1), "snr_db": round(snr, 2) if np.isfinite(snr) else None,
            "per_method": per_method, "per_roi": roi_hr,
            "timeline": {k: _r(v, 2) for k, v in hr_timeline(main, fs, HR_WINDOW_S, HR_STEP_S).items()},
            "bin_spacing_bpm": round(60 / traces.duration, 2),
        },
        "signals": {
            "t": _r(t, 3), "green_raw": _r(g_raw, 3), "green_detrended": _r(g_detr, 4),
            "green_filtered": _r(g_filt, 4), "pulse": _r(_z(main), 3),
        },
        "spectrum": {
            "bpm": _r(60 * freqs[show], 2), "power": _r(power[show] / (power[show].max() + 1e-30), 4),
            "raw_bpm": _r(60 * raw_f[raw_show], 2), "raw_power": _r(raw_p[raw_show] / (raw_p[raw_show].max() + 1e-30), 4),
            "band_bpm": [60 * HR_BAND[0], 60 * HR_BAND[1]],
        },
        "rois": {"names": traces.roi_names, "pulses": roi_pulses,
                 "coherence": [_r(row, 3) for row in coherence_matrix(traces, method)]},
        "skin": {"ita": round(ita, 1), "category": ita_category(ita)},
        "preview": (_jpeg_data_url(_face_crop(preview.overlay, preview.landmarks))
                    if preview is not None and preview.overlay is not None else None),
        "detector": None,
    }

    if detector is not None:
        probs, spans, Xd = detector.predict_traces(traces, method)
        if len(probs):
            p = float(probs.mean())
            thr = detector.threshold
            verdict = "uncertain" if abs(p - thr) < UNCERTAIN_MARGIN else ("likely fake" if p >= thr else "likely real")
            profile = []
            for f in (detector.meta.get("feature_profile", [])[:6] if Xd is not None else []):
                j = detector.feature_names.index(f["feature"])
                v = float(np.nanmean(Xd[:, j]))
                profile.append({**f, "value": round(v, 4) if np.isfinite(v) else None})
            result["detector"] = {
                "prob_fake": round(p, 4), "threshold": round(thr, 4), "verdict": verdict,
                "windows": [{"t0": round(a, 2), "t1": round(b, 2), "prob": round(float(q), 4)} for (a, b), q in zip(spans, probs)],
                "profile": profile,
                "model": {k: detector.meta.get(k) for k in ("model_label", "train_source", "test_video_auc", "trained_at")},
            }
        else:
            result["detector"] = {"verdict": "too short", "prob_fake": None, "windows": []}
    return result


def analyze_video(path: str | Path, *, method: str = "pos", max_s: float = MAX_ANALYZE_S, start_s: float = 0.0,
                  detector=None, progress=None) -> dict:
    t0 = time.time()
    traces, preview = extract_traces(path, start_s=start_s, max_s=max_s, want_preview=True, progress=progress)
    t1 = time.time()
    if progress:
        progress("Reading the pulse and comparing patches", 0.0)
    result = analyze_traces(traces, preview, method=method, detector=detector, file_bytes=os.path.getsize(path))
    result["timing"] = {"extract_s": round(t1 - t0, 2), "analyze_s": round(time.time() - t1, 2)}
    return result


def describe_features(names: list[str]) -> list[dict]:
    return [{"feature": n, "label": describe(n)} for n in names]
