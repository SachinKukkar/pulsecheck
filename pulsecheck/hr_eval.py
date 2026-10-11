"""Step 4 check: heart rate against a real pulse (UBFC-rPPG's finger oximeter).

Ground-truth heart rate for a window is read from the oximeter's PPG waveform
with exactly the same spectral method, so both sides get the same frequency
resolution. Error is reported per (method, window length), which also answers
"how short is too short?" (a 10 s window only resolves 6 bpm steps).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import HR_BAND
from .datasets import load_ubfc_gt
from .extract import load_index
from .metrics import hr_metrics
from .pipeline import ita_category, skin_ita
from .rppg import METHODS, pulse
from .signals import bandpass, detrend, peak_frequency, power_spectrum, snr_db
from .traces import Traces

WINDOWS_S = (5, 10, 20, 30)
HEADLINE = ("pos", 10)


def _downsample(x: np.ndarray, n: int = 400) -> list[float]:
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int)
    return [round(float(v), 4) for v in x[idx]]


def gt_pulse_on_clock(tr: Traces, gt_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Oximeter PPG resampled onto the trace clock, band-passed like the video pulse; and a validity mask."""
    gt_t, gt_ppg = load_ubfc_gt(gt_path)
    tq = tr.t + tr.meta.get("t_offset_s", 0.0)
    valid = (tq >= gt_t[0]) & (tq <= gt_t[-1])
    g = np.interp(tq, gt_t, gt_ppg)
    return bandpass(detrend(g), tr.fs, *HR_BAND), valid


def evaluate_hr(cache_dir: str | Path, methods=METHODS, windows_s=WINDOWS_S, headline=HEADLINE,
                source: str = "UBFC-rPPG", log=print) -> dict:
    cache_dir = Path(cache_dir)
    items = [d for d in load_index(cache_dir) if d.get("gt")]
    if not items:
        raise RuntimeError(f"no cached videos with ground truth in {cache_dir}")
    rows, subjects, example = [], [], None
    for it in items:
        tr = Traces.load(cache_dir / f"{it['key']}.npz")
        fs = tr.fs
        gt, valid = gt_pulse_on_clock(tr, it["gt"])
        vid = {m: pulse(tr.pooled(), fs, m) for m in methods}
        ita = skin_ita(tr.meta["skin_rgb"])
        subjects.append({"subject": it["key"].removeprefix("ubfc_"), "ita": round(ita, 1),
                         "skin_category": ita_category(ita), "duration_s": round(tr.duration, 1),
                         "motion": round(float(np.median(tr.motion)), 4)})
        for W in windows_s:
            n = int(round(W * fs))
            for a in range(0, len(tr.t) - n + 1, n):  # non-overlapping full windows
                sl = slice(a, a + n)
                if not valid[sl].all():
                    continue
                f, p = power_spectrum(gt[sl], fs)
                f_gt = peak_frequency(f, p, HR_BAND)
                for m in methods:
                    fv, pv = power_spectrum(vid[m][sl], fs)
                    rows.append({"subject": subjects[-1]["subject"], "method": m, "window_s": W, "t0": round(a / fs, 2),
                                 "est": 60 * peak_frequency(fv, pv, HR_BAND), "gt": 60 * f_gt,
                                 "snr": snr_db(fv, pv, f_gt), "ita": ita})
        if example is None and len(tr.t) > 10 * fs:
            sl = slice(int(5 * fs), int(15 * fs))
            z = lambda x: (x - x.mean()) / (x.std() + 1e-12)  # noqa: E731
            example = {"subject": subjects[-1]["subject"], "t": _downsample(np.arange(sl.stop - sl.start) / fs),
                       "oximeter": _downsample(z(gt[sl])), "video_pos": _downsample(z(vid["pos"][sl])),
                       "video_green": _downsample(z(vid["green"][sl])) if "green" in vid else None}

    R = {k: np.array([r[k] for r in rows]) for k in ("method", "window_s", "est", "gt", "snr")}
    subj = np.array([r["subject"] for r in rows])
    summary = []
    for m in methods:
        for W in windows_s:
            sel = (R["method"] == m) & (R["window_s"] == W)
            if sel.any():
                summary.append({"method": m, "window_s": W, **hr_metrics(R["est"][sel], R["gt"][sel]),
                                "snr_mean": float(np.nanmean(R["snr"][sel]))})
    hm, hw = headline
    hsel = (R["method"] == hm) & (R["window_s"] == hw)
    per_subject = []
    for s in subjects:
        sel = hsel & (subj == s["subject"])
        if sel.any():
            per_subject.append({**s, "mae": float(np.mean(np.abs(R["est"][sel] - R["gt"][sel]))), "n": int(sel.sum())})
    by_skin = []
    for cat in ("very light", "light", "intermediate", "tan", "brown", "dark"):
        maes = [p["mae"] for p in per_subject if p["skin_category"] == cat]
        by_skin.append({"category": cat, "n_subjects": len(maes), "mae": float(np.mean(maes)) if maes else None})

    est_h, gt_h = R["est"][hsel], R["gt"][hsel]
    report = {
        "kind": "hr_validation", "source": source, "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_videos": len(items), "methods": list(methods), "windows_s": list(windows_s),
        "headline": {"method": hm, "window_s": hw, **hr_metrics(est_h, gt_h)},
        "summary": summary,
        "scatter": {"gt": [round(float(v), 2) for v in gt_h], "est": [round(float(v), 2) for v in est_h],
                    "subject": [str(s) for s in subj[hsel]]},
        "per_subject": sorted(per_subject, key=lambda p: p["mae"]),
        "by_skin": by_skin,
        "example": example,
    }
    log(f"heart rate on {len(items)} videos ({source}):")
    for s in summary:
        if s["window_s"] == hw:
            log(f"  {s['method']:5s} {hw:>2} s windows: MAE {s['mae']:.2f} bpm  RMSE {s['rmse']:.2f}  r {s['pearson_r'] or 0:.2f}  n={s['n']}")
    return report
