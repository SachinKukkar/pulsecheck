"""Pitfall experiment: how much does ordinary video compression hurt the pulse?

Re-encodes UBFC-rPPG's uncompressed videos with H.264 at several CRF values
(lower CRF = higher quality; 23 is x264's default, ~28+ is typical of social
media) and measures heart-rate error at each level.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .datasets import ubfc
from .extract import extract_dataset
from .hr_eval import evaluate_hr

CRFS = (18, 23, 28, 33, 38)


def _encode(ffmpeg: str, src: str, dst: Path, crf: int, max_s: float | None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += ["-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)


def compression_study(ubfc_root: str | Path, work_dir: str | Path, *, baseline_cache: str | Path | None = None,
                      crfs=CRFS, limit: int | None = None, max_s: float = 30.0, workers: int | None = None,
                      log=print) -> dict:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    work_dir = Path(work_dir)
    items = ubfc(ubfc_root)[:limit] if limit else ubfc(ubfc_root)
    rows = []

    def summarise(report, label, crf, kbps):
        get = lambda m, w: next((s for s in report["summary"] if s["method"] == m and s["window_s"] == w), {})  # noqa: E731
        rows.append({"label": label, "crf": crf, "kbps": kbps,
                     "mae_pos_10s": get("pos", 10).get("mae"), "mae_pos_30s": get("pos", 30).get("mae"),
                     "mae_green_10s": get("green", 10).get("mae"), "snr_pos_10s": get("pos", 10).get("snr_mean")})
        log(f"  {label:>12}: POS MAE {rows[-1]['mae_pos_10s'] or float('nan'):.2f} bpm (10 s)  at ~{kbps or 0:.0f} kbps")

    base_cache = Path(baseline_cache) if baseline_cache else work_dir / "uncompressed"
    extract_dataset(items, base_cache, max_s=max_s, workers=workers, log=log)
    summarise(evaluate_hr(base_cache, windows_s=(10, 30), log=lambda *_: None), "uncompressed", None, None)

    for crf in crfs:
        enc_items, sizes, durs = [], [], []
        for it in items:
            dst = work_dir / f"crf{crf}" / "videos" / f"{it.key}.mp4"
            if not dst.exists():
                _encode(ffmpeg, it.path, dst, crf, max_s)
            sizes.append(dst.stat().st_size)
            durs.append(max_s)
            enc_items.append(replace(it, path=str(dst)))
        cache = work_dir / f"crf{crf}" / "cache"
        extract_dataset(enc_items, cache, max_s=max_s, workers=workers, log=lambda *_: None)
        kbps = float(np.mean(np.array(sizes) * 8 / 1000 / np.array(durs)))
        summarise(evaluate_hr(cache, windows_s=(10, 30), log=lambda *_: None), f"CRF {crf}", crf, kbps)

    return {"kind": "compression_study", "source": "UBFC-rPPG re-encoded with H.264 (libx264)",
            "generated_at": time.strftime("%Y-%m-%d %H:%M"), "n_videos": len(items), "max_s": max_s, "rows": rows}
