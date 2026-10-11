"""Steps 1-2 end to end: video file -> per-patch colour traces on a uniform clock."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import FRAME_MAX_SIDE, MIN_FACE_FRACTION, TARGET_FS
from .face import FaceLandmarker
from .roi import GRID_LANDMARKS, ROI_NAMES, SkinModel, draw_overlay, face_size, grid_means, roi_means, roi_polygons
from .signals import fill_nans, resample_uniform
from .traces import Traces
from .video import choose_timestamps, probe, read_frames

SKIN_WARMUP_FRAMES = 10

Progress = Callable[[str, float], None]


class NoFaceError(RuntimeError):
    pass


@dataclass
class Preview:
    frame: np.ndarray  # RGB uint8
    landmarks: np.ndarray  # (478, 2) in that frame's pixels
    overlay: np.ndarray  # frame with patches drawn


def extract_traces(
    path: str | Path,
    *,
    start_s: float = 0.0,
    max_s: float | None = None,
    fs: float = TARGET_FS,
    skin_mask: bool = True,
    max_side: int | None = FRAME_MAX_SIDE,
    landmarker: FaceLandmarker | None = None,
    want_preview: bool = False,
    progress: Progress | None = None,
) -> tuple[Traces, Preview | None]:
    info = probe(path)
    expected_s = max_s or (info.frame_count / info.fps if info.fps and info.frame_count else None)
    own = landmarker is None
    lm = landmarker or FaceLandmarker(video=True)
    skin = SkinModel() if skin_mask else None

    t_c: list[float] = []
    t_i: list[float] = []
    means: list[np.ndarray | None] = []
    counts: list[np.ndarray | None] = []
    grids: list[np.ndarray | None] = []
    centers: list[np.ndarray] = []
    sizes: list[float] = []
    warmup: list[tuple[int, np.ndarray, dict, np.ndarray]] = []
    preview: Preview | None = None
    preview_locked = False
    frame_shape = None
    nan_row = np.full((len(ROI_NAMES), 3), np.nan)
    nan_grid = np.full((len(GRID_LANDMARKS), 3), np.nan)

    def flush_warmup():
        for k, rgb_k, polys_k, lmk_k in warmup:
            means[k], counts[k] = roi_means(rgb_k, polys_k, skin)
            grids[k] = grid_means(rgb_k, lmk_k, skin)
        warmup.clear()

    try:
        for tc, ti, rgb in read_frames(path, start_s=start_s, max_s=max_s, max_side=max_side):
            frame_shape = rgb.shape
            t_c.append(tc)
            t_i.append(ti)
            lmk = lm.detect(rgb, int(round(ti * 1000)))
            if lmk is None:
                means.append(None)
                counts.append(None)
                grids.append(None)
                centers.append(np.array([np.nan, np.nan]))
                sizes.append(np.nan)
            else:
                polys = roi_polygons(lmk)
                centers.append(lmk.mean(axis=0))
                sizes.append(face_size(lmk))
                # Preview frame: the first face at least 1 s in (the opening frames are often blurred).
                if want_preview and (preview is None or (not preview_locked and ti - start_s >= 1.0)):
                    preview = Preview(frame=rgb.copy(), landmarks=lmk.copy(), overlay=None)
                    preview_locked = ti - start_s >= 1.0
                if skin is not None and not skin.ready:
                    skin.add(rgb, polys)
                    warmup.append((len(means), rgb, polys, lmk))
                    means.append(None)
                    counts.append(None)
                    grids.append(None)
                    if len(warmup) >= SKIN_WARMUP_FRAMES:
                        skin.fit()
                        flush_warmup()
                else:
                    m, c = roi_means(rgb, polys, skin)
                    means.append(m)
                    counts.append(c)
                    grids.append(grid_means(rgb, lmk, skin))
            if progress and len(t_c) % 15 == 0 and expected_s:
                progress("Finding the face and averaging skin patches", min(ti / expected_s, 1.0))
        if warmup:
            skin.fit()
            flush_warmup()
    finally:
        if own:
            lm.close()

    n = len(t_c)
    have = np.array([m is not None for m in means])
    face_fraction = float(have.mean()) if n else 0.0
    if n == 0 or face_fraction < MIN_FACE_FRACTION:
        raise NoFaceError(f"a face was found in only {face_fraction:.0%} of {n} frames")

    first, last = np.flatnonzero(have)[[0, -1]]
    keep = slice(first, last + 1)
    M = np.stack([m if m is not None else nan_row for m in means])[keep]
    C = np.stack([c if c is not None else np.zeros(len(ROI_NAMES)) for c in counts])[keep].astype(float)
    t_all, clock = choose_timestamps(np.array(t_c), np.array(t_i))
    t = t_all[keep]
    # Seconds from the first decoded frame to trace t=0, so ground truth can be aligned.
    t_offset = float(t[0]) + start_s
    t = t - t[0]
    if t[-1] < 2.0:
        raise NoFaceError(f"only {t[-1]:.1f} s of video with a face; need at least a few seconds")

    tu, Mu = resample_uniform(t, M, fs)
    C[~have[keep]] = np.nan
    _, Cu = resample_uniform(t, C, fs)
    G = np.stack([g if g is not None else nan_grid for g in grids])[keep]
    _, Gu = resample_uniform(t, G, fs)

    ctr = fill_nans(np.stack(centers)[keep])
    size = fill_nans(np.array(sizes)[keep])
    speed = np.r_[0.0, np.linalg.norm(np.diff(ctr, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
    _, motion = resample_uniform(t, speed / np.maximum(size, 1.0), fs)

    cheek = np.nanmedian(M[:, [ROI_NAMES.index("cheek_left"), ROI_NAMES.index("cheek_right")]], axis=(0, 1))
    meta = {
        "source": str(path),
        "nominal_fps": info.fps,
        "clock": clock,
        "frames_decoded": n,
        "frame_size": [int(frame_shape[1]), int(frame_shape[0])] if frame_shape else None,
        "original_size": [info.width, info.height],
        "face_fraction": face_fraction,
        "face_px": float(np.nanmedian(np.array(sizes, dtype=float))),
        "effective_fps": float((n - 1) / max(t[-1], 1e-6)) if n > 1 else 0.0,
        "skin_rgb": [float(v) for v in cheek],
        "skin_mask": bool(skin is not None and skin.ready),
        "start_s": start_s,
        "t_offset_s": t_offset,
    }
    traces = Traces(
        t=tu, fs=fs, rgb=Mu, counts=np.nan_to_num(Cu).round().astype(int),
        roi_names=list(ROI_NAMES), motion=motion, meta=meta, grid_rgb=Gu,
    )
    if preview is not None:
        preview.overlay = draw_overlay(preview.frame, preview.landmarks, skin)
    if progress:
        progress("Finding the face and averaging skin patches", 1.0)
    return traces, preview


def skin_ita(skin_rgb: list[float] | np.ndarray) -> float:
    """Individual Typology Angle (degrees) of a mean skin colour.

    ITA = arctan((L* - 50) / b*) in CIELAB. Higher is lighter: > 55 very light,
    41-55 light, 28-41 intermediate, 10-28 tan, -30-10 brown, < -30 dark. It is
    computed from video pixels, so lighting and white balance shift it: use it
    to spot gaps in a dataset, not to label people.
    """
    rgb = np.clip(np.asarray(skin_rgb, dtype=np.float32) / 255.0, 0, 1).reshape(1, 1, 3)
    L, _, b = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)[0, 0]
    return float(np.degrees(np.arctan2(L - 50.0, b)))


ITA_BINS = [(-90, -30, "dark"), (-30, 10, "brown"), (10, 28, "tan"), (28, 41, "intermediate"), (41, 55, "light"), (55, 90, "very light")]


def ita_category(ita: float) -> str:
    for lo, hi, name in ITA_BINS:
        if lo <= ita < hi:
            return name
    return "unknown"
