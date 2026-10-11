"""Step 1a: read frames and when each one was captured.

Every beats-a-minute number depends on the frame clock, so this module is
careful about it. Files from phones and browsers (WebM from MediaRecorder) are
often variable-frame-rate and report a nonsense nominal FPS, so we prefer the
per-frame presentation timestamps and only fall back to index / FPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import FRAME_MAX_SIDE


class VideoError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: str
    fps: float  # nominal, as reported by the container (0 if unknown/implausible)
    frame_count: int  # nominal; unreliable for WebM
    width: int
    height: int


def _plausible_fps(fps: float) -> float:
    return float(fps) if 1.0 <= fps <= 240.0 else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"could not open video: {path}")
    try:
        return VideoInfo(
            path=str(path),
            fps=_plausible_fps(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def _resize(frame: np.ndarray, max_side: int | None) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def read_frames(
    path: str | Path,
    *,
    start_s: float = 0.0,
    max_s: float | None = None,
    max_side: int | None = FRAME_MAX_SIDE,
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Yield ``(t_container, t_index, rgb_frame)`` for each decoded frame.

    ``t_container`` is the decoder's presentation timestamp in seconds (NaN if
    the backend gave none); ``t_index`` is frame_index / nominal_fps. Callers
    pick one with :func:`choose_timestamps` after reading the whole clip.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoError(f"could not open video: {path}")
    fps = _plausible_fps(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    idx = -1
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            idx += 1
            t_pos = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_pos = t_pos / 1000.0 if t_pos and t_pos > 0 else float("nan")
            t_idx = idx / fps
            t_ref = t_pos if np.isfinite(t_pos) else t_idx
            if t_ref < start_s:
                continue
            if max_s is not None and t_ref - start_s >= max_s:
                break
            rgb = cv2.cvtColor(_resize(bgr, max_side), cv2.COLOR_BGR2RGB)
            yield t_pos, t_idx, rgb
    finally:
        cap.release()
    if idx < 0:
        raise VideoError(f"no frames could be decoded from: {path}")


def choose_timestamps(t_container: np.ndarray, t_index: np.ndarray) -> tuple[np.ndarray, str]:
    """Return the more trustworthy timestamp series and its name.

    Container timestamps win when they are all present, strictly increasing and
    imply a sane frame rate; otherwise use the index clock. The first frame is
    allowed to report 0 (some backends do), which we repair by extrapolation.
    """
    t = np.asarray(t_container, dtype=float).copy()
    if len(t) >= 3 and np.isnan(t[0]) and np.all(np.isfinite(t[1:])):
        t[0] = t[1] - (t[2] - t[1])
    if len(t) >= 3 and np.all(np.isfinite(t)):
        dt = np.diff(t)
        med = float(np.median(dt))
        if np.all(dt > 0) and 1 / 240 <= med <= 1.0:
            return t - t[0], "container"
    ti = np.asarray(t_index, dtype=float)
    return ti - (ti[0] if len(ti) else 0.0), "index"
