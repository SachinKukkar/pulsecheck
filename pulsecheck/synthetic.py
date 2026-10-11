"""Synthetic test clips with a known pulse painted onto a still face.

Used to test the whole pipeline end to end before any dataset arrives, and as
the sample clip on the website. Nothing here is evidence that the detector
works on real deepfakes: the "fake" clips are a cartoon of one hypothesis
(the swapped inner face does not share the forehead's pulse).

Model of a frame:  pixel = base * (1 - depth * pulse(t) * PBV * skin) * light(t) + noise,
then shifted by a small head movement. PBV is how strongly R, G and B respond
to blood (green most).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .face import FaceLandmarker
from .roi import FACE_OVAL, SkinModel, roi_polygons

PBV = np.array([0.33, 0.77, 0.53], dtype=np.float32) / 0.77

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
LEFT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
# Roughly what a face-swap replaces: brows down to the chin, inside the jaw.
INNER_FACE = [70, 63, 105, 66, 107, 9, 336, 296, 334, 293, 300, 383, 372, 345, 352, 376, 433, 416,
              434, 430, 431, 262, 428, 199, 208, 32, 211, 210, 214, 192, 213, 147, 123, 116, 143, 156]

KINDS = ("real", "fake_mismatch", "fake_flat", "fake_desync")


@dataclass
class SynthParams:
    kind: str = "real"
    seconds: float = 15.0
    fps: float = 30.0
    bpm: float = 72.0
    hrv: float = 0.04  # +/- fractional heart-rate variation (breathing)
    depth: float = 0.006  # pulse depth as a fraction of skin colour
    noise: float = 1.5  # sensor noise, grey levels (std)
    motion_px: float = 1.5  # head sway amplitude, pixels
    light_drift: float = 0.03  # slow brightness drift, fraction
    seed: int = 0


def _hull_mask(shape, pts, dilate=0) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts).astype(np.int32)), 1)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m.astype(np.float32)


def face_masks(image: np.ndarray, lmk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (H, W) in [0, 1]: (outer skin = forehead and edges, inner swap region)."""
    face = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(face, [np.round(lmk[FACE_OVAL]).astype(np.int32)], 1)
    face = face.astype(np.float32)
    size = np.linalg.norm(lmk[33] - lmk[263])
    d = max(3, int(size * 0.08)) | 1
    for part in (RIGHT_EYE, LEFT_EYE, RIGHT_BROW, LEFT_BROW, LIPS):
        face *= 1 - _hull_mask(image.shape, lmk[part], dilate=d)
    skin = SkinModel(chi2=9.2)
    skin.add(image, roi_polygons(lmk))
    if skin.fit():
        face *= skin.mask(image).astype(np.float32)
    inner = _hull_mask(image.shape, lmk[INNER_FACE]) * face
    outer = face - inner
    k = max(3, int(size * 0.15)) | 1
    return cv2.GaussianBlur(outer, (k, k), 0), cv2.GaussianBlur(inner, (k, k), 0)


def pulse_wave(t: np.ndarray, bpm: float, hrv: float, rng: np.random.Generator, phase: float = 0.0):
    """PPG-like wave (fundamental + 2 harmonics) with breathing-driven rate changes."""
    breath = 0.2 + 0.1 * rng.random()
    f = bpm / 60 * (1 + hrv * np.sin(2 * np.pi * breath * t + rng.uniform(0, 2 * np.pi)))
    phi = 2 * np.pi * np.cumsum(f) * (t[1] - t[0]) + phase
    wave = np.sin(phi) + 0.35 * np.sin(2 * phi - 0.6) + 0.1 * np.sin(3 * phi - 1.2)
    return wave / wave.std(), 60 * f


def render(image: np.ndarray, lmk: np.ndarray, p: SynthParams) -> tuple[list[np.ndarray], dict]:
    rng = np.random.default_rng(p.seed)
    n = int(round(p.seconds * p.fps))
    t = np.arange(n) / p.fps
    outer, inner = face_masks(image, lmk)
    p_out, bpm_out = pulse_wave(t, p.bpm, p.hrv, rng)

    flicker = np.zeros(n)
    if p.kind == "real":
        # Same heartbeat everywhere; the cheeks lag the forehead by ~20 ms (pulse transit).
        p_in, _ = pulse_wave(t - 0.02, p.bpm, p.hrv, np.random.default_rng(p.seed))
    elif p.kind == "fake_mismatch":  # the swapped face carries a different person's rate
        p_in, _ = pulse_wave(t, p.bpm * rng.choice([0.75, 1.3]), p.hrv, rng, rng.uniform(0, 2 * np.pi))
    elif p.kind == "fake_flat":  # no pulse in the generated face, only frame-to-frame flicker
        p_in = np.zeros(n)
        flicker = rng.standard_normal(n) * 0.004
    elif p.kind == "fake_desync":  # same rate, but its own drifting phase and variability
        p_in, _ = pulse_wave(t, p.bpm, p.hrv * 3, rng, rng.uniform(0.5, 2.5))
    else:
        raise ValueError(f"unknown kind {p.kind!r}")

    light = 1 + p.light_drift * np.sin(2 * np.pi * rng.uniform(0.03, 0.08) * t + rng.uniform(0, 6.3))
    sway = rng.uniform(0.15, 0.35)
    dx = p.motion_px * np.sin(2 * np.pi * sway * t + rng.uniform(0, 6.3))
    dy = 0.6 * p.motion_px * np.sin(2 * np.pi * sway * 0.7 * t + rng.uniform(0, 6.3))

    base = image.astype(np.float32)
    h, w = image.shape[:2]
    ys, xs = np.nonzero((outer + inner) > 1e-3)
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))  # only the face is modulated
    outer_b, inner_b = outer[box][..., None], inner[box][..., None]

    def frames() -> Iterator[np.ndarray]:  # lazily, so a clip never sits in memory
        for k in range(n):
            frame = base * light[k]
            blood = (p_out[k] * outer_b + p_in[k] * inner_b) * PBV
            frame[box] *= 1 - p.depth * blood
            if flicker[k]:
                frame[box] *= 1 + flicker[k] * inner_b
            frame += rng.standard_normal(frame.shape, dtype=np.float32) * np.float32(p.noise)
            M = np.float32([[1, 0, dx[k]], [0, 1, dy[k]]])
            frame = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            yield np.clip(frame, 0, 255).round().astype(np.uint8)

    truth = {"params": asdict(p), "mean_bpm": float(np.mean(bpm_out)), "label": int(p.kind != "real"),
             "ppg": p_out, "bpm_trace": bpm_out, "time": t}  # arrays: popped before saving labels.json
    return frames(), truth


def write_video(path: str | Path, frames: Iterable[np.ndarray], fps: float, crf: int | None = None) -> None:
    """Write RGB frames as H.264 MP4. ``crf=None`` is lossless (4:4:4); a number is a normal lossy encode."""
    import imageio_ffmpeg

    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[:2]
    if crf is None:
        pix, params = "yuv444p", ["-qp", "0", "-preset", "veryfast"]
    else:
        pix, params = "yuv420p", ["-crf", str(crf), "-preset", "medium", "-movflags", "+faststart"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out=pix,
        quality=None, macro_block_size=2, output_params=params, ffmpeg_log_level="error",
    )
    writer.send(None)
    writer.send(np.ascontiguousarray(first))
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()


def load_face_image(image_path: str | Path | None = None) -> np.ndarray:
    if image_path:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise FileNotFoundError(image_path)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from skimage import data  # dev dependency; NASA portrait, public domain

    return data.astronaut()


def make_clip(out_path: str | Path, params: SynthParams, image: np.ndarray | None = None, crf: int | None = None) -> dict:
    image = load_face_image() if image is None else image
    with FaceLandmarker(video=False) as lm:
        lmk = lm.detect(image)
    if lmk is None:
        raise RuntimeError("no face found in the base image")
    frames, truth = render(image, lmk, params)
    write_video(out_path, frames, params.fps, crf)
    for k in ("ppg", "bpm_trace", "time"):
        truth.pop(k)
    return truth


CRF_MIX = (None, 12, 17, 22)  # lossless, then progressively lossier H.264


def _render_pair(job) -> list[dict]:
    out_dir, image, lmk, i, common, fake_kind, seed, crf = job
    items = []
    for kind in ("real", fake_kind):
        p = SynthParams(kind=kind, seed=seed * 1000 + i * 2 + (kind != "real"), **common)
        frames, truth = render(image, lmk, p)
        name = f"{i:03d}_{kind}.mp4"
        write_video(out_dir / name, frames, p.fps, crf)
        gt = None
        if kind == "real":  # UBFC-style ground truth, so the heart-rate evaluation runs unchanged
            gt = f"{i:03d}_{kind}_gt.txt"
            np.savetxt(out_dir / gt, np.vstack([truth["ppg"], truth["bpm_trace"], truth["time"]]))
        for k in ("ppg", "bpm_trace", "time"):
            truth.pop(k)
        items.append({"file": name, "group": f"pair{i:03d}", "gt": gt, "crf": crf, **truth})
    return items


def make_dataset(out_dir: str | Path, n_pairs: int = 20, seconds: float = 15.0, seed: int = 0,
                 image_path: str | Path | None = None, crf="mixed", workers: int | None = None) -> Path:
    """Paired real / fake-like clips + labels.json. Each pair shares a group so it can't leak across folds.

    With ``crf="mixed"`` each pair is encoded at a random quality from CRF_MIX, the same for its real
    and fake clip, so a classifier cannot win by telling compression levels apart.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image = load_face_image(image_path)
    with FaceLandmarker(video=False) as lm:
        lmk = lm.detect(image)
    rng = np.random.default_rng(seed)
    jobs = []
    for i in range(n_pairs):
        common = dict(
            seconds=seconds, bpm=float(rng.uniform(55, 110)), noise=float(rng.uniform(0.8, 3.0)),
            motion_px=float(rng.uniform(0.0, 3.0)), light_drift=float(rng.uniform(0.0, 0.05)),
            depth=float(rng.uniform(0.004, 0.009)),
        )
        pair_crf = CRF_MIX[int(rng.integers(len(CRF_MIX)))] if crf == "mixed" else crf
        jobs.append((out_dir, image, lmk, i, common, KINDS[1 + i % 3], seed, pair_crf))
    with ProcessPoolExecutor(max_workers=workers or max(1, min(4, (os.cpu_count() or 2) // 3))) as pool:
        items = [it for pair in pool.map(_render_pair, jobs) for it in pair]
    (out_dir / "labels.json").write_text(json.dumps(items, indent=1))
    return out_dir
