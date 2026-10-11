"""The cached product of the expensive step: mean colour of each patch per frame.

Landmarking a video is slow (tens of milliseconds a frame); everything after it
is fast. So datasets are processed once into ``Traces`` saved as ``.npz`` and
every experiment (methods, features, classifiers, the CNN) re-reads those.

Two sets of patches are stored:
  * ``rgb``      - the six named regions (forehead halves, cheeks, nose, chin)
                   used for heart rate and the hand-made features;
  * ``grid_rgb`` - 32 small landmark-centred patches, the rows of the
                   "PPG map" the CNN learns from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Traces:
    t: np.ndarray  # (T,) seconds on a uniform clock, starting at 0
    fs: float  # samples per second of that clock
    rgb: np.ndarray  # (T, R, 3) mean R, G, B of each named patch
    counts: np.ndarray  # (T, R) pixels averaged per patch (skin-masked)
    roi_names: list[str]
    motion: np.ndarray  # (T,) face speed in face-widths per second
    meta: dict = field(default_factory=dict)
    grid_rgb: np.ndarray | None = None  # (T, 32, 3) mean R, G, B of each grid patch

    @property
    def duration(self) -> float:
        return len(self.t) / self.fs

    def roi(self, name: str) -> np.ndarray:
        return self.rgb[:, self.roi_names.index(name)]

    def pooled(self, names: list[str] | None = None) -> np.ndarray:
        """Pixel-weighted mean colour over several patches, shape (T, 3)."""
        idx = [self.roi_names.index(n) for n in names] if names else list(range(len(self.roi_names)))
        w = self.counts[:, idx].astype(float)
        w_sum = w.sum(axis=1, keepdims=True)
        w = np.where(w_sum > 0, w / np.where(w_sum > 0, w_sum, 1), 1.0 / len(idx))
        return np.einsum("tr,trc->tc", w, self.rgb[:, idx])

    def slice(self, sl: slice) -> "Traces":
        return Traces(
            t=self.t[sl] - self.t[sl][0],
            fs=self.fs,
            rgb=self.rgb[sl],
            counts=self.counts[sl],
            roi_names=list(self.roi_names),
            motion=self.motion[sl],
            meta=dict(self.meta),
            grid_rgb=None if self.grid_rgb is None else self.grid_rgb[sl],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        arrays = dict(
            t=self.t.astype(np.float64),
            fs=np.float64(self.fs),
            rgb=self.rgb.astype(np.float32),
            counts=self.counts.astype(np.int32),
            roi_names=np.array(self.roi_names),
            motion=self.motion.astype(np.float32),
            meta=np.array(json.dumps(self.meta)),
        )
        if self.grid_rgb is not None:
            arrays["grid_rgb"] = self.grid_rgb.astype(np.float32)
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)  # atomic: an interrupted run never leaves a half-written cache file

    @classmethod
    def load(cls, path: str | Path) -> "Traces":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                t=z["t"],
                fs=float(z["fs"]),
                rgb=z["rgb"].astype(np.float64),
                counts=z["counts"],
                roi_names=[str(s) for s in z["roi_names"]],
                motion=z["motion"].astype(np.float64),
                meta=json.loads(str(z["meta"])),
                grid_rgb=z["grid_rgb"].astype(np.float64) if "grid_rgb" in z else None,
            )
