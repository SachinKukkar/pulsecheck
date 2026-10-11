"""Step 1b: find the face.

MediaPipe Face Landmarker returns 478 points per face (468 mesh points plus
10 iris points). We only need their pixel positions to cut out skin patches.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from .config import MODELS_DIR

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: Path = MODEL_PATH, download: bool = True) -> Path:
    """Return the landmarker model path, downloading and checking it if missing."""
    path = Path(path)
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing; run `python -m pulsecheck download-model`")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    digest = _sha256(tmp)
    if digest != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {MODEL_URL}: got {digest}")
    tmp.replace(path)
    return path


class FaceLandmarker:
    """Thin wrapper: RGB frame in, (478, 2) float32 pixel coordinates out.

    ``video=True`` uses MediaPipe's VIDEO mode, which tracks the face between
    frames (faster and steadier) and needs increasing timestamps.
    """

    def __init__(self, model_path: Path | None = None, *, video: bool = True, min_confidence: float = 0.5):
        from mediapipe.tasks.python import BaseOptions, vision

        self._video = video
        self._last_ts = -1
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or MODEL_PATH))),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray | None:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            ts = self._last_ts + 1 if timestamp_ms is None else int(timestamp_ms)
            ts = max(ts, self._last_ts + 1)  # MediaPipe rejects non-increasing timestamps
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = rgb.shape[:2]
        pts = result.face_landmarks[0]
        return np.array([(p.x * w, p.y * h) for p in pts], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
