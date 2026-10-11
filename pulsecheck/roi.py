"""Step 2: cut skin patches (regions of interest) out of each frame.

Each patch is a polygon through MediaPipe landmark indices, chosen by plotting
the landmark numbers on a real face (``python -m pulsecheck landmarks``).
"left"/"right" are the subject's own left and right, so ``cheek_left`` appears
on the right-hand side of a non-mirrored image.

A per-video skin-colour model (built from the cheeks, the most reliably bare
skin) masks out hair, eyebrows and beard inside the polygons. It is learned
from each video's own pixels instead of fixed thresholds, so it does not
favour any particular skin tone.
"""

from __future__ import annotations

import cv2
import numpy as np

ROI_LANDMARKS: dict[str, list[int]] = {
    "forehead_left": [10, 338, 297, 333, 334, 296, 336, 9, 151],
    "forehead_right": [10, 151, 9, 107, 66, 105, 104, 67, 109],
    "cheek_left": [346, 347, 330, 266, 425, 411, 352],
    "cheek_right": [117, 118, 101, 36, 205, 187, 123],
    "nose": [168, 351, 419, 248, 281, 5, 51, 3, 196, 122],
    "chin": [18, 313, 418, 262, 428, 199, 208, 32, 194, 83],
}
ROI_NAMES: list[str] = list(ROI_LANDMARKS)

# Pull each polygon towards its centre so its edges stay off brows and hairline.
ROI_SHRINK: dict[str, float] = {
    "forehead_left": 0.88,
    "forehead_right": 0.88,
    "cheek_left": 0.92,
    "cheek_right": 0.92,
    "nose": 0.9,
    "chin": 0.9,
}

# Face outline (MediaPipe FACEMESH_FACE_OVAL order); used for overlays and size.
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
]

SKIN_MODEL_ROIS = ("cheek_left", "cheek_right")

# The PPG-map grid (FakeCatcher used 32 face regions): 32 small square patches centred
# on landmarks spread over bare skin. Rows are ordered so that neighbouring rows are
# neighbouring skin - forehead (image left to right), subject's right cheek, nose
# bridge, subject's left cheek, chin - which lets a CNN's convolutions see local structure.
GRID_LANDMARKS = [
    104, 69, 108, 151, 337, 299, 333, 9,  # forehead, then the glabella
    116, 118, 101, 123, 50, 36, 187, 205,  # right cheek: under-eye row, middle row, lower row
    6, 197, 195,  # nose bridge, top to bottom
    345, 347, 330, 352, 280, 266, 411, 425,  # left cheek (mirror of the right)
    32, 208, 200, 428, 262,  # chin, image left to right
]
GRID_PATCH = 0.14  # patch side as a fraction of the outer-eye-corner distance

# Categorical slots 1-6 of the site's chart palette (RGB), in ROI_NAMES order, so a
# patch has the same colour in the overlay image and in every chart.
ROI_COLORS = [(0x2A, 0x78, 0xD6), (0xEB, 0x68, 0x34), (0x1B, 0xAF, 0x7A), (0xED, 0xA1, 0x00), (0xE8, 0x7B, 0xA4), (0x00, 0x83, 0x00)]


def roi_polygons(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Map each ROI name to an (N, 2) float32 polygon in pixel coordinates."""
    polys = {}
    for name, idx in ROI_LANDMARKS.items():
        pts = landmarks[idx].astype(np.float32)
        c = pts.mean(axis=0)
        polys[name] = c + (pts - c) * ROI_SHRINK.get(name, 1.0)
    return polys


def face_size(landmarks: np.ndarray) -> float:
    """Face scale in pixels: distance between the outer eye corners."""
    return float(np.linalg.norm(landmarks[33] - landmarks[263]))


class SkinModel:
    """Gaussian model of skin chrominance (Cr, Cb) learned from one video.

    Luma is ignored so the model survives brightness changes; pixels whose
    squared Mahalanobis distance exceeds ``chi2`` are treated as non-skin.
    The default 13.8 keeps 99.9% of a 2-D Gaussian, so it only removes pixels
    that are clearly not skin (hair, brows, glasses frames, background).
    """

    def __init__(self, chi2: float = 13.8):
        self.chi2 = chi2
        self._samples: list[np.ndarray] = []
        self.mean: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None

    @property
    def ready(self) -> bool:
        return self.mean is not None

    def add(self, frame_rgb: np.ndarray, polys: dict[str, np.ndarray]) -> None:
        for name in SKIN_MODEL_ROIS:
            sub, mask, _ = _crop_mask(frame_rgb, polys[name])
            if sub is None:
                continue
            crcb = cv2.cvtColor(sub, cv2.COLOR_RGB2YCrCb)[..., 1:][mask]
            if len(crcb):
                self._samples.append(crcb.astype(np.float32))

    def fit(self) -> bool:
        if not self._samples:
            return False
        x = np.concatenate(self._samples)
        if len(x) < 50:
            return False
        # Trim the 5% most extreme samples so stray hair in a cheek can't widen the model.
        mu = np.median(x, axis=0)
        d = np.linalg.norm(x - mu, axis=1)
        x = x[d <= np.quantile(d, 0.95)]
        cov = np.cov(x.T) + np.eye(2) * 4.0  # floor: never narrower than ~2 levels
        self.mean = x.mean(axis=0)
        self.inv_cov = np.linalg.inv(cov)
        self._samples.clear()
        return True

    def mask(self, sub_rgb: np.ndarray) -> np.ndarray:
        crcb = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32) - self.mean
        d2 = np.einsum("...i,ij,...j->...", crcb, self.inv_cov, crcb)
        return d2 <= self.chi2


def _crop_mask(frame: np.ndarray, poly: np.ndarray):
    """Return (crop, polygon mask, origin) for the polygon's bounding box."""
    h, w = frame.shape[:2]
    x0, y0 = np.floor(poly.min(axis=0)).astype(int)
    x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, None, None
    sub = frame[y0:y1, x0:x1]
    m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    pts = np.round((poly - (x0, y0)) * 4).astype(np.int32)  # 2 fractional bits for sub-pixel edges
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=2)
    return sub, m.astype(bool), (x0, y0)


def roi_means(
    frame_rgb: np.ndarray,
    polys: dict[str, np.ndarray],
    skin: SkinModel | None = None,
    min_skin_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean R, G, B of each ROI and the number of pixels averaged.

    Returns ``means`` of shape (n_roi, 3) (NaN where a patch is unusable) and
    ``counts`` of shape (n_roi,). If the skin mask would leave less than
    ``min_skin_fraction`` of a patch, the whole patch is used instead: a dark,
    oddly lit or heavily made-up face should degrade gracefully, not vanish.
    """
    n = len(polys)
    means = np.full((n, 3), np.nan, dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for i, poly in enumerate(polys.values()):
        sub, m, _ = _crop_mask(frame_rgb, poly)
        if sub is None:
            continue
        total = int(m.sum())
        if total < 4:
            continue
        if skin is not None and skin.ready:
            ms = m & skin.mask(sub)
            if ms.sum() >= max(4, min_skin_fraction * total):
                m = ms
        px = sub[m]
        means[i] = px.mean(axis=0)
        counts[i] = len(px)
    return means, counts


def grid_boxes(landmarks: np.ndarray, frame_shape) -> list[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) of each grid patch, clipped to the frame."""
    h, w = frame_shape[:2]
    half = max(1, int(round(GRID_PATCH * face_size(landmarks) / 2)))
    boxes = []
    for idx in GRID_LANDMARKS:
        x, y = np.round(landmarks[idx]).astype(int)
        boxes.append((max(x - half, 0), max(y - half, 0), min(x + half + 1, w), min(y + half + 1, h)))
    return boxes


def grid_means(frame_rgb: np.ndarray, landmarks: np.ndarray, skin: SkinModel | None = None,
               min_skin_fraction: float = 0.25) -> np.ndarray:
    """Mean R, G, B of each of the 32 grid patches, shape (32, 3); NaN where a patch is unusable."""
    out = np.full((len(GRID_LANDMARKS), 3), np.nan)
    use_skin = skin is not None and skin.ready
    for j, (x0, y0, x1, y1) in enumerate(grid_boxes(landmarks, frame_rgb.shape)):
        sub = frame_rgb[y0:y1, x0:x1]
        if sub.shape[0] < 2 or sub.shape[1] < 2:
            continue
        px = sub.reshape(-1, 3)
        if use_skin:
            m = skin.mask(sub).reshape(-1)
            if m.sum() >= max(4, min_skin_fraction * len(px)):
                px = px[m]
        out[j] = px.mean(axis=0)
    return out


def draw_overlay(frame_rgb: np.ndarray, landmarks: np.ndarray, skin: SkinModel | None = None) -> np.ndarray:
    """Frame with the face outline, the six patches and (optionally) masked-out pixels."""
    out = frame_rgb.copy()
    polys = roi_polygons(landmarks)
    if skin is not None and skin.ready:
        for poly in polys.values():
            sub, m, (x0, y0) = _crop_mask(frame_rgb, poly)
            if sub is None:
                continue
            reject = m & ~skin.mask(sub)
            region = out[y0 : y0 + sub.shape[0], x0 : x0 + sub.shape[1]]
            region[reject] = (region[reject] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
    scale = max(1, round(max(frame_rgb.shape[:2]) / 400))
    oval = np.round(landmarks[FACE_OVAL]).astype(np.int32)
    cv2.polylines(out, [oval], True, (255, 255, 255), scale, cv2.LINE_AA)
    for (name, poly), col in zip(polys.items(), ROI_COLORS):
        cv2.polylines(out, [np.round(poly).astype(np.int32)], True, col, scale + 1, cv2.LINE_AA)
    return out


def draw_landmark_numbers(frame_rgb: np.ndarray, landmarks: np.ndarray, scale: int = 6) -> np.ndarray:
    """Upscaled crop of the face with every landmark index written next to it."""
    x0, y0 = np.floor(landmarks.min(axis=0) - 12).astype(int).clip(0)
    x1, y1 = np.ceil(landmarks.max(axis=0) + 12).astype(int)
    x1, y1 = min(x1, frame_rgb.shape[1]), min(y1, frame_rgb.shape[0])
    crop = frame_rgb[y0:y1, x0:x1]
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = (big * 0.55 + 255 * 0.45).astype(np.uint8)
    for i, (x, y) in enumerate(landmarks[:468]):
        p = (int((x - x0) * scale), int((y - y0) * scale))
        cv2.circle(big, p, 3, (220, 0, 0), -1)
        cv2.putText(big, str(i), (p[0] + 3, p[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return big
