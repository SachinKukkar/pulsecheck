"""Dataset loaders: turn a folder on disk into a list of labelled videos.

Every loader returns ``VideoItem``s with a ``group``: all videos that share a
person or a source video get the same group, so they always land on the same
side of a train/test split (step 7 of the guide).

Expected layouts (what the official downloads unpack to):

UBFC-rPPG    <root>/DATASET_2/subject1/vid.avi + ground_truth.txt
             <root>/DATASET_1/<name>/vid.avi + gtdump.xmp   (optional)
FF++         <root>/original_sequences/youtube/c23/videos/000.mp4
             <root>/manipulated_sequences/Deepfakes/c23/videos/000_003.mp4  (target_source)
Celeb-DF v2  <root>/Celeb-real/id0_0000.mp4, <root>/YouTube-real/00000.mp4,
             <root>/Celeb-synthesis/id0_id1_0000.mp4, <root>/List_of_testing_videos.txt
synthetic    <root>/*.mp4 + labels.json (made by ``python -m pulsecheck synth-dataset``)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import SPLITS_DIR

FFPP_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


@dataclass
class VideoItem:
    key: str  # unique and filesystem-safe; names the cache file
    path: str
    dataset: str
    label: int  # 0 = real, 1 = fake
    group: str  # leakage group (person / source video)
    split: str | None  # "train" | "val" | "test" | None
    method: str  # "original", a manipulation name, "Celeb-synthesis", "real", ...
    gt: str | None = None  # ground-truth pulse file (UBFC)

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")


def hash_split(group: str, val: float = 0.15, test: float = 0.15) -> str:
    """Deterministic split by group, for datasets without an official one."""
    u = int(hashlib.md5(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "val" if u < test + val else "train"


# --------------------------------------------------------------------------- UBFC-rPPG

def ubfc(root: str | Path) -> list[VideoItem]:
    root = Path(root)
    items = []
    for vid in sorted(root.rglob("vid.avi")):
        d = vid.parent
        gt = next((d / n for n in ("ground_truth.txt", "gtdump.xmp") if (d / n).exists()), None)
        rel = d.relative_to(root).as_posix()
        items.append(VideoItem(key=f"ubfc_{_safe(rel)}", path=str(vid), dataset="ubfc", label=0,
                               group=f"ubfc_{_safe(rel)}", split=None, method="real", gt=str(gt) if gt else None))
    if not items:
        raise FileNotFoundError(f"no */vid.avi found under {root}")
    return items


def load_ubfc_gt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth pulse oximeter trace: (time in seconds from video start, PPG waveform).

    DATASET_2 ground_truth.txt has 3 rows: PPG, heart rate, time (s).
    DATASET_1 gtdump.xmp is CSV: time (ms), heart rate, SpO2, PPG.
    """
    path = Path(path)
    if path.suffix.lower() == ".xmp":
        d = np.loadtxt(path, delimiter=",")
        t, ppg = d[:, 0] / 1000.0, d[:, 3]
    else:
        d = np.loadtxt(path)
        if d.ndim != 2 or d.shape[0] < 3:
            raise ValueError(f"unexpected ground_truth.txt shape {d.shape} in {path}")
        ppg, t = d[0], d[2]
        if np.nanmax(t) > 1000:  # some exports store milliseconds
            t = t / 1000.0
    if len(t) < 2 or not np.all(np.diff(t) > 0):
        t = np.arange(len(ppg)) / 30.0  # UBFC is recorded at 30 fps, one sample per frame
    return t - t[0], ppg.astype(float)


# --------------------------------------------------------------------------- FaceForensics++

def ffpp(root: str | Path, compression: str = "c23", methods=FFPP_METHODS,
         splits_dir: str | Path = SPLITS_DIR / "ffpp") -> list[VideoItem]:
    root = Path(root)
    split_of: dict[str, tuple[str, str]] = {}
    for s in ("train", "val", "test"):
        f = Path(splits_dir) / f"{s}.json"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run `python -m pulsecheck fetch-splits`")
        for a, b in json.loads(f.read_text()):
            pair = "ffpp_" + "_".join(sorted((a, b)))
            split_of[a] = split_of[b] = (s, pair)

    items = []
    for vid in sorted((root / "original_sequences" / "youtube" / compression).rglob("*.mp4")):
        s, pair = split_of.get(vid.stem, (None, f"ffpp_{vid.stem}"))
        items.append(VideoItem(key=f"ffpp_{compression}_original_{vid.stem}", path=str(vid), dataset="ffpp",
                               label=0, group=pair, split=s, method="original"))
    for m in methods:
        for vid in sorted((root / "manipulated_sequences" / m / compression).rglob("*.mp4")):
            target = vid.stem.split("_")[0]
            s, pair = split_of.get(target, (None, f"ffpp_{target}"))
            items.append(VideoItem(key=f"ffpp_{compression}_{m}_{vid.stem}", path=str(vid), dataset="ffpp",
                                   label=1, group=pair, split=s, method=m))
    if not items:
        raise FileNotFoundError(f"no FF++ videos for compression {compression} under {root}")
    return items


# --------------------------------------------------------------------------- Celeb-DF v2

def celebdf(root: str | Path, test_only: bool = True) -> list[VideoItem]:
    root = Path(root)
    listing = root / "List_of_testing_videos.txt"
    test = set()
    if listing.exists():
        for line in listing.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                test.add(parts[1].replace("\\", "/"))
    items = []
    for folder, label in (("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)):
        for vid in sorted((root / folder).glob("*.mp4")):
            rel = f"{folder}/{vid.name}"
            split = "test" if rel in test else "train"
            if test_only and test and split != "test":
                continue
            m = re.match(r"(id\d+)_", vid.stem)
            group = f"celebdf_{m.group(1)}" if m else f"celebdf_yt_{vid.stem}"
            items.append(VideoItem(key=f"celebdf_{folder}_{vid.stem}", path=str(vid), dataset="celebdf",
                                   label=label, group=group, split=split, method=folder))
    if not items:
        raise FileNotFoundError(f"no Celeb-DF videos under {root}")
    return items


# --------------------------------------------------------------------------- synthetic

def synthetic(root: str | Path) -> list[VideoItem]:
    root = Path(root)
    labels = json.loads((root / "labels.json").read_text())
    return [
        VideoItem(key=f"synth_{Path(d['file']).stem}", path=str(root / d["file"]), dataset="synthetic",
                  label=int(d["label"]), group=d["group"], split=hash_split(d["group"]),
                  method=d["params"]["kind"], gt=str(root / d["gt"]) if d.get("gt") else None)
        for d in labels
    ]


LOADERS = {"ubfc": ubfc, "ffpp": ffpp, "celebdf": celebdf, "synthetic": synthetic}


# --------------------------------------------------------------------------- download check

EXPECTED = {
    "ubfc": {"DATASET_2 subjects with vid.avi + ground_truth.txt": 42},
    "ffpp": {"original": 1000, **{m: 1000 for m in FFPP_METHODS}},
    "celebdf": {"Celeb-real": 590, "YouTube-real": 300, "Celeb-synthesis": 5639, "List_of_testing_videos.txt lines": 518},
}


def check_dataset(name: str, root: str | Path, compression: str = "c23") -> list[tuple[str, int, int]]:
    """(what, found, expected) for each part of a download, to catch partial downloads early."""
    root = Path(root)
    rows = []
    if name == "ubfc":
        found = sum(1 for d in (root / "DATASET_2").glob("*") if (d / "vid.avi").exists() and (d / "ground_truth.txt").exists())
        rows.append(("DATASET_2 subjects with vid.avi + ground_truth.txt", found, 42))
    elif name == "ffpp":
        rows.append(("original", len(list((root / "original_sequences/youtube" / compression).rglob("*.mp4"))), 1000))
        for m in FFPP_METHODS:
            rows.append((m, len(list((root / "manipulated_sequences" / m / compression).rglob("*.mp4"))), 1000))
    elif name == "celebdf":
        for folder, n in (("Celeb-real", 590), ("YouTube-real", 300), ("Celeb-synthesis", 5639)):
            rows.append((folder, len(list((root / folder).glob("*.mp4"))), n))
        listing = root / "List_of_testing_videos.txt"
        lines = len([ln for ln in listing.read_text().splitlines() if ln.strip()]) if listing.exists() else 0
        rows.append(("List_of_testing_videos.txt lines", lines, 518))
    else:
        raise ValueError(name)
    return rows
