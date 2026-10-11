"""Fetch UBFC-rPPG DATASET_2 from the authors' public Google Drive folder.

Needs ``pip install gdown``. By downloading you accept the dataset's terms
(Agreement.xlsx in the same Drive folder). Its readme also says: facial images
of DATASET_2 subjects 21 and 27 must not appear in any publication,
presentation or report - this project never publishes dataset frames.

Each video is ~1.3 GB (uncompressed 640x480 RGB at 30 fps), so the whole
set is ~55-65 GB. Downloads resume, and finished files are skipped, so the
command can be re-run to fetch more subjects.
"""

from __future__ import annotations

import re
from pathlib import Path

UBFC_ROOT_FOLDER = "1o0XU4gTIo46YfwaWjIgbtCncc-oF44Xk"
UBFC_DATASET2_FOLDER = "1q4vWuF2GJvKP5xyeX8dxaJ2fmq97-4ai"


def _subject_number(name: str) -> int:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else 10**6


def download_ubfc(root: str | Path, first: int | None = None, subjects: list[str] | None = None, log=print) -> Path:
    import gdown

    root = Path(root)
    out = root / "DATASET_2"
    files = gdown.download_folder(id=UBFC_DATASET2_FOLDER, skip_download=True, quiet=True)
    by_subject: dict[str, list] = {}
    for f in files:
        subject = Path(f.path.replace("\\", "/")).parts[0]
        by_subject.setdefault(subject, []).append(f)
    names = sorted(by_subject, key=_subject_number)
    if subjects:
        names = [n for n in names if n in set(subjects)]
    elif first:
        names = names[:first]
    log(f"UBFC-rPPG DATASET_2: {len(by_subject)} subjects available, fetching {len(names)} into {out}")
    blocked = []
    for k, name in enumerate(names, 1):
        for f in by_subject[name]:
            dest = out / Path(f.path.replace("\\", "/"))
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            log(f"[{k}/{len(names)}] {dest.relative_to(root)}")
            try:
                gdown.download(id=f.id, output=str(dest), quiet=True, resume=True)
            except gdown.exceptions.FileURLRetrievalError:
                # Google Drive's anonymous-download quota ("too many users have downloaded this
                # file recently"). It resets within 24 h; a signed-in browser download also works.
                blocked.append((dest.relative_to(root).as_posix(), f"https://drive.google.com/uc?id={f.id}"))
                log("   blocked by Google Drive's download quota; skipping for now")
    if blocked:
        log(f"{len(blocked)} file(s) blocked by the quota. Re-run later, or download them in a browser:")
        for rel, url in blocked:
            log(f"   {rel}  <-  {url}")
    log("done")
    return out
