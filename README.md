---
title: PulseCheck
emoji: 🫀
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Heart rate and deepfake screening from the pulse in video
---

# PulseCheck

**Heart rate from ordinary face video, and a deepfake check built on the same pulse.**

Each heartbeat pushes blood under the skin, so for a moment the skin reflects a little
less green light. A camera can record that. PulseCheck measures the pulse from video
(remote photoplethysmography, rPPG), then checks whether every patch of the face shares
the same heartbeat. A real face does; many generated faces don't (FakeCatcher, 2020).

- **Pipeline**: MediaPipe face landmarks → six skin regions + a 32-patch grid →
  Green / CHROM / POS rPPG → band-pass + FFT → heart rate and signal quality.
- **Two detectors, trained locally**: scikit-learn models on 91 patch-agreement features
  (CPU), and a CNN on FakeCatcher-style PPG maps (PyTorch, RTX 3050 GPU), compared on the
  same splits.
- **Honest evaluation**: person-disjoint official splits, GroupKFold, thresholds frozen on
  validation, cross-dataset tests, false alarms on genuine videos, and the leaky-split score
  shown next to the honest one.
- **Website**: upload or record a clip and see every step as interactive charts; a results
  dashboard; a live lab for the FFT maths. FastAPI + Plotly, Docker, Hugging Face Spaces.

## Results

<!-- results:start -->
*Not run yet.* Results appear here once the real-dataset commands in [GUIDE.md](GUIDE.md)
have been run; `python -m pulsecheck update-readme` writes this section from `reports/*.json`.
<!-- results:end -->

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126   # optional, for the CNN
python -m pulsecheck download-model
python -m pytest
python -m pulsecheck serve              # http://127.0.0.1:7860
```

Full walkthrough: **[GUIDE.md](GUIDE.md)**. Getting the data: **[DATASETS.md](DATASETS.md)**.
Technical report: `python -m pulsecheck report-doc` → `docs/PulseCheck_Technical_Report.docx`.

## Commands

| Command | What it does |
|---|---|
| `check-data` | count a download's videos against the official numbers |
| `extract` | landmark a dataset into cached traces (parallel, resumable) |
| `eval-hr` | heart-rate error against the oximeter (UBFC-rPPG) |
| `compression-study` | heart-rate error vs H.264 compression |
| `train` | select, train and test the classical detector |
| `train-cnn` | train and test the PPG-map CNN on the GPU |
| `analyze` | heart rate (+ verdict) for one video |
| `report-doc` / `update-readme` | regenerate the Word report / this README's results |
| `serve` | run the website |

`python -m pulsecheck <command> --help` lists every option.

## Layout

| Path | What it is |
|---|---|
| `pulsecheck/video.py` | frames and trustworthy timestamps |
| `pulsecheck/face.py`, `roi.py` | MediaPipe landmarks, skin patches, per-video skin model |
| `pulsecheck/pipeline.py`, `traces.py` | video → per-patch colour traces on a 30 Hz clock (cached) |
| `pulsecheck/signals.py`, `rppg.py` | detrend, band-pass, FFT, SNR; Green / CHROM / POS |
| `pulsecheck/features.py`, `detector.py` | 91 window features; classical models |
| `pulsecheck/deep.py` | PPG maps and the CNN (PyTorch) |
| `pulsecheck/hr_eval.py`, `detect_eval.py`, `studies.py` | evaluation protocols → `reports/*.json` |
| `pulsecheck/datasets.py`, `extract.py`, `download.py` | dataset loaders, download checks, parallel extraction |
| `pulsecheck/report_doc.py` | the Word report |
| `app/` | FastAPI backend and the static website |
| `tests/` | pytest suite |

## Data, licences and responsible use

No dataset is included or redistributed. UBFC-rPPG, FaceForensics++ and Celeb-DF each
have their own terms; get them from their authors ([DATASETS.md](DATASETS.md)). The code
is MIT-licensed.

This is a learning project, not a forensic tool. High-quality face swaps can carry over the
pulse of their source video (Seibold et al., 2025), so a believable pulse is evidence, not
proof. Never use a verdict to accuse anyone.

## References

Verkruysse et al. 2008 · de Haan & Jeanne 2013 (CHROM) · Wang et al. 2017 (POS) ·
Ciftci, Demir & Yin 2020 (FakeCatcher) · Bobbia et al. 2019 (UBFC-rPPG) ·
Rössler et al. 2019 (FaceForensics++) · Li et al. 2020 (Celeb-DF) · Seibold et al. 2025.
Full citations with DOIs are on the website's About page and in the technical report.
