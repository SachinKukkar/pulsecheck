"""Shared constants.

Every frequency here is in hertz. Beats a minute = hertz x 60.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("PULSECHECK_MODELS", ROOT / "models"))
REPORTS_DIR = Path(os.environ.get("PULSECHECK_REPORTS", ROOT / "reports"))
CACHE_DIR = Path(os.environ.get("PULSECHECK_CACHE", ROOT / "cache"))
SPLITS_DIR = ROOT / "data" / "splits"

# Where a resting/active heartbeat can live: 0.7-3.0 Hz = 42-180 beats a minute.
HR_BAND = (0.7, 3.0)

# FakeCatcher keeps 0.7-14 Hz because the "noise" above the pulse carries
# generator artefacts. We cap the top at 12 Hz and at 45% of the frame rate.
WIDE_BAND = (0.7, 12.0)

# Every trace is resampled onto this uniform clock, so the rest of the code
# never has to care about variable-frame-rate webcam recordings.
TARGET_FS = 30.0

# Window used inside CHROM and POS (de Haan 2013, Wang 2017): about one beat.
RPPG_WINDOW_S = 1.6

# Heart-rate timeline: 10 s windows (0.1 Hz = 6 bpm bin spacing), 1 s hop.
HR_WINDOW_S = 10.0
HR_STEP_S = 1.0

# Deepfake features are computed on 8 s windows with a 4 s hop; each window
# votes and the video score is the mean of its windows' probabilities.
DETECT_WINDOW_S = 8.0
DETECT_STEP_S = 4.0

# Uploaded clips are analysed for at most this many seconds (website and CLI).
MAX_ANALYZE_S = 20.0

# Frames larger than this (longest side, pixels) are shrunk before landmarking.
# Averaging a patch is nearly unchanged by area downscaling, and it is faster.
FRAME_MAX_SIDE = 720

# A frame counts as unusable if the face is missing; a clip is rejected when
# fewer than this fraction of frames have a face.
MIN_FACE_FRACTION = 0.6
