"""End to end: video file -> landmarks -> patches -> pulse -> heart rate."""

import numpy as np
import pytest

from pulsecheck import rppg, signals
from pulsecheck.analysis import analyze_traces
from pulsecheck.pipeline import NoFaceError, extract_traces, ita_category, skin_ita
from pulsecheck.roi import GRID_LANDMARKS, ROI_NAMES
from pulsecheck.traces import Traces


def test_extract_recovers_known_heart_rate(known_pulse_clip, tmp_path):
    path, truth = known_pulse_clip
    tr, preview = extract_traces(path, want_preview=True)
    assert tr.fs == 30.0 and abs(tr.duration - 12.0) < 0.2
    assert tr.rgb.shape[1:] == (len(ROI_NAMES), 3)
    assert tr.grid_rgb.shape[1:] == (len(GRID_LANDMARKS), 3)
    assert tr.meta["face_fraction"] == 1.0 and tr.meta["skin_mask"]
    assert preview is not None and preview.overlay.shape == preview.frame.shape
    for m in rppg.METHODS:
        bpm = signals.hr_bpm(rppg.pulse(tr.pooled(), tr.fs, m), tr.fs)
        assert bpm == pytest.approx(truth["mean_bpm"], abs=2.0), m

    tr.save(tmp_path / "t.npz")
    back = Traces.load(tmp_path / "t.npz")
    assert np.allclose(back.rgb, tr.rgb, atol=1e-3) and np.allclose(back.grid_rgb, tr.grid_rgb, atol=1e-3)


def test_analysis_result_is_complete(known_pulse_clip):
    path, truth = known_pulse_clip
    tr, preview = extract_traces(path, want_preview=True)
    r = analyze_traces(tr, preview)
    assert r["hr"]["bpm"] == pytest.approx(truth["mean_bpm"], abs=2.0)
    assert set(r["hr"]["per_method"]) == {"pos", "chrom", "green"}
    assert len(r["rois"]["coherence"]) == 6
    assert r["preview"].startswith("data:image/jpeg;base64,")
    assert r["detector"] is None  # no model passed
    assert len(r["signals"]["t"]) == len(tr.t)


def test_no_face_is_reported(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from pulsecheck.synthetic import write_video

    frames = (np.full((120, 160, 3), 128, np.uint8) for _ in range(60))
    write_video(tmp_path / "blank.mp4", frames, 30.0)
    with pytest.raises(NoFaceError):
        extract_traces(tmp_path / "blank.mp4")


def test_skin_tone_angle_orders_light_above_dark():
    light, dark = skin_ita([235, 200, 180]), skin_ita([95, 60, 45])
    assert light > dark
    assert ita_category(light) in ("very light", "light") and ita_category(dark) in ("brown", "dark")
