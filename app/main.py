"""PulseCheck web app.

Uploads are analysed one at a time on a background thread (a free Hugging Face
CPU Space has 2 vCPUs); the browser polls /api/jobs/<id> for progress. Videos
are written to a temporary file for decoding and deleted as soon as the job
finishes; nothing is kept.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pulsecheck import __version__
from pulsecheck.analysis import analyze_video
from pulsecheck.cli import clean_json
from pulsecheck.config import MAX_ANALYZE_S, MODELS_DIR, REPORTS_DIR
from pulsecheck.detector import Detector
from pulsecheck.pipeline import NoFaceError
from pulsecheck.video import VideoError

log = logging.getLogger("pulsecheck")
logging.basicConfig(level=logging.INFO)

STATIC = Path(__file__).parent / "static"
SAMPLES = STATIC / "samples"
MAX_UPLOAD_MB = int(os.environ.get("PULSECHECK_MAX_UPLOAD_MB", "60"))
MAX_QUEUE = int(os.environ.get("PULSECHECK_MAX_QUEUE", "6"))
JOB_TTL_S = 15 * 60
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
REPORT_FILES = {"hr": "hr_validation.json", "detector": "detector.json", "cnn": "cnn.json",
                "compression": "compression_study.json"}


def _load_detectors() -> list:
    """Every detector trained on real data: models/detector.joblib (classical) and models/cnn.pt (CNN)."""
    found = []
    if (MODELS_DIR / "detector.joblib").exists():
        try:
            found.append(Detector.load(MODELS_DIR / "detector.joblib"))
        except Exception:  # a broken model file must not take the site down
            log.exception("could not load the classical detector")
    if (MODELS_DIR / "cnn.pt").exists():
        try:
            from pulsecheck.deep import CNNDetector  # needs PyTorch

            found.append(CNNDetector.load(MODELS_DIR / "cnn.pt"))
        except Exception:
            log.exception("could not load the CNN detector")
    return found


DETECTORS = _load_detectors()
# The site's verdict comes from whichever model scored best on *validation* data,
# so the choice never peeks at the test set the reported numbers come from.
DETECTOR = max(DETECTORS, key=lambda d: d.meta.get("val_video_auc") or 0.0) if DETECTORS else None
app = FastAPI(title="PulseCheck", version=__version__, docs_url="/api/docs", redoc_url=None)
_pool = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_sample_cache: dict[tuple[str, str], dict] = {}


def _prune() -> None:
    now = time.time()
    with _lock:
        for jid in [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL_S]:
            _jobs.pop(jid, None)


def _queue_len() -> int:
    return sum(1 for j in _jobs.values() if j["status"] in ("queued", "running"))


def _run(job_id: str, path: Path, method: str, seconds: float, cleanup: bool, cache_key=None) -> None:
    job = _jobs[job_id]
    job["status"] = "running"
    job["started"] = time.time()

    def progress(stage: str, frac: float) -> None:
        job["stage"], job["progress"] = stage, round(float(frac), 3)

    try:
        result = analyze_video(path, method=method, max_s=seconds, detector=DETECTOR, progress=progress)
        result = clean_json(result)
        if cache_key:
            _sample_cache[cache_key] = result
        job.update(status="done", result=result, progress=1.0, stage="Done")
    except NoFaceError as e:
        job.update(status="error", error=f"No usable face: {e}. Face the camera in good, even light.")
    except VideoError as e:
        job.update(status="error", error=f"Could not read that video ({e}). Try MP4 (H.264) or WebM.")
    except Exception as e:  # report, don't crash the worker
        log.exception("analysis failed")
        job.update(status="error", error=f"Analysis failed: {type(e).__name__}: {e}")
    finally:
        job["finished"] = time.time()
        if cleanup:
            shutil.rmtree(path.parent, ignore_errors=True)


def _new_job(path: Path, method: str, seconds: float, cleanup: bool, cache_key=None) -> dict:
    _prune()
    with _lock:
        if _queue_len() >= MAX_QUEUE:
            if cleanup:
                shutil.rmtree(path.parent, ignore_errors=True)
            raise HTTPException(429, "The analyser is busy; please try again in a minute.")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"id": job_id, "status": "queued", "stage": "Waiting in the queue", "progress": 0.0,
                         "created": time.time(), "method": method}
    _pool.submit(_run, job_id, path, method, seconds, cleanup, cache_key)
    return {"job_id": job_id}


def _check_params(method: str, seconds: float) -> tuple[str, float]:
    if method not in ("pos", "chrom", "green"):
        raise HTTPException(400, "method must be pos, chrom or green")
    return method, float(min(max(seconds, 5.0), MAX_ANALYZE_S))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), method: str = Form("pos"), seconds: float = Form(MAX_ANALYZE_S)):
    method, seconds = _check_params(method, seconds)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        suffix = ".webm" if "webm" in (file.content_type or "") else ".mp4"
    tmpdir = Path(tempfile.mkdtemp(prefix="pulsecheck_"))
    dest = tmpdir / f"upload{suffix}"
    size, limit = 0, MAX_UPLOAD_MB * 1024 * 1024
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"File is larger than {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    except HTTPException:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    if size == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(400, "Empty file.")
    return _new_job(dest, method, seconds, cleanup=True)


@app.post("/api/analyze-sample")
async def analyze_sample(name: str = Form(...), method: str = Form("pos")):
    method, seconds = _check_params(method, MAX_ANALYZE_S)
    path = (SAMPLES / name).resolve()
    if path.parent != SAMPLES.resolve() or not path.exists():
        raise HTTPException(404, "No such sample.")
    key = (name, method)
    if key in _sample_cache:  # samples are deterministic: answer instantly after the first run
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"id": job_id, "status": "done", "stage": "Done", "progress": 1.0, "created": time.time(),
                         "method": method, "result": _sample_cache[key]}
        return {"job_id": job_id}
    return _new_job(path, method, seconds, cleanup=False, cache_key=key)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown or expired job.")
    out = {k: v for k, v in job.items() if k in ("id", "status", "stage", "progress", "error", "result", "method")}
    if job["status"] == "queued":
        with _lock:
            ahead = sum(1 for j in _jobs.values() if j["status"] in ("queued", "running") and j["created"] < job["created"])
        out["queue_position"] = ahead
    return out


@app.get("/api/info")
async def info():
    reports = {k: (REPORTS_DIR / f).exists() for k, f in REPORT_FILES.items()}
    def describe(d):
        out = {k: d.meta.get(k) for k in ("model", "model_label", "train_source", "trained_at", "val_video_auc",
                                          "test_video_auc", "test_balanced_accuracy", "n_train_videos", "rppg_method")}
        out["threshold"] = d.threshold
        out["active"] = d is DETECTOR
        return out

    model = describe(DETECTOR) if DETECTOR is not None else None
    models = [describe(d) for d in DETECTORS]
    samples = json.loads((SAMPLES / "samples.json").read_text()) if (SAMPLES / "samples.json").exists() else []
    return {"version": __version__, "model": model, "models": models, "reports": reports, "samples": samples,
            "limits": {"max_upload_mb": MAX_UPLOAD_MB, "max_seconds": MAX_ANALYZE_S}}


@app.get("/api/reports")
async def reports():
    """Evaluation reports produced from real datasets (reports/*.json); missing ones are simply absent."""
    out = {key: json.loads((REPORTS_DIR / f).read_text()) for key, f in REPORT_FILES.items() if (REPORTS_DIR / f).exists()}
    return JSONResponse(out, headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True, "model_loaded": DETECTOR is not None}


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return resp


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
