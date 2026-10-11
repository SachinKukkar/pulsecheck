"""The website's API: upload a clip, poll the job, get the analysis."""

import time

import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_and_info():
    assert client.get("/healthz").json()["ok"] is True
    info = client.get("/api/info").json()
    assert {"version", "model", "models", "reports", "limits"} <= set(info)


def test_index_page_is_served():
    r = client.get("/")
    assert r.status_code == 200 and "PulseCheck" in r.text


def test_reports_endpoint_only_returns_real_reports():
    rep = client.get("/api/reports").json()
    for body in rep.values():
        assert body.get("source") != "synthetic"


def test_rejects_bad_method_and_empty_file():
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"x", "video/mp4")}, data={"method": "nope"}).status_code == 400
    assert client.post("/api/analyze", files={"file": ("a.mp4", b"", "video/mp4")}).status_code == 400


def test_upload_and_poll(known_pulse_clip):
    path, truth = known_pulse_clip
    with open(path, "rb") as f:
        r = client.post("/api/analyze", files={"file": ("clip.mp4", f, "video/mp4")}, data={"method": "pos"})
    assert r.status_code == 200
    job = r.json()["job_id"]
    for _ in range(240):
        s = client.get(f"/api/jobs/{job}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert s["status"] == "done", s.get("error")
    assert s["result"]["hr"]["bpm"] == pytest.approx(truth["mean_bpm"], abs=2.0)


def test_unknown_job_is_404():
    assert client.get("/api/jobs/doesnotexist").status_code == 404
