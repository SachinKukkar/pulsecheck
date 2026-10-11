"""Command line: one subcommand per step of GUIDE.md.

    python -m pulsecheck <command> --help
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import CACHE_DIR, MAX_ANALYZE_S, MODELS_DIR, REPORTS_DIR, ROOT, SPLITS_DIR

SPLITS_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{}.json"
DETECTOR_PATH = MODELS_DIR / "detector.joblib"


def clean_json(x):
    """numpy -> python, NaN/inf -> None, so reports are valid JSON."""
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), separators=(",", ":")))
    print(f"wrote {path}")


def _kv(pairs: list[str] | None) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        name, _, path = p.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=CACHE_DIR, got {p!r}")
        out[name] = path
    return out


# --------------------------------------------------------------------------- commands

def cmd_download_model(a):
    from .face import ensure_model

    print(ensure_model())


def cmd_fetch_splits(a):
    out = SPLITS_DIR / "ffpp"
    out.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        urllib.request.urlretrieve(SPLITS_URL.format(s), out / f"{s}.json")
        print(f"{s}: {len(json.loads((out / f'{s}.json').read_text()))} pairs")


def cmd_download_ubfc(a):
    from .download import download_ubfc

    print("By downloading UBFC-rPPG you accept its terms (Agreement.xlsx in the dataset's Drive folder).")
    download_ubfc(a.root, first=a.first, subjects=a.subjects)


def cmd_check_data(a):
    from .datasets import check_dataset

    rows = check_dataset(a.dataset, a.root, a.compression)
    ok = all(found == want for _, found, want in rows)
    for what, found, want in rows:
        mark = "ok  " if found == want else ("MISSING" if found == 0 else "PARTIAL")
        print(f"  [{mark}] {what}: {found} of {want}")
    print("complete" if ok else "incomplete: re-check the download (Drive splits big folders into several zips; extract all of them)")
    if not ok:
        raise SystemExit(1)


def cmd_landmarks(a):
    import cv2

    from .face import FaceLandmarker
    from .roi import draw_landmark_numbers, draw_overlay

    path = Path(a.path)
    img = cv2.imread(str(path))
    if img is None:  # a video: take its first frame
        cap = cv2.VideoCapture(str(path))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read an image or video frame from {path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with FaceLandmarker(video=False) as lm:
        pts = lm.detect(rgb)
    if pts is None:
        raise SystemExit("no face found")
    out = Path(a.out)
    cv2.imwrite(str(out), cv2.cvtColor(draw_landmark_numbers(rgb, pts), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out.with_name(out.stem + "_patches.png")), cv2.cvtColor(draw_overlay(rgb, pts), cv2.COLOR_RGB2BGR))
    print(f"wrote {out} and {out.with_name(out.stem + '_patches.png')}")


def cmd_analyze(a):
    from .analysis import analyze_video
    from .detector import Detector

    det = Detector.load(a.model) if a.model and Path(a.model).exists() else None
    res = analyze_video(a.video, method=a.method, max_s=a.seconds, detector=det)
    hr = res["hr"]
    print(f"heart rate: {hr['bpm']} bpm  (SNR {hr['snr_db']} dB, method {a.method})")
    print("  per method: " + ", ".join(f"{m} {v['bpm']}" for m, v in hr["per_method"].items()))
    print("  per patch:  " + ", ".join(f"{k} {v}" for k, v in hr["per_roi"].items()))
    for w in res["warnings"]:
        print(f"  [{w['level']}] {w['text']}")
    if res["detector"]:
        d = res["detector"]
        print(f"detector: {d['verdict']}  (p_fake {d['prob_fake']}, threshold {d.get('threshold')})")
    else:
        print("detector: no trained model (models/detector.joblib); run `train` first")
    if a.json:
        write_json(Path(a.json), res)


def cmd_synth_clip(a):
    from .synthetic import SynthParams, make_clip

    truth = make_clip(a.out, SynthParams(kind=a.kind, bpm=a.bpm, seconds=a.seconds, seed=a.seed), crf=a.crf)
    print(f"wrote {a.out}: {truth['params']['kind']}, mean {truth['mean_bpm']:.1f} bpm")


def cmd_synth_dataset(a):
    from .synthetic import make_dataset

    print(make_dataset(a.out, n_pairs=a.pairs, seconds=a.seconds, seed=a.seed))


def cmd_extract(a):
    from . import datasets
    from .extract import extract_dataset

    if a.dataset in datasets.EXPECTED and not (a.allow_partial or a.limit):
        rows = datasets.check_dataset(a.dataset, a.root, a.compression)
        if a.dataset == "ffpp" and a.methods:
            rows = [r for r in rows if r[0] in ("original", *a.methods)]
        short = [r for r in rows if r[1] != r[2]]
        if short:
            for what, found, want in short:
                print(f"  {what}: {found} of {want}")
            raise SystemExit("This download looks incomplete (see `python -m pulsecheck check-data`). "
                             "Finish it, or pass --allow-partial to process what is there.")

    if a.dataset == "ffpp":
        items = datasets.ffpp(a.root, compression=a.compression, methods=a.methods or datasets.FFPP_METHODS)
        if a.split:
            items = [it for it in items if it.split in a.split]
    elif a.dataset == "celebdf":
        items = datasets.celebdf(a.root, test_only=not a.include_train)
    else:
        items = datasets.LOADERS[a.dataset](a.root)
    if a.limit:  # a balanced quick subset: first N of each method
        by = {}
        for it in items:
            by.setdefault(it.method, []).append(it)
        items = [it for group in by.values() for it in group[: a.limit]]
    default = f"{a.dataset}_{a.compression}" if a.dataset == "ffpp" else a.dataset
    extract_dataset(items, a.cache or CACHE_DIR / default, max_s=a.max_seconds or None, workers=a.workers,
                    skin_mask=not a.no_skin_mask, overwrite=a.overwrite)


def cmd_eval_hr(a):
    from .hr_eval import evaluate_hr

    write_json(Path(a.out), evaluate_hr(a.cache, source=a.source))


def cmd_compression(a):
    from .studies import compression_study

    rep = compression_study(a.root, a.work, baseline_cache=a.baseline_cache, crfs=a.crf, limit=a.limit,
                            max_s=a.max_seconds, workers=a.workers)
    write_json(Path(a.out), rep)


def cmd_train(a):
    from .detect_eval import run_detection

    det, rep = run_detection(a.cache, cross=_kv(a.cross), real_only=_kv(a.real), extra=_kv(a.extra),
                             models=a.models, method=a.method, source=a.source)
    det.save(a.model_out)
    print(f"wrote {a.model_out}")
    write_json(Path(a.report_out), rep)


def cmd_train_cnn(a):
    from .deep import run_cnn

    det, rep = run_cnn(a.cache, cross=_kv(a.cross), real_only=_kv(a.real), extra=_kv(a.extra), method=a.method,
                       epochs=a.epochs, batch=a.batch, seed=a.seed, device=a.device, source=a.source)
    det.save(a.model_out)
    print(f"wrote {a.model_out}")
    write_json(Path(a.report_out), rep)


def readme_results() -> str:
    """Markdown results table from the real reports (never synthetic ones)."""
    from .report_doc import METHOD, _f, _load, _pct

    hr, det, cnn = (_load(n) for n in ("hr_validation.json", "detector.json", "cnn.json"))
    if not (hr or det or cnn):
        return ("*Not run yet.* Results appear here once the real-dataset commands in [GUIDE.md](GUIDE.md)\n"
                "have been run; `python -m pulsecheck update-readme` writes this section from `reports/*.json`.")
    rows = ["| Test | Result | Data |", "|---|---|---|"]
    if hr:
        h = hr["headline"]
        rows.append(f"| Heart rate, {hr['source']} ({METHOD[h['method']]}, {h['window_s']} s windows) | MAE **{_f(h['mae'])} bpm**, "
                    f"RMSE {_f(h['rmse'])}, r = {_f(h['pearson_r'])} | {hr['n_videos']} videos vs a finger pulse oximeter |")
    for rep, name in ((det, det and det.get("chosen_label")), (cnn, "CNN on PPG maps")):
        if not rep:
            continue
        tv = rep["test_video"]
        ci = tv.get("auc_ci95") or [None, None]
        rows.append(f"| {name}, {rep['source']} test | AUC **{_f(tv.get('auc'), 3)}** (95% CI {_f(ci[0], 2)}-{_f(ci[1], 2)}), "
                    f"balanced accuracy {_pct(tv.get('balanced_accuracy'))} | {tv['n']} held-out videos |")
        for g in rep.get("generalization", []):
            if g["kind"] == "real_only":
                rows.append(f"| {name}, {g['name']} genuine videos | false alarms {_pct(g.get('false_alarm_rate'))} | {g['n_videos']} videos never seen |")
            else:
                rows.append(f"| {name}, {g['name']} | AUC {_f(g.get('auc'), 3)} | {g['n_videos']} videos never seen |")
    rows.append("")
    rows.append(f"*Generated {time.strftime('%Y-%m-%d')} from `reports/*.json` by `python -m pulsecheck update-readme`.*")
    return "\n".join(rows)


def cmd_update_readme(a):
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    path.write_text(text[:i] + "\n" + readme_results() + "\n" + text[j:], encoding="utf-8")
    print(f"updated the results section of {path}")


def cmd_report_doc(a):
    from .report_doc import build_report

    print(f"wrote {build_report(a.out, author=a.author, space_url=a.space_url)}")


def cmd_smoke(a):
    """Synthetic end-to-end run: proves the plumbing, says nothing about real deepfakes."""
    from .datasets import synthetic
    from .detect_eval import run_detection
    from .extract import extract_dataset
    from .hr_eval import evaluate_hr
    from .synthetic import make_dataset

    work = Path(a.work)
    t0 = time.time()
    if not (work / "videos" / "labels.json").exists():
        print(f"1/4 rendering {a.pairs} real/fake-like pairs ...")
        make_dataset(work / "videos", n_pairs=a.pairs, seconds=a.seconds, seed=0)
    print("2/4 extracting traces ...")
    extract_dataset(synthetic(work / "videos"), work / "cache", max_s=None, workers=a.workers)

    # A software test only: everything it writes stays inside the work folder, never in
    # reports/ or models/, so nothing synthetic can reach the website or the write-up.
    print("3/4 heart-rate check against the painted-on pulse ...")
    hr = evaluate_hr(work / "cache", windows_s=(5, 10), headline=("pos", 10), source="synthetic")
    write_json(work / "hr_validation.json", hr)
    print("4/4 training and testing the detector ...")
    det, rep = run_detection(work / "cache", models=a.models, source="synthetic")
    write_json(work / "detector.json", rep)
    det.save(work / "detector.joblib")
    print(f"smoke test finished in {(time.time() - t0) / 60:.1f} min; outputs in {work}")


def cmd_serve(a):
    import uvicorn

    uvicorn.run("app.main:app", host=a.host, port=a.port, reload=a.reload)


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pulsecheck", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download-model", help="fetch MediaPipe face_landmarker.task (checksum-verified)")
    s.set_defaults(fn=cmd_download_model)
    s = sub.add_parser("fetch-splits", help="fetch the official FaceForensics++ train/val/test split lists")
    s.set_defaults(fn=cmd_fetch_splits)

    s = sub.add_parser("download-ubfc", help="fetch UBFC-rPPG DATASET_2 from Google Drive (needs gdown; resumable)")
    s.add_argument("--root", required=True, help="e.g. C:/datasets/UBFC-rPPG")
    s.add_argument("--first", type=int, default=None, help="only the first N subjects")
    s.add_argument("--subjects", nargs="*", help="specific folders, e.g. subject1 subject3")
    s.set_defaults(fn=cmd_download_ubfc)

    s = sub.add_parser("check-data", help="count a dataset download's videos against the official numbers")
    s.add_argument("dataset", choices=["ubfc", "ffpp", "celebdf"])
    s.add_argument("--root", required=True)
    s.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"], help="FF++ only")
    s.set_defaults(fn=cmd_check_data)

    s = sub.add_parser("landmarks", help="draw the 468 landmark numbers and the six patches on a face")
    s.add_argument("path", help="image, or video (first frame is used)")
    s.add_argument("--out", default="landmarks.png")
    s.set_defaults(fn=cmd_landmarks)

    s = sub.add_parser("analyze", help="heart rate (+ verdict if a model is trained) for one video")
    s.add_argument("video")
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--seconds", type=float, default=MAX_ANALYZE_S)
    s.add_argument("--model", default=str(DETECTOR_PATH))
    s.add_argument("--json", help="also write the full result here")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("synth-clip", help="render one synthetic clip with a known pulse")
    s.add_argument("out")
    s.add_argument("--kind", default="real", choices=["real", "fake_mismatch", "fake_flat", "fake_desync"])
    s.add_argument("--bpm", type=float, default=72.0)
    s.add_argument("--seconds", type=float, default=15.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--crf", type=int, default=None, help="H.264 quality (omit for lossless)")
    s.set_defaults(fn=cmd_synth_clip)

    s = sub.add_parser("synth-dataset", help="render paired synthetic real/fake-like clips")
    s.add_argument("out")
    s.add_argument("--pairs", type=int, default=20)
    s.add_argument("--seconds", type=float, default=15.0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth_dataset)

    s = sub.add_parser("extract", help="landmark a whole dataset into cached traces (slow, resumable)")
    s.add_argument("dataset", choices=["ubfc", "ffpp", "celebdf", "synthetic"])
    s.add_argument("--root", required=True, help="folder the dataset was unpacked into")
    s.add_argument("--cache", help="output folder (default cache/<dataset>[_<compression>])")
    s.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"], help="FF++ only")
    s.add_argument("--methods", nargs="*", help="FF++ manipulations (default: all five)")
    s.add_argument("--split", nargs="*", choices=["train", "val", "test"], help="FF++: only these splits")
    s.add_argument("--include-train", action="store_true", help="Celeb-DF: also the non-test videos")
    s.add_argument("--max-seconds", type=float, default=20.0, help="use the first N seconds (default 20; 0 = whole video)")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--limit", type=int, default=None, help="first N videos of each method (quick runs)")
    s.add_argument("--no-skin-mask", action="store_true")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--allow-partial", action="store_true", help="process an incomplete download anyway")
    s.set_defaults(fn=cmd_extract)

    s = sub.add_parser("eval-hr", help="heart-rate error against the oximeter (UBFC-rPPG)")
    s.add_argument("--cache", default=str(CACHE_DIR / "ubfc"))
    s.add_argument("--source", default="UBFC-rPPG")
    s.add_argument("--out", default=str(REPORTS_DIR / "hr_validation.json"))
    s.set_defaults(fn=cmd_eval_hr)

    s = sub.add_parser("compression-study", help="heart-rate error vs H.264 compression level (UBFC)")
    s.add_argument("--root", required=True, help="UBFC-rPPG folder")
    s.add_argument("--work", default=str(CACHE_DIR / "compression"))
    s.add_argument("--baseline-cache", default=None, help="reuse an existing UBFC cache made with the same --max-seconds")
    s.add_argument("--crf", type=int, nargs="*", default=[18, 23, 28, 33, 38])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--max-seconds", type=float, default=30.0)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--out", default=str(REPORTS_DIR / "compression_study.json"))
    s.set_defaults(fn=cmd_compression)

    s = sub.add_parser("train", help="select, train and test the detector; write model + report")
    s.add_argument("--cache", default=str(CACHE_DIR / "ffpp_c23"), help="training dataset cache (needs splits)")
    s.add_argument("--cross", nargs="*", help="NAME=CACHE: other datasets to test on, e.g. Celeb-DF=cache/celebdf")
    s.add_argument("--real", nargs="*", help="NAME=CACHE: all-real sets for false alarms, e.g. UBFC=cache/ubfc")
    s.add_argument("--extra", nargs="*", help="NAME=CACHE: same dataset, other variant, e.g. FF++c40=cache/ffpp_c40")
    s.add_argument("--models", nargs="*", default=["logreg", "svm", "hist_gb", "random_forest"])
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--source", default="FaceForensics++ c23")
    s.add_argument("--model-out", default=str(DETECTOR_PATH))
    s.add_argument("--report-out", default=str(REPORTS_DIR / "detector.json"))
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("train-cnn", help="train the PPG-map CNN on the GPU (needs PyTorch); write model + report")
    s.add_argument("--cache", default=str(CACHE_DIR / "ffpp_c23"), help="training dataset cache (needs splits)")
    s.add_argument("--cross", nargs="*", help="NAME=CACHE: other datasets to test on")
    s.add_argument("--real", nargs="*", help="NAME=CACHE: all-real sets for false alarms")
    s.add_argument("--extra", nargs="*", help="NAME=CACHE: same dataset, other variant")
    s.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    s.add_argument("--source", default="FaceForensics++ c23")
    s.add_argument("--model-out", default=str(MODELS_DIR / "cnn.pt"))
    s.add_argument("--report-out", default=str(REPORTS_DIR / "cnn.json"))
    s.set_defaults(fn=cmd_train_cnn)

    s = sub.add_parser("report-doc", help="build the Word technical report from the code and reports/*.json")
    s.add_argument("--out", default=str(ROOT / "docs" / "PulseCheck_Technical_Report.docx"))
    s.add_argument("--author", default=None, help="name printed on the title page")
    s.add_argument("--space-url", default=None, help="the live Hugging Face Space URL, once deployed")
    s.set_defaults(fn=cmd_report_doc)

    s = sub.add_parser("update-readme", help="write README.md's results table from reports/*.json")
    s.set_defaults(fn=cmd_update_readme)

    s = sub.add_parser("smoke-test", help="synthetic end-to-end run of every step (minutes, no datasets)")
    s.add_argument("--work", default=str(CACHE_DIR / "smoke"))
    s.add_argument("--pairs", type=int, default=24)
    s.add_argument("--seconds", type=float, default=16.0)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--models", nargs="*", default=["logreg", "hist_gb"])
    s.set_defaults(fn=cmd_smoke)

    s = sub.add_parser("serve", help="run the website locally")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=7860)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
