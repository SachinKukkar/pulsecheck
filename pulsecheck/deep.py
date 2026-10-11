"""Step 6, deep version: a small CNN on PPG maps (after FakeCatcher), trained on the GPU.

A PPG map is an image whose rows are the 32 grid patches and whose columns are
time: each row is that patch's pulse (POS, standardised) over one 8 s window.
Two channels:
  0 - heart-rate band (0.7-3 Hz): does every row carry the same beat, in step?
  1 - wide band (0.7-12 Hz): includes the frame-to-frame flicker generators leave.
On a real face the rows line up; a generated face tends to break that pattern
in ways the hand-made features may not anticipate.

The protocol is identical to the classical models (detect_eval.py): official
splits, early stopping and threshold on val only, every number from test or
from datasets never trained on. PyTorch is only needed here; the rest of the
project runs without it.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .config import DETECT_STEP_S, DETECT_WINDOW_S, HR_BAND, TARGET_FS, WIDE_BAND
from .detector import video_labels, video_scores
from .extract import load_index
from .metrics import best_threshold, detection_metrics, score_hist
from .rppg import raw_pulse
from .signals import bandpass, detrend, windows
from .traces import Traces

MAP_VERSION = 1
WIN = int(round(DETECT_WINDOW_S * TARGET_FS))  # 240 samples
CROP = 224  # training crops a random 224-sample slice of each window (time-shift augmentation)


# --------------------------------------------------------------------------- PPG maps

def _zrows(x: np.ndarray) -> np.ndarray:
    """Standardise each row of (rows, time); dead rows become zeros."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (x - mu) / np.where(sd > 1e-12, sd, 1), 0.0)


def ppg_maps(traces: Traces, method: str = "pos", win_s: float = DETECT_WINDOW_S,
             step_s: float = DETECT_STEP_S) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """(n_windows, 2, 32, WIN) float32 maps and each window's (start, end) seconds."""
    if traces.grid_rgb is None:
        raise ValueError("these traces have no 32-patch grid; re-extract them with this version")
    fs = traces.fs
    if abs(fs - TARGET_FS) > 1e-6:
        raise ValueError(f"expected traces at {TARGET_FS} Hz, got {fs}")
    g = traces.grid_rgb
    rows = []
    for j in range(g.shape[1]):
        col = g[:, j]
        if not np.isfinite(col).all() or col[:, 1].std() == 0:
            rows.append(None)
            continue
        rows.append(detrend(raw_pulse(col, fs, method)))
    ok = np.array([r is not None for r in rows])
    raw = np.stack([r if r is not None else np.zeros(len(traces.t)) for r in rows])  # (32, T)
    wide_hi = min(WIDE_BAND[1], 0.45 * fs)
    hr = bandpass(raw, fs, *HR_BAND, axis=1) * ok[:, None]
    wide = bandpass(raw, fs, WIDE_BAND[0], wide_hi, axis=1) * ok[:, None]
    maps, spans = [], []
    w = int(round(win_s * fs))
    for sl in windows(len(traces.t), fs, win_s, step_s):
        if sl.stop - sl.start < w:
            continue  # the CNN needs full-length windows
        maps.append(np.stack([_zrows(hr[:, sl]), _zrows(wide[:, sl])]).astype(np.float32))
        spans.append((sl.start / fs, sl.stop / fs))
    if not maps:
        return np.empty((0, 2, g.shape[1], w), np.float32), []
    return np.stack(maps), spans


@dataclass
class MapTable:
    X: np.ndarray  # (N, 2, 32, WIN) float16
    y: np.ndarray
    groups: np.ndarray
    vids: np.ndarray
    splits: np.ndarray
    manip: np.ndarray

    def subset(self, mask) -> "MapTable":
        return MapTable(self.X[mask], self.y[mask], self.groups[mask], self.vids[mask], self.splits[mask], self.manip[mask])

    @property
    def n_videos(self) -> int:
        return len(np.unique(self.vids))


def _maps_one(args):
    cache_dir, item, method = args
    try:
        X, _ = ppg_maps(Traces.load(Path(cache_dir) / f"{item['key']}.npz"), method)
    except ValueError:
        X = None
    return item, X


def map_table(cache_dir: str | Path, method: str = "pos", workers: int | None = None, log=print) -> MapTable:
    """PPG maps for every cached video, memoised as float16 in ``ppgmaps_<method>.npz``."""
    cache_dir = Path(cache_dir)
    items = load_index(cache_dir)
    keys = np.array([it["key"] for it in items])
    memo = cache_dir / f"ppgmaps_{method}.npz"
    if memo.exists():
        z = np.load(memo, allow_pickle=False)
        if int(z["version"]) == MAP_VERSION and np.array_equal(z["keys"], keys):
            return MapTable(z["X"], z["y"], z["groups"], z["vids"], z["splits"], z["manip"])
    t0 = time.time()
    jobs = [(str(cache_dir), it, method) for it in items]
    if len(jobs) > 40:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_maps_one, jobs, chunksize=16))
    else:
        results = [_maps_one(j) for j in jobs]
    Xs, meta, missing = [], [], 0
    for item, X in results:
        if X is None:
            missing += 1
            continue
        for x in X:
            Xs.append(x.astype(np.float16))
            meta.append((item["label"], item["group"], item["key"], item.get("split") or "", item["method"]))
    if missing:
        log(f"  {missing} cached videos have no 32-patch grid (extracted by an older version) and were skipped")
    if not Xs:
        raise RuntimeError(f"no PPG maps from {cache_dir}")
    y, g, v, s, m = (np.array(c) for c in zip(*meta))
    T = MapTable(np.stack(Xs), y.astype(int), g, v, s, m)
    np.savez(memo, version=MAP_VERSION, keys=keys, X=T.X, y=T.y, groups=T.groups, vids=T.vids, splits=T.splits, manip=T.manip)
    log(f"PPG maps: {len(T.y)} windows from {T.n_videos} videos in {time.time() - t0:.0f} s")
    return T


# --------------------------------------------------------------------------- the network

def build_model(in_ch: int = 2, dropout: float = 0.3):
    """~80k parameters: four conv blocks over (patch rows x time), then global pooling.

    Kernels are short across rows (3 neighbouring patches) and long across time
    (9 -> 3 samples), so early layers compare a patch's pulse with its neighbours'.
    """
    import torch.nn as nn

    def block(cin, cout, k, pool):
        layers = [nn.Conv2d(cin, cout, k, padding=(k[0] // 2, k[1] // 2), bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        return layers

    return nn.Sequential(
        *block(in_ch, 16, (3, 9), (1, 2)),
        *block(16, 32, (3, 7), (2, 2)),
        *block(32, 64, (3, 5), (2, 2)),
        *block(64, 64, (3, 3), None),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1),
    )


def _device(name: str | None):
    import torch

    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _predict(model, X: np.ndarray, device, batch: int = 512) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(X), batch):
            xb = torch.from_numpy(X[a:a + batch].astype(np.float32)).to(device)
            off = (xb.shape[-1] - CROP) // 2  # evaluate on the centre crop, as trained
            out.append(torch.sigmoid(model(xb[..., off:off + CROP]).float()).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def _video_auc(p, T: MapTable) -> float:
    from sklearn.metrics import roc_auc_score

    _, vs = video_scores(p, T.vids)
    vy = video_labels(T.y, T.vids)
    return float(roc_auc_score(vy, vs)) if len(np.unique(vy)) == 2 else float("nan")


def train_cnn(Ttr: MapTable, Tva: MapTable, *, epochs: int = 40, batch: int = 256, lr: float = 1e-3,
              weight_decay: float = 1e-2, patience: int = 8, seed: int = 0, device: str | None = None, log=print):
    """Train with AdamW + cosine LR, class-balanced loss, early stopping on val video AUC."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    dev = _device(device)
    model = build_model().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    n_fake, n_real = int(Ttr.y.sum()), int((1 - Ttr.y).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_real / max(n_fake, 1)], device=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Xtr = torch.from_numpy(Ttr.X)  # float16 on the CPU; batches are moved to the GPU as needed
    ytr = torch.from_numpy(Ttr.y.astype(np.float32))
    rng = np.random.default_rng(seed)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    log(f"CNN: {n_params:,} parameters on {gpu}; {len(ytr)} training windows ({n_fake} fake, {n_real} real)")

    history, best, best_state, stale = [], -1.0, None, 0
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(ytr))
        total, seen = 0.0, 0
        for a in range(0, len(order), batch):
            idx = torch.from_numpy(order[a:a + batch])
            xb = Xtr[idx].to(dev, non_blocking=True).float()
            off = int(rng.integers(0, xb.shape[-1] - CROP + 1))  # random time shift
            xb = xb[..., off:off + CROP]
            if rng.random() < 0.5:
                xb = -xb  # pulse polarity is not what makes a face fake
            yb = ytr[idx].to(dev).unsqueeze(1)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
            seen += len(idx)
        sched.step()
        pv = _predict(model, Tva.X, dev)
        from sklearn.metrics import roc_auc_score

        row = {"epoch": ep, "train_loss": total / max(seen, 1), "val_window_auc": float(roc_auc_score(Tva.y, pv)),
               "val_video_auc": _video_auc(pv, Tva), "seconds": round(time.time() - t0, 1)}
        history.append(row)
        log(f"  epoch {ep:2d}  loss {row['train_loss']:.4f}  val AUC window {row['val_window_auc']:.3f} video {row['val_video_auc']:.3f}  ({row['seconds']} s)")
        if row["val_video_auc"] > best + 1e-4:
            best, stale = row["val_video_auc"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log(f"  early stop: no val improvement for {patience} epochs")
                break
    model.load_state_dict(best_state)
    best_epoch = max(history, key=lambda r: r["val_video_auc"])["epoch"]
    info = {"parameters": n_params, "device": gpu, "epochs_run": len(history), "best_epoch": best_epoch,
            "train_minutes": round((time.time() - t_start) / 60, 2), "batch": batch, "lr": lr,
            "weight_decay": weight_decay, "seed": seed, "torch": str(torch.__version__)}
    return model, history, info


# --------------------------------------------------------------------------- run + report

class CNNDetector:
    """Loads models/cnn.pt; same interface the website uses for the classical detector."""

    def __init__(self, model, threshold: float, meta: dict, device: str = "cpu"):
        self.model, self.threshold, self.meta, self.device = model, threshold, meta, device
        self.feature_names: list[str] = []

    def predict_traces(self, traces: Traces, method: str = "pos"):
        X, spans = ppg_maps(traces, self.meta.get("rppg_method", method))
        if not len(X):
            return np.empty(0), [], None
        return _predict(self.model, X, _device(self.device)), spans, None

    def save(self, path: str | Path) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Plain JSON types only, so the file loads with torch.load(weights_only=True) - the safe mode
        # that refuses to run pickled code (a str subclass like torch.__version__ would be rejected).
        meta = json.loads(json.dumps(self.meta, default=str))
        torch.save({"state_dict": self.model.state_dict(), "threshold": float(self.threshold), "meta": meta}, path)
        path.with_suffix(".json").write_text(json.dumps({"threshold": float(self.threshold), **meta}, indent=2))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "CNNDetector":
        import torch

        d = torch.load(path, map_location=device, weights_only=True)  # weights only: no pickled code runs
        model = build_model()
        model.load_state_dict(d["state_dict"])
        model.eval()
        return cls(model.to(device), float(d["threshold"]), d["meta"], device)


def run_cnn(train_cache: str | Path, *, cross: dict[str, str] | None = None, real_only: dict[str, str] | None = None,
            extra: dict[str, str] | None = None, method: str = "pos", epochs: int = 40, batch: int = 256,
            seed: int = 0, device: str | None = None, source: str = "FaceForensics++", log=print):
    T = map_table(train_cache, method, log=log)
    tr, va, te = (T.splits == s for s in ("train", "val", "test"))
    if not (tr.any() and va.any() and te.any()):
        raise RuntimeError("training cache needs train, val and test splits")
    Ttr, Tva, Tte = T.subset(tr), T.subset(va), T.subset(te)
    log(f"train {Ttr.n_videos} videos / val {Tva.n_videos} / test {Tte.n_videos}")
    model, history, info = train_cnn(Ttr, Tva, epochs=epochs, batch=batch, seed=seed, device=device, log=log)
    dev = next(model.parameters()).device

    pv = _predict(model, Tva.X, dev)
    _, vs_va = video_scores(pv, Tva.vids)
    vy_va = video_labels(Tva.y, Tva.vids)
    thr = best_threshold(vy_va, vs_va)
    val_auc = _video_auc(pv, Tva)

    pt = _predict(model, Tte.X, dev)
    ids_te, vs_te = video_scores(pt, Tte.vids)
    vy_te = video_labels(Tte.y, Tte.vids)
    test_video = detection_metrics(vy_te, vs_te, thr)
    test_window = detection_metrics(Tte.y, pt, thr, ci=False)
    log(f"TEST video-level: AUC {test_video.get('auc', float('nan')):.3f}  balanced acc {test_video.get('balanced_accuracy') or 0:.3f}")

    man_of = dict(zip(Tte.vids, Tte.manip))
    vman = np.array([man_of[i] for i in ids_te])
    per_method = []
    for m in sorted(set(vman) - {"original", "real"}):
        sel = (vy_te == 0) | (vman == m)
        if len(np.unique(vy_te[sel])) == 2:
            mm = detection_metrics(vy_te[sel], vs_te[sel], thr, curves=True, ci=False)
            per_method.append({"method": m, "auc": mm["auc"], "detection_rate": float((vs_te[vman == m] >= thr).mean()),
                               "n_fake": int((vman == m).sum()), "roc": mm["roc"]})

    def external(caches, kind):
        out = []
        for name, cdir in (caches or {}).items():
            E = map_table(cdir, method, log=log)
            p = _predict(model, E.X, dev)
            _, vs = video_scores(p, E.vids)
            vy = video_labels(E.y, E.vids)
            m = detection_metrics(vy, vs, thr)
            out.append({"name": name, "kind": kind, "n_videos": int(len(vy)), **m, "hist": score_hist(vs, vy)})
            log(f"  {name}: " + (f"false-alarm rate {m['false_alarm_rate']:.1%}" if kind == "real_only" else f"AUC {m.get('auc', float('nan')):.3f}"))
        return out

    generalization = external(cross, "cross_dataset") + external(extra, "same_dataset_variant") + external(real_only, "real_only")
    meta = {"model": "cnn", "model_label": "CNN on PPG maps", "version": __version__, "trained_at": time.strftime("%Y-%m-%d %H:%M"),
            "train_source": source, "rppg_method": method, "window_s": DETECT_WINDOW_S, "step_s": DETECT_STEP_S,
            "n_train_videos": Ttr.n_videos, "n_train_windows": int(len(Ttr.y)), "val_video_auc": val_auc,
            "test_video_auc": test_video.get("auc"), "test_balanced_accuracy": test_video.get("balanced_accuracy"), **info}
    det = CNNDetector(model.cpu(), thr, meta)
    report = {
        "kind": "cnn", "source": source, "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "protocol": {"model_selection": "early stopping on val video AUC", "threshold": "max balanced accuracy on val",
                     "aggregation": "mean of window probabilities", "window_s": DETECT_WINDOW_S, "step_s": DETECT_STEP_S,
                     "rppg_method": method, "input": f"2 x 32 x {CROP} PPG map (HR band, wide band)"},
        "data": {s: {"videos": int(sub.n_videos), "windows": int(len(sub.y)), "fake_videos": int(video_labels(sub.y, sub.vids).sum())}
                 for s, sub in (("train", Ttr), ("val", Tva), ("test", Tte))},
        "training": info, "history": history, "threshold": thr, "val_video_auc": val_auc,
        "test_video": test_video, "test_window": test_window, "test_hist": score_hist(vs_te, vy_te),
        "per_method": per_method, "generalization": generalization,
    }
    return det, report
