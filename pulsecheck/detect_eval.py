"""Steps 6-7 at scale: train on one dataset, test it honestly, write the numbers.

Protocol (FaceForensics++):
  * official train / val / test splits, which keep both videos of every
    source-target pair on one side (so no person leaks across)
  * model selection by 5-fold GroupKFold inside train
  * decision threshold chosen on val, then frozen
  * every reported number comes from test (or from datasets never trained on)
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from . import __version__
from .config import DETECT_STEP_S, DETECT_WINDOW_S
from .detector import MODEL_LABELS, MODEL_NAMES, Detector, make_model, select_model, video_labels, video_scores
from .extract import load_index
from .features import describe, video_features
from .metrics import best_threshold, detection_metrics, score_hist
from .traces import Traces

FEATURE_CACHE_VERSION = 2  # bump whenever features.py changes, so cached tables are rebuilt


@dataclass
class FeatureTable:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    vids: np.ndarray
    splits: np.ndarray
    manip: np.ndarray
    names: list[str]

    def subset(self, mask: np.ndarray) -> "FeatureTable":
        return FeatureTable(self.X[mask], self.y[mask], self.groups[mask], self.vids[mask],
                            self.splits[mask], self.manip[mask], self.names)

    @property
    def n_videos(self) -> int:
        return len(np.unique(self.vids))


def _one(args):
    cache_dir, item, method = args
    tr = Traces.load(Path(cache_dir) / f"{item['key']}.npz")
    X, names, _ = video_features(tr, method)
    return item, X, names


def feature_table(cache_dir: str | Path, method: str = "pos", workers: int | None = None, log=print) -> FeatureTable:
    """Window features for every cached video, memoised in ``features_<method>.npz``."""
    cache_dir = Path(cache_dir)
    items = load_index(cache_dir)
    keys = np.array([it["key"] for it in items])
    memo = cache_dir / f"features_{method}.npz"
    if memo.exists():
        z = np.load(memo, allow_pickle=False)
        if int(z["version"]) == FEATURE_CACHE_VERSION and np.array_equal(z["keys"], keys):
            return FeatureTable(z["X"], z["y"], z["groups"], z["vids"], z["splits"], z["manip"],
                                [str(s) for s in z["names"]])
    t0 = time.time()
    jobs = [(str(cache_dir), it, method) for it in items]
    if len(jobs) > 40:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, jobs, chunksize=16))
    else:
        results = [_one(j) for j in jobs]
    rows, meta, names = [], [], None
    for item, X, nm in results:
        names = nm
        for x in X:
            rows.append(x)
            meta.append((item["label"], item["group"], item["key"], item.get("split") or "", item["method"]))
    if not rows:
        raise RuntimeError(f"no feature windows from {cache_dir}")
    y, g, v, s, m = (np.array(c) for c in zip(*meta))
    T = FeatureTable(np.vstack(rows), y.astype(int), g, v, s, m, names)
    np.savez_compressed(memo, version=FEATURE_CACHE_VERSION, keys=keys, X=T.X, y=T.y, groups=T.groups,
                        vids=T.vids, splits=T.splits, manip=T.manip, names=np.array(T.names))
    log(f"features: {len(T.y)} windows from {T.n_videos} videos in {time.time() - t0:.0f} s")
    return T


def _video_eval(det_model, T: FeatureTable, threshold: float, ci: bool = True) -> tuple[dict, np.ndarray, np.ndarray]:
    p = det_model.predict_proba(T.X)[:, 1]
    ids, vs = video_scores(p, T.vids)
    vy = video_labels(T.y, T.vids)
    return detection_metrics(vy, vs, threshold, ci=ci), vy, vs


def _distributions(T: FeatureTable, names: list[str], bins: int = 24) -> list[dict]:
    out = []
    for name in names:
        x = T.X[:, T.names.index(name)]
        ok = np.isfinite(x)
        if ok.sum() < 10:
            continue
        lo, hi = np.percentile(x[ok], [1, 99])
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, bins + 1)
        dens = lambda v: (np.histogram(np.clip(v, lo, hi), edges, density=True)[0]).round(5).tolist()  # noqa: E731
        out.append({"feature": name, "label": describe(name), "edges": edges.round(5).tolist(),
                    "real": dens(x[ok & (T.y == 0)]), "fake": dens(x[ok & (T.y == 1)])})
    return out


def run_detection(train_cache: str | Path, *, cross: dict[str, str] | None = None,
                  real_only: dict[str, str] | None = None, extra: dict[str, str] | None = None,
                  models=MODEL_NAMES, method: str = "pos", seed: int = 0, source: str = "FaceForensics++",
                  log=print) -> tuple[Detector, dict]:
    T = feature_table(train_cache, method, log=log)
    tr, va, te = (T.splits == s for s in ("train", "val", "test"))
    if not (tr.any() and va.any() and te.any()):
        raise RuntimeError("training cache needs train, val and test splits")
    Ttr, Tva, Tte = T.subset(tr), T.subset(va), T.subset(te)
    log(f"train {Ttr.n_videos} videos / val {Tva.n_videos} / test {Tte.n_videos}  ({len(T.names)} features)")

    log("model selection, 5-fold GroupKFold on train:")
    best, cv_rows = select_model(Ttr.X, Ttr.y, Ttr.groups, Ttr.vids, models, seed=seed, log=log)

    # Pitfall demo: the same model scored with a random (leaky) split of windows.
    leaky = np.full(len(Ttr.y), np.nan)
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Ttr.X):
        leaky[b] = make_model(best, seed).fit(Ttr.X[a], Ttr.y[a]).predict_proba(Ttr.X[b])[:, 1]
    _, lv = video_scores(leaky, Ttr.vids)
    vy_tr = video_labels(Ttr.y, Ttr.vids)
    grouped_row = next(r for r in cv_rows if r["model"] == best)
    split_demo = {
        "random_window_split": {"window_auc": float(roc_auc_score(Ttr.y, leaky)), "video_auc": float(roc_auc_score(vy_tr, lv))},
        "grouped_split": {"window_auc": grouped_row["window_auc"], "video_auc": grouped_row["video_auc"]},
    }
    log(f"  leaky random split video AUC {split_demo['random_window_split']['video_auc']:.3f} vs grouped "
        f"{split_demo['grouped_split']['video_auc']:.3f}")

    model = make_model(best, seed).fit(Ttr.X, Ttr.y)
    _, vy_va, vs_va = _video_eval(model, Tva, 0.5, ci=False)
    thr = best_threshold(vy_va, vs_va)
    val_auc = float(roc_auc_score(vy_va, vs_va)) if len(np.unique(vy_va)) == 2 else float("nan")
    log(f"chosen: {MODEL_LABELS[best]}; val video AUC {val_auc:.3f}; threshold {thr:.3f} (max balanced accuracy on val)")

    test_video, vy_te, vs_te = _video_eval(model, Tte, thr)
    pw = model.predict_proba(Tte.X)[:, 1]
    test_window = detection_metrics(Tte.y, pw, thr, ci=False)
    log(f"TEST video-level: AUC {test_video.get('auc', float('nan')):.3f}  balanced acc "
        f"{test_video.get('balanced_accuracy') or 0:.3f}  (window-level AUC {test_window.get('auc', float('nan')):.3f})")

    # Per manipulation: real test videos vs that method's test fakes.
    ids_te, _ = video_scores(pw, Tte.vids)
    man_of = dict(zip(Tte.vids, Tte.manip))
    vman = np.array([man_of[i] for i in ids_te])
    per_method = []
    for m in sorted(set(vman) - {"original", "real"}):
        sel = (vy_te == 0) | (vman == m)
        if len(np.unique(vy_te[sel])) == 2:
            mm = detection_metrics(vy_te[sel], vs_te[sel], thr, curves=True, ci=False)
            per_method.append({"method": m, "auc": mm["auc"], "detection_rate": float((vs_te[vman == m] >= thr).mean()),
                               "n_fake": int((vman == m).sum()), "roc": mm["roc"]})

    log("permutation importance on test windows...")
    imp = permutation_importance(model, Tte.X, Tte.y, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1][:15]
    importance = [{"feature": T.names[i], "label": describe(T.names[i]), "auc_drop": float(imp.importances_mean[i]),
                   "std": float(imp.importances_std[i])} for i in order]
    # Typical values per class (train windows) so the website can say "yours: x, typical real: y".
    quart = lambda x: [round(float(v), 4) for v in np.nanpercentile(x, [25, 50, 75])]  # noqa: E731
    profile = [{"feature": d["feature"], "label": d["label"],
                "real_q": quart(Ttr.X[Ttr.y == 0, T.names.index(d["feature"])]),
                "fake_q": quart(Ttr.X[Ttr.y == 1, T.names.index(d["feature"])])} for d in importance]

    def external(caches: dict[str, str] | None, kind: str) -> list[dict]:
        out = []
        for name, cdir in (caches or {}).items():
            E = feature_table(cdir, method, log=log)
            m, vy, vs = _video_eval(model, E, thr)
            row = {"name": name, "kind": kind, "n_videos": int(len(vy)), **m, "hist": score_hist(vs, vy)}
            out.append(row)
            if kind == "real_only":
                log(f"  {name}: false-alarm rate {m['false_alarm_rate']:.1%} on {len(vy)} real videos")
            else:
                log(f"  {name}: AUC {m.get('auc', float('nan')):.3f} on {len(vy)} videos")
        return out

    generalization = external(cross, "cross_dataset") + external(extra, "same_dataset_variant") + external(real_only, "real_only")

    meta = {
        "train_source": source, "rppg_method": method, "window_s": DETECT_WINDOW_S, "step_s": DETECT_STEP_S,
        "n_train_videos": Ttr.n_videos, "n_train_windows": int(len(Ttr.y)), "val_video_auc": val_auc,
        "test_video_auc": test_video.get("auc"), "test_balanced_accuracy": test_video.get("balanced_accuracy"),
    }
    det = Detector(model=model, feature_names=T.names, threshold=thr, meta={
        "model": best, "model_label": MODEL_LABELS[best], "version": __version__,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"), "feature_profile": profile, **meta})

    report = {
        "kind": "detector", "source": source, "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "protocol": {"model_selection": "5-fold GroupKFold on train", "threshold": "max balanced accuracy on val",
                     "aggregation": "mean of window probabilities", "window_s": DETECT_WINDOW_S, "step_s": DETECT_STEP_S,
                     "rppg_method": method},
        "data": {s: {"videos": int(sub.n_videos), "windows": int(len(sub.y)), "fake_videos": int(video_labels(sub.y, sub.vids).sum())}
                 for s, sub in (("train", Ttr), ("val", Tva), ("test", Tte))},
        "model_selection": cv_rows, "chosen_model": best, "chosen_label": MODEL_LABELS[best], "threshold": thr,
        "val_video_auc": val_auc,
        "test_video": test_video, "test_window": test_window,
        "test_hist": score_hist(vs_te, vy_te),
        "per_method": per_method, "split_demo": split_demo,
        "importance": importance, "distributions": _distributions(Tte, [d["feature"] for d in importance[:6]]),
        "generalization": generalization,
    }
    return det, report
