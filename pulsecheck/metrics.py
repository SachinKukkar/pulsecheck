"""How you know it works: heart-rate error and detection metrics, JSON-ready."""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn import metrics as skm


def _f(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


# --------------------------------------------------------------------------- heart rate

def hr_metrics(est: np.ndarray, gt: np.ndarray) -> dict:
    """Mean absolute error and friends, in beats a minute."""
    est, gt = np.asarray(est, float), np.asarray(gt, float)
    ok = np.isfinite(est) & np.isfinite(gt)
    est, gt = est[ok], gt[ok]
    if len(est) == 0:
        return {"n": 0}
    err = est - gt
    sd = err.std(ddof=1) if len(err) > 1 else 0.0
    r = stats.pearsonr(est, gt)[0] if len(est) > 2 and est.std() > 0 and gt.std() > 0 else np.nan
    return {
        "n": int(len(err)),
        "mae": _f(np.abs(err).mean()),
        "rmse": _f(np.sqrt((err**2).mean())),
        "mape": _f((np.abs(err) / gt).mean() * 100),
        "pearson_r": _f(r),
        "within_5bpm": _f((np.abs(err) <= 5).mean()),
        "within_10bpm": _f((np.abs(err) <= 10).mean()),
        "bias": _f(err.mean()),
        "loa_low": _f(err.mean() - 1.96 * sd),
        "loa_high": _f(err.mean() + 1.96 * sd),
    }


# --------------------------------------------------------------------------- detection

def downsample_curve(x: np.ndarray, y: np.ndarray, thr: np.ndarray | None = None, max_points: int = 200) -> dict:
    """Keep a curve's shape in at most ``max_points`` points (always keeps both ends)."""
    n = len(x)
    idx = np.unique(np.r_[np.linspace(0, n - 1, min(n, max_points)).round().astype(int)]) if n else np.array([], int)
    out = {"x": [round(float(v), 5) for v in x[idx]], "y": [round(float(v), 5) for v in y[idx]]}
    if thr is not None:
        out["thr"] = [round(float(v), 5) if np.isfinite(v) else None for v in thr[np.minimum(idx, len(thr) - 1)]]
    return out


def equal_error_rate(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Error rate where false-alarm rate = miss rate, and the threshold that achieves it."""
    fpr, tpr, thr = skm.roc_curve(y, score)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising balanced accuracy (Youden's J). Pick it on validation data only."""
    fpr, tpr, thr = skm.roc_curve(y, score)
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return min(max(t, 0.0), 1.0)


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float | None, float | None]:
    """95% interval for ROC AUC by resampling videos with replacement."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        aucs.append(skm.roc_auc_score(y[i], score[i]))
    if not aucs:
        return None, None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)


def detection_metrics(y: np.ndarray, score: np.ndarray, threshold: float, curves: bool = True, ci: bool = True) -> dict:
    """Threshold-free (AUC, AP, EER) and thresholded (accuracy, confusion) metrics. Label 1 = fake."""
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    out: dict = {"n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum()), "threshold": threshold}
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out.update(
        accuracy=_f(skm.accuracy_score(y, pred)),
        balanced_accuracy=_f(skm.balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None,
        precision=_f(skm.precision_score(y, pred, zero_division=0)),
        recall=_f(skm.recall_score(y, pred, zero_division=0)),
        specificity=_f(tn / (tn + fp)) if tn + fp else None,
        f1=_f(skm.f1_score(y, pred, zero_division=0)),
        false_alarm_rate=_f(fp / (tn + fp)) if tn + fp else None,
        confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    )
    if len(np.unique(y)) < 2:
        return out
    out["auc"] = _f(skm.roc_auc_score(y, score))
    out["average_precision"] = _f(skm.average_precision_score(y, score))
    out["eer"], _ = equal_error_rate(y, score)
    if ci:
        out["auc_ci95"] = list(bootstrap_auc(y, score))
    if curves:
        fpr, tpr, thr = skm.roc_curve(y, score)
        out["roc"] = downsample_curve(fpr, tpr, thr)
        prec, rec, thr_pr = skm.precision_recall_curve(y, score)
        out["pr"] = downsample_curve(rec[::-1], prec[::-1], np.r_[thr_pr, np.nan][::-1])
    return out


def score_hist(score: np.ndarray, y: np.ndarray, bins: int = 20) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    return {
        "edges": [round(float(e), 3) for e in edges],
        "real": np.histogram(score[y == 0], edges)[0].tolist(),
        "fake": np.histogram(score[y == 1], edges)[0].tolist(),
    }
