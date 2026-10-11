"""Step 6: turn window features into real-or-fake.

Each window gets a probability of being fake; a video's score is the mean of
its windows' probabilities (every window votes). Candidate models are compared
with GroupKFold so that no person / source video is on both sides of a fold.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_model(name: str, seed: int = 0):
    impute = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if name == "logreg":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))])
    if name == "svm":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("clf", SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed))])
    if name == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              l2_regularization=1.0, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return Pipeline([impute, ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                                class_weight="balanced_subsample", n_jobs=-1,
                                                                random_state=seed))])
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ("logreg", "svm", "hist_gb", "random_forest")
MODEL_LABELS = {"logreg": "Logistic regression", "svm": "SVM (RBF)", "hist_gb": "Gradient boosting",
                "random_forest": "Random forest"}


def video_scores(window_prob: np.ndarray, video_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean window probability per video. Returns (unique video ids, scores) in first-seen order."""
    ids, first, inv = np.unique(video_ids, return_index=True, return_inverse=True)
    order = np.argsort(first)
    sums = np.bincount(inv, weights=window_prob)
    cnt = np.bincount(inv)
    return ids[order], (sums / cnt)[order]


def video_labels(y: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    ids, first = np.unique(video_ids, return_index=True)
    return np.asarray(y)[first[np.argsort(first)]]


def grouped_oof(model_name: str, X, y, groups, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold window probabilities with GroupKFold (no group on both sides)."""
    oof = np.full(len(y), np.nan)
    n_splits = min(n_splits, len(np.unique(groups)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model(model_name, seed).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def select_model(X, y, groups, video_ids, candidates=MODEL_NAMES, n_splits: int = 5, seed: int = 0,
                 log=print) -> tuple[str, list[dict]]:
    """Cross-validate every candidate on the training split; best video-level AUC wins."""
    rows = []
    for name in candidates:
        t0 = time.time()
        oof = grouped_oof(name, X, y, groups, n_splits, seed)
        vid, vs = video_scores(oof, video_ids)
        vy = video_labels(y, video_ids)
        row = {
            "model": name, "label": MODEL_LABELS[name],
            "window_auc": float(roc_auc_score(y, oof)),
            "video_auc": float(roc_auc_score(vy, vs)),
            "seconds": round(time.time() - t0, 1),
        }
        log(f"  {MODEL_LABELS[name]:20s} window AUC {row['window_auc']:.3f}  video AUC {row['video_auc']:.3f}  ({row['seconds']} s)")
        rows.append(row)
    best = max(rows, key=lambda r: r["video_auc"])["model"]
    return best, rows


@dataclass
class Detector:
    model: object
    feature_names: list[str]
    threshold: float = 0.5
    meta: dict = field(default_factory=dict)

    def window_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty(0)
        return self.model.predict_proba(X)[:, 1]

    def video_proba(self, X: np.ndarray) -> float:
        p = self.window_proba(X)
        return float(p.mean()) if len(p) else float("nan")

    def predict_traces(self, traces, method: str | None = None):
        """Window probabilities, window spans, and the feature rows used (for explanations)."""
        from .features import video_features

        X, names, spans = video_features(traces, method or self.meta.get("rppg_method", "pos"))
        if not len(X):
            return np.empty(0), [], None
        Xd = X[:, [names.index(n) for n in self.feature_names]]
        return self.window_proba(Xd), spans, Xd

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names,
                     "threshold": self.threshold, "meta": self.meta}, path)
        path.with_suffix(".json").write_text(json.dumps(
            {"threshold": self.threshold, "n_features": len(self.feature_names), **self.meta}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        d = joblib.load(path)  # only load model files you trained yourself: joblib runs pickled code
        return cls(model=d["model"], feature_names=d["feature_names"], threshold=d["threshold"], meta=d["meta"])
