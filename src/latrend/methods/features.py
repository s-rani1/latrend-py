from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.method import LCMethod
from ..core.trajectories import trajectories


def _trajectory_features(df: pd.DataFrame, polyDegree: int = 3) -> dict[str, float]:
    t = df["Time"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    out: dict[str, float] = {}

    out["y_nobs"] = float(len(y))
    out["y_mean"] = float(np.mean(y)) if len(y) else float("nan")
    out["y_sd"] = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    out["y_min"] = float(np.min(y)) if len(y) else float("nan")
    out["y_max"] = float(np.max(y)) if len(y) else float("nan")
    out["y_auc"] = float(np.trapezoid(y, t)) if len(y) > 1 else 0.0

    # Simple deltas / gradients
    if len(y) > 1:
        dy = np.diff(y)
        dt = np.diff(t)
        out["y_delta_mean"] = float(np.mean(dy))
        out["y_delta_sd"] = float(np.std(dy, ddof=1)) if len(dy) > 1 else 0.0
        grad = dy / np.where(dt == 0, np.nan, dt)
        grad = grad[~np.isnan(grad)]
        out["y_grad_mean"] = float(np.mean(grad)) if len(grad) else 0.0
        out["y_grad_sd"] = float(np.std(grad, ddof=1)) if len(grad) > 1 else 0.0
    else:
        out["y_delta_mean"] = 0.0
        out["y_delta_sd"] = 0.0
        out["y_grad_mean"] = 0.0
        out["y_grad_sd"] = 0.0

    # Polynomial coefficients (raw) on Time
    if len(y) >= polyDegree + 1:
        X = np.column_stack([t ** p for p in range(polyDegree + 1)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    else:
        coef = np.zeros(polyDegree + 1, dtype=float)
    for p, c in enumerate(coef):
        out[f"y_poly_{p}"] = float(c)

    # Linear trend (intercept + slope)
    if len(y) >= 2:
        X = np.column_stack([np.ones_like(t), t])
        coef_lin, *_ = np.linalg.lstsq(X, y, rcond=None)
        out["y_intercept"] = float(coef_lin[0])
        out["y_slope"] = float(coef_lin[1])
    else:
        out["y_intercept"] = float(y[0]) if len(y) == 1 else 0.0
        out["y_slope"] = 0.0

    return out


@dataclass
class lcMethodFeatures(LCMethod):
    nClusters: int = 2
    polyDegree: int = 3
    scale: bool = True
    seed: int | None = None
    kmeans_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.nClusters < 1:
            raise ValueError("nClusters must be >= 1")
        if self.polyDegree < 0:
            raise ValueError("polyDegree must be >= 0")
        if self.name == "Noname":
            self.name = "Features"

    def cluster(self, data: pd.DataFrame) -> pd.Series:
        trajs = trajectories(self, data)
        ids = list(trajs.keys())

        feats = [_trajectory_features(trajs[_id], polyDegree=self.polyDegree) for _id in ids]
        feat_df = pd.DataFrame(feats, index=pd.Index(ids))

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = feat_df.to_numpy(dtype=float)
        if self.scale:
            X = StandardScaler().fit_transform(X)

        kwargs = dict(n_init=10, random_state=self.seed)
        if self.kmeans_kwargs:
            kwargs.update(self.kmeans_kwargs)
        km = KMeans(n_clusters=self.nClusters, **kwargs)
        labels0 = km.fit_predict(X)
        labels = labels0.astype(int) + 1
        return pd.Series(labels, index=pd.Index(ids), name="Cluster")
