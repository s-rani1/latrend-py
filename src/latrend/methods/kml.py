from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from ..core.matrix import tsmatrix
from ..core.method import LCMethod


def _impute_matrix(mat: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing trajectory values for clustering.

    Strategy:
    1) column mean (same timepoint across subjects),
    2) row mean (subject-level),
    3) 0.0 fallback.
    """

    x = mat.copy()
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.mean(axis=0))
    x = x.T.fillna(x.mean(axis=1)).T
    x = x.fillna(0.0)
    return x


def _relabel_deterministic(labels1: np.ndarray, x: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Relabel clusters deterministically based on trajectory level/slope signature.

    This stabilizes label numbering across seeds/runs.
    """

    signatures: list[tuple[float, float, int]] = []
    for c in range(1, n_clusters + 1):
        idx = labels1 == c
        if not np.any(idx):
            signatures.append((float("inf"), float("inf"), c))
            continue
        sub = x[idx]
        level = float(np.mean(sub[:, 0]))
        slope = float(np.mean(sub[:, -1] - sub[:, 0]))
        signatures.append((level, slope, c))
    order = [c for _, _, c in sorted(signatures)]
    remap = {old: new for new, old in enumerate(order, start=1)}
    return np.array([remap[int(v)] for v in labels1], dtype=int)


def _trajectory_score(labels1: np.ndarray, x: np.ndarray, n_clusters: int, distance: str) -> float:
    """
    Cluster quality objective for strict mode: mean within-cluster pairwise distance.
    """

    metric_map = {"euclidean": "euclidean", "manhattan": "cityblock", "correlation": "correlation"}
    metric = metric_map[distance]

    total_dist = 0.0
    total_pairs = 0
    for c in range(1, n_clusters + 1):
        sub = x[labels1 == c]
        n = sub.shape[0]
        if n < 2:
            continue
        d = pdist(sub, metric=metric)
        total_dist += float(np.nansum(d))
        total_pairs += int(len(d))
    if total_pairs == 0:
        return float("inf")
    return total_dist / total_pairs


@dataclass
class lcMethodKML(LCMethod):
    """
    KML-style trajectory clustering in pure Python.

    This implementation clusters per-subject trajectory vectors with KMeans.
    """

    nClusters: int = 2
    mode: str = "kml_fast"
    center: bool = True
    scale: bool = False
    distance: str = "euclidean"
    seed: int | None = None
    nStarts: int = 20
    nInit: int = 100
    maxIter: int = 500
    kmeans_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.nClusters < 1:
            raise ValueError("nClusters must be >= 1")
        if self.mode not in {"kml_fast", "kml_strict"}:
            raise ValueError("mode must be one of: 'kml_fast', 'kml_strict'")
        if self.distance not in {"euclidean", "manhattan", "correlation"}:
            raise ValueError("distance must be one of: 'euclidean', 'manhattan', 'correlation'")
        if self.nStarts < 1:
            raise ValueError("nStarts must be >= 1")
        if self.name == "Noname":
            self.name = "KML"

    def cluster(self, data: pd.DataFrame) -> pd.Series:
        mat = tsmatrix(data, id=self.id, time=self.time, y=self.outcome)
        mat = _impute_matrix(mat)

        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        x = mat.to_numpy(dtype=float)
        if self.center:
            # KML-style behavior is primarily shape-based; center each trajectory.
            x = x - x.mean(axis=1, keepdims=True)
        if self.scale:
            x = StandardScaler().fit_transform(x)

        kwargs = dict(n_init=int(self.nInit), max_iter=int(self.maxIter))
        if self.kmeans_kwargs:
            kwargs.update(self.kmeans_kwargs)

        # Fast mode: single seeded run.
        if self.mode == "kml_fast":
            km = KMeans(n_clusters=self.nClusters, random_state=self.seed, **kwargs)
            labels = km.fit_predict(x).astype(int) + 1
            labels = _relabel_deterministic(labels, x, self.nClusters)
            return pd.Series(labels, index=mat.index.astype(object), name="Cluster")

        # Strict mode: many independent starts, select best trajectory-centric score.
        rng = np.random.default_rng(self.seed)
        start_seeds = rng.integers(0, 2**31 - 1, size=int(self.nStarts), endpoint=False)
        best_score = float("inf")
        best_inertia = float("inf")
        best_labels: np.ndarray | None = None

        for s in start_seeds:
            km = KMeans(n_clusters=self.nClusters, random_state=int(s), **kwargs)
            labels = km.fit_predict(x).astype(int) + 1
            score = _trajectory_score(labels, x, self.nClusters, self.distance)
            inertia = float(getattr(km, "inertia_", float("inf")))
            if (score < best_score) or (np.isclose(score, best_score) and inertia < best_inertia):
                best_score = score
                best_inertia = inertia
                best_labels = labels

        assert best_labels is not None
        best_labels = _relabel_deterministic(best_labels, x, self.nClusters)
        return pd.Series(best_labels, index=mat.index.astype(object), name="Cluster")
