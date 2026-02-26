from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .method import LCMethod


def postprobFromAssignments(clusters: pd.Series, nClusters: int | None = None) -> pd.DataFrame:
    """
    Construct a hard (0/1) posterior probability matrix from cluster assignments.

    Mirrors latrend::postprobFromAssignments().
    """

    if not isinstance(clusters, pd.Series):
        clusters = pd.Series(clusters)
    if nClusters is None:
        nClusters = int(clusters.nunique())
    ids = pd.Index(clusters.index)
    cols = list(range(1, nClusters + 1))
    arr = np.zeros((len(ids), nClusters), dtype=float)
    for i, label in enumerate(clusters.to_numpy()):
        try:
            j = int(label) - 1
        except Exception:
            continue
        if 0 <= j < nClusters:
            arr[i, j] = 1.0
    return pd.DataFrame(arr, index=ids, columns=cols)


@dataclass
class LCModel:
    method: LCMethod
    data: pd.DataFrame
    clusters: pd.Series  # index: individual ids; values: 1..k
    postprob: pd.DataFrame | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def nClusters(self) -> int:
        if self.clusters.empty:
            return 0
        return int(pd.Series(self.clusters).nunique())

    def ids(self) -> list[Any]:
        return list(pd.Index(self.clusters.index))

    def classCounts(self) -> pd.Series:
        return self.clusters.value_counts().sort_index()

    def classProportions(self) -> pd.Series:
        counts = self.classCounts()
        total = counts.sum()
        return counts / total if total else counts

    def classEntropy(self) -> float:
        if self.postprob is None:
            return 0.0
        p = self.postprob.to_numpy(dtype=float)
        p = np.clip(p, 1e-12, 1.0)
        ent = -np.sum(p * np.log(p), axis=1)
        return float(np.mean(ent))


@dataclass
class LCModels:
    models: list[LCModel] = field(default_factory=list)

    def __iter__(self):
        return iter(self.models)

    def __len__(self) -> int:
        return len(self.models)

    def append(self, model: LCModel) -> None:
        self.models.append(model)

    def bestModel(self, key: str = "silhouette", maximize: bool = True) -> LCModel:
        if not self.models:
            raise ValueError("No models available")
        scores = []
        for model in self.models:
            metrics = model.meta.get("metrics", {})
            scores.append(metrics.get(key, float("nan")))
        scores_arr = np.asarray(scores, dtype=float)
        if np.all(np.isnan(scores_arr)):
            raise ValueError(f"Metric '{key}' not found on any model")
        idx = int(np.nanargmax(scores_arr) if maximize else np.nanargmin(scores_arr))
        return self.models[idx]
