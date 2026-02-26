from __future__ import annotations

import numpy as np

from ..core.matrix import tsmatrix
from ..core.model import LCModel


def silhouette_score_long(model: LCModel) -> float:
    """
    Silhouette score based on wide matrix representation (individual x time).

    Missing values are imputed with per-time means.

    Requires scikit-learn to be installed.
    """

    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        raise ImportError(
            "scikit-learn is required for silhouette_score_long(). "
            "Install with: pip install scikit-learn"
        )

    mat = tsmatrix(model.data, id=model.method.id, time=model.method.time, y=model.method.outcome)
    X = mat.to_numpy(dtype=float)

    # Simple imputation: per-column mean
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if inds[0].size:
        X = X.copy()
        X[inds] = np.take(col_means, inds[1])

    labels = model.clusters.reindex(mat.index).to_numpy(dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(X, labels, metric="euclidean"))
