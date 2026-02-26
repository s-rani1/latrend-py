from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import pandas as pd

from .method import LCMethod
from .model import LCModel, LCModels, postprobFromAssignments


def fitLatrendMethod(method: LCMethod, data: pd.DataFrame) -> tuple[LCMethod, pd.DataFrame]:
    """
    Mirrors latrend::.fitLatrendMethod() (R):
    validate -> prepareData -> preFit -> fit -> postFit
    """

    method.validate(data)
    prepared = method.prepareData(data)
    method.preFit(prepared)
    method.fit(prepared)
    method.postFit(prepared)
    return method, prepared


def latrendCluster(method: LCMethod, data: pd.DataFrame) -> LCModel:
    # Optional R backend for methods created via lcMethodR / dynamic lcMethod* wrappers.
    if hasattr(method, "r_method"):
        try:
            from ..backends.r import latrendCluster_r  # noqa: WPS433
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to load R backend support.") from e
        return latrendCluster_r(method, data)

    fitted_method, prepared = fitLatrendMethod(method, data)
    clusters = fitted_method.cluster(prepared)

    # Ensure index is ids and labels are 1..k
    if not isinstance(clusters, pd.Series):
        clusters = pd.Series(clusters)
    clusters = clusters.copy()
    clusters.index = clusters.index.astype(object)
    clusters.index.name = fitted_method.id

    model = LCModel(
        method=fitted_method,
        data=prepared,
        clusters=clusters,
        postprob=postprobFromAssignments(clusters),
    )

    # Metrics are computed on-demand.
    model.meta.setdefault("metrics", {})

    return model


def latrendBatchCluster(
    methods: Iterable[LCMethod] | LCMethod,
    data: pd.DataFrame,
    nClusters: Iterable[int] | None = None,
) -> LCModels:
    models = LCModels()
    if isinstance(methods, LCMethod):
        method = methods
        if nClusters is None:
            if not hasattr(method, "nClusters"):
                raise ValueError("nClusters must be provided for methods without a nClusters attribute")
            nClusters = [int(getattr(method, "nClusters"))]
        for k in nClusters:
            try:
                cloned = replace(method, nClusters=int(k))
            except Exception:
                cloned = method
                setattr(cloned, "nClusters", int(k))
            models.append(latrendCluster(cloned, data))
        return models

    for method in methods:
        models.append(latrendCluster(method, data))
    return models


def latrendRepCluster(
    method: LCMethod,
    data: pd.DataFrame,
    *,
    nRep: int = 10,
    metric: str = "silhouette",
    seeds: Iterable[int] | None = None,
) -> LCModels:
    """
    Repeat clustering multiple times (e.g. different random initializations) and return all models.

    This is a small Python approximation of latrend::latrendRepCluster().
    """

    if nRep < 1:
        raise ValueError("nRep must be >= 1")

    if seeds is None:
        seeds = range(1, nRep + 1)

    out = LCModels()
    for i, seed in enumerate(seeds):
        if i >= nRep:
            break
        try:
            m = replace(method, seed=int(seed))
        except Exception:
            m = method
            if hasattr(m, "seed"):
                setattr(m, "seed", int(seed))
        model = latrendCluster(m, data)
        if metric == "silhouette":
            from ..metrics.cluster import silhouette_score_long
            model.meta.setdefault("metrics", {})["silhouette"] = silhouette_score_long(model)
        out.append(model)
    return out
