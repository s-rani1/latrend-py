from __future__ import annotations

from typing import Any

import pandas as pd

from ..core.model import LCModel, postprobFromAssignments
from ..methods.rmethod import RLCMethod


def is_r_available() -> bool:
    try:
        import rpy2  # noqa: F401
    except Exception:
        return False
    return True


def _require_rpy2():
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "R backend requires rpy2 + an R installation with the R package 'latrend' available. "
            "Install with: pip install 'latrend[r]' and ensure R can load latrend."
        ) from e

    pandas2ri.activate()
    return ro, pandas2ri, importr


def _convert_kwargs(ro, kwargs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k == "formula" and isinstance(v, str) and "~" in v:
            out[k] = ro.Formula(v)
        else:
            out[k] = v
    return out


def latrendCluster_r(method: RLCMethod, data: pd.DataFrame, **kwargs: Any) -> LCModel:
    """
    Delegate clustering to R latrend via rpy2 and return a Python LCModel.
    """

    ro, pandas2ri, importr = _require_rpy2()

    try:
        r_latrend = importr("latrend")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "R package 'latrend' is not available in your R library. "
            "Install it in R first (e.g. install.packages('latrend'))."
        ) from e

    # Build R method + R data.frame
    if not hasattr(r_latrend, method.r_method):
        raise ValueError(f"R latrend does not export a function named '{method.r_method}'")
    r_method_fun = getattr(r_latrend, method.r_method)

    r_args = _convert_kwargs(ro, method.r_args)
    r_method = r_method_fun(**r_args)

    r_df = pandas2ri.py2rpy(data)

    r_model = r_latrend.latrendCluster(r_method, r_df, **_convert_kwargs(ro, kwargs))

    # Extract clusters/postprob (best-effort)
    r_clusters = r_latrend.clusters(r_model)
    ids = list(r_clusters.names) if getattr(r_clusters, "names", None) is not None else None
    labels = list(r_clusters)
    if ids is None:
        # Fall back to stable id order from the Python data
        ids = list(pd.Index(data[method.id].unique()).astype(object))

    clusters = pd.Series([int(x) for x in labels], index=pd.Index(ids), name="Cluster")
    clusters.index.name = method.id

    postprob = None
    try:
        r_pp = r_latrend.postprob(r_model)
        pp_py = pandas2ri.rpy2py(r_pp)
        if isinstance(pp_py, pd.DataFrame):
            postprob = pp_py
    except Exception:
        postprob = None

    if postprob is None:
        postprob = postprobFromAssignments(clusters)

    model = LCModel(method=method, data=data, clusters=clusters, postprob=postprob)
    model.meta["r"] = {"model": r_model, "method": r_method, "method_name": method.r_method}
    return model

