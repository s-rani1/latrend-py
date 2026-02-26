from __future__ import annotations

from typing import Any

import pandas as pd


def tsmatrixToFrame(x: pd.DataFrame, id: str = "Id", time: str = "Time", y: str = "Y") -> pd.DataFrame:
    """
    Convert a wide trajectory matrix (index=Id, columns=Time) to long-format frame.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError("tsmatrixToFrame expects a pandas.DataFrame")
    if x.index.name is None:
        ids = x.index.astype(str)
    else:
        ids = x.index
    out = (
        x.copy()
        .assign(**{id: ids})
        .melt(id_vars=[id], var_name=time, value_name=y)
        .dropna(subset=[y])
    )
    return out


def tsframeToMatrix(x: pd.DataFrame, id: str = "Id", time: str = "Time", y: str = "Y") -> pd.DataFrame:
    """
    Convert a long-format trajectory frame to wide matrix.
    """

    if not isinstance(x, pd.DataFrame):
        raise TypeError("tsframeToMatrix expects a pandas.DataFrame")
    for col in (id, time, y):
        if col not in x.columns:
            raise ValueError(f"Missing column '{col}'")

    mat = x.pivot_table(index=id, columns=time, values=y, aggfunc="first")
    # Preserve original column order if time is sortable
    try:
        mat = mat.reindex(sorted(mat.columns), axis=1)
    except Exception:
        pass
    return mat


def tsmatrix(x: Any, id: str = "Id", time: str = "Time", y: str = "Y") -> pd.DataFrame:
    # Avoid import cycle at module import time
    try:
        from .model import LCModel  # noqa: WPS433
    except Exception:  # pragma: no cover
        LCModel = None  # type: ignore[misc,assignment]

    if LCModel is not None and isinstance(x, LCModel):
        return tsmatrix(x.data, id=x.method.id, time=x.method.time, y=x.method.outcome)
    if isinstance(x, pd.DataFrame):
        if id in x.columns and time in x.columns and y in x.columns:
            return tsframeToMatrix(x, id=id, time=time, y=y)
        # Otherwise assume it's already a wide matrix
        return x
    raise TypeError("tsmatrix expects a pandas.DataFrame (wide or long)")


def tsframe(x: Any, id: str = "Id", time: str = "Time", y: str = "Y") -> pd.DataFrame:
    # Avoid import cycle at module import time
    try:
        from .model import LCModel  # noqa: WPS433
    except Exception:  # pragma: no cover
        LCModel = None  # type: ignore[misc,assignment]

    if LCModel is not None and isinstance(x, LCModel):
        return tsframe(x.data, id=x.method.id, time=x.method.time, y=x.method.outcome)
    if isinstance(x, pd.DataFrame):
        if id in x.columns and time in x.columns and y in x.columns:
            return x
        return tsmatrixToFrame(x, id=id, time=time, y=y)
    raise TypeError("tsframe expects a pandas.DataFrame (wide or long)")
