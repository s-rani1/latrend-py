from __future__ import annotations

from typing import Any

import pandas as pd

from .method import LCMethod


def trajectories(method: LCMethod | None, data: pd.DataFrame) -> dict[Any, pd.DataFrame]:
    """
    Split long-format data into per-individual trajectories.
    """

    if method is None:
        id_col, time_col, outcome_col = "Id", "Time", "Y"
    else:
        id_col, time_col, outcome_col = method.id, method.time, method.outcome

    if not all(c in data.columns for c in (id_col, time_col, outcome_col)):
        raise ValueError("Data does not contain required columns for trajectories()")

    df = data[[id_col, time_col, outcome_col]].copy()
    df = df.dropna(subset=[id_col, time_col, outcome_col])
    df = df.sort_values([id_col, time_col], kind="mergesort")
    return {k: v.rename(columns={id_col: "Id", time_col: "Time", outcome_col: "Y"}).reset_index(drop=True) for k, v in df.groupby(id_col, sort=False)}

