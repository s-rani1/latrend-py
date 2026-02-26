from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..core.method import LCMethod


@dataclass
class RLCMethod(LCMethod):
    """
    Placeholder method that delegates fitting/clustering to the upstream R latrend package via rpy2.

    This enables API-level parity while pure-Python ports of individual methods are implemented.
    """

    r_method: str = ""
    r_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.r_method:
            raise ValueError("r_method must be provided")
        if self.name == "Noname":
            self.name = self.r_method

    def prepareData(self, data: pd.DataFrame) -> pd.DataFrame:
        # Keep all columns intact; only drop rows missing core columns.
        df = data.copy()
        self.validate(df)
        df = df.dropna(subset=[self.id, self.time, self.outcome])
        return df

    def cluster(self, data: pd.DataFrame):
        raise RuntimeError(
            "RLCMethod cannot be clustered directly. Use latrendCluster(method, data), "
            "which will delegate to the R backend when rpy2/R/latrend are available."
        )


def lcMethodR(
    r_method: str,
    *,
    id: str = "Id",
    time: str = "Time",
    outcome: str = "Y",
    name: str | None = None,
    transform=None,
    **kwargs: Any,
) -> RLCMethod:
    """
    Create an R-backed lcMethod by name, e.g.
    lcMethodR("lcMethodLcmmGMM", formula="Y~Time", nClusters=3).
    """

    r_args = dict(kwargs)
    # Most upstream constructors accept these.
    r_args.setdefault("id", id)
    r_args.setdefault("time", time)
    r_args.setdefault("outcome", outcome)
    if name is not None:
        r_args.setdefault("name", name)

    return RLCMethod(
        id=id,
        time=time,
        outcome=outcome,
        transform=transform,
        name=name or r_method,
        r_method=r_method,
        r_args=r_args,
    )
