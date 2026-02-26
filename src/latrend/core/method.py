from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


@dataclass
class LCMethod:
    """
    Base class for latrend clustering methods.

    Mirrors the lifecycle in the R package:
    validate -> prepareData -> preFit -> fit -> postFit -> cluster
    """

    id: str = "Id"
    time: str = "Time"
    outcome: str = "Y"
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    name: str = "Noname"
    _is_fitted: bool = field(default=False, init=False, repr=False)

    def validate(self, data: pd.DataFrame) -> None:
        missing = [c for c in (self.id, self.time, self.outcome) if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def prepareData(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        self.validate(df)
        df = df[[self.id, self.time, self.outcome] + [c for c in df.columns if c not in (self.id, self.time, self.outcome)]]
        df = df.dropna(subset=[self.id, self.time, self.outcome])
        df[self.time] = pd.to_numeric(df[self.time], errors="coerce")
        df[self.outcome] = pd.to_numeric(df[self.outcome], errors="coerce")
        df = df.dropna(subset=[self.time, self.outcome])

        df = df.sort_values([self.id, self.time], kind="mergesort").reset_index(drop=True)
        if self.transform is not None:
            df = self.transform(df)
        return df

    def preFit(self, data: pd.DataFrame) -> None:
        return None

    def fit(self, data: pd.DataFrame) -> None:
        self._is_fitted = True

    def postFit(self, data: pd.DataFrame) -> None:
        return None

    def cluster(self, data: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def isFitted(self) -> bool:
        return self._is_fitted

    def getName(self) -> str:
        return self.name

    def getParams(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "time": self.time,
            "outcome": self.outcome,
            "name": self.name,
        }

