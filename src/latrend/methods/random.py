from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.method import LCMethod


@dataclass
class lcMethodRandom(LCMethod):
    nClusters: int = 2
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.nClusters < 1:
            raise ValueError("nClusters must be >= 1")
        if self.name == "Noname":
            self.name = "Random"

    def cluster(self, data: pd.DataFrame) -> pd.Series:
        rng = np.random.default_rng(self.seed)
        ids = pd.Index(data[self.id].unique())
        clusters = rng.integers(1, self.nClusters + 1, size=len(ids))
        return pd.Series(clusters, index=ids, name="Cluster")

