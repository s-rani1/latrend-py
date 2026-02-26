from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.method import LCMethod
from ..core.trajectories import trajectories


def _parse_simple_formula(formula: str) -> tuple[str, list[tuple[str, int]]]:
    """
    Very small subset of R-like formulas.

    Supports: "Y ~ Time", "Y ~ Time + Time^2", "Y ~ poly(Time,3)".
    Returns (lhs, [(var, power), ...]) excluding intercept (always included).
    """

    if "~" not in formula:
        raise ValueError("Formula must contain '~', e.g. 'Y ~ Time'")
    lhs_raw, rhs_raw = formula.split("~", 1)
    lhs = lhs_raw.strip()
    rhs = rhs_raw.strip()

    terms: list[tuple[str, int]] = []
    for term in [t.strip() for t in rhs.split("+") if t.strip()]:
        if term in {"1"}:
            continue
        if term in {"0", "-1"}:
            raise ValueError("Intercept removal ('0' or '-1') is not supported yet")
        if term.startswith("poly(") and term.endswith(")"):
            inner = term[len("poly(") : -1]
            var_part, deg_part = [p.strip() for p in inner.split(",", 1)]
            deg = int(deg_part)
            terms.extend([(var_part, p) for p in range(1, deg + 1)])
            continue
        if "^" in term:
            var, pow_raw = [p.strip() for p in term.split("^", 1)]
            terms.append((var, int(pow_raw)))
            continue
        terms.append((term, 1))
    if not terms:
        raise ValueError("No RHS terms parsed from formula")
    return lhs, terms


@dataclass
class lcMethodLMKM(LCMethod):
    formula: str = "Y ~ Time"
    nClusters: int = 2
    seed: int | None = None
    kmeans_kwargs: dict | None = None

    def __post_init__(self) -> None:
        if self.nClusters < 1:
            raise ValueError("nClusters must be >= 1")
        if self.name == "Noname":
            self.name = "LMKM"

    def cluster(self, data: pd.DataFrame) -> pd.Series:
        y_name, terms = _parse_simple_formula(self.formula)
        if y_name != self.outcome:
            # Keep outcome consistent with method slot; allow users to omit.
            pass

        trajs = trajectories(self, data)
        ids = list(trajs.keys())

        coef_rows = []
        for _id in ids:
            df = trajs[_id]
            y = df["Y"].to_numpy(dtype=float)
            X_cols = [np.ones_like(y)]
            for var, power in terms:
                if var != "Time":
                    raise ValueError("Only Time is supported as predictor in the Python LMKM port.")
                x = df["Time"].to_numpy(dtype=float) ** power
                X_cols.append(x)
            X = np.column_stack(X_cols)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            coef_rows.append(coef)
        coef_mat = np.vstack(coef_rows)

        from sklearn.cluster import KMeans

        kwargs = dict(n_init=10, random_state=self.seed)
        if self.kmeans_kwargs:
            kwargs.update(self.kmeans_kwargs)
        km = KMeans(n_clusters=self.nClusters, **kwargs)
        labels0 = km.fit_predict(coef_mat)
        labels = labels0.astype(int) + 1
        return pd.Series(labels, index=pd.Index(ids), name="Cluster")

