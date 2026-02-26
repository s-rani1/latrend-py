"""
Data simulation utilities for latrend-py.

Provides:
  - ``generateTrajectories`` – wide matrix + true labels (low-level).
  - ``generateData``         – dict wrapper.
  - ``generateLongData``     – long-format DataFrame with Id/Time/Y/Cluster.
  - ``latrendData``          – a built-in dataset that closely approximates the
    ``latrendData`` shipped with the R package (200 individuals, 10 time points,
    3 ground-truth clusters with distinct intercepts and slopes plus random
    effects and noise).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrajectorySpec:
    mean: float
    sd: float


def generateTrajectories(
    nIndividuals: int = 100,
    nTime: int = 10,
    means: list[float] | None = None,
    sd: float = 1.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate simulated trajectories (wide matrix) + true cluster labels.

    Roughly matches latrend::generateTrajectories() / generateData().
    """

    rng = np.random.default_rng(seed)
    if means is None:
        means = [0.0, 2.0, 4.0]
    nClusters = len(means)

    ids = np.arange(1, nIndividuals + 1)
    true = rng.integers(1, nClusters + 1, size=nIndividuals)

    time = np.arange(1, nTime + 1)
    y = np.empty((nIndividuals, nTime), dtype=float)
    for i in range(nIndividuals):
        y[i, :] = rng.normal(loc=means[true[i] - 1], scale=sd, size=nTime)

    mat = pd.DataFrame(y, index=ids, columns=time)
    mat.index.name = "Id"
    return mat, pd.Series(true, index=ids, name="Cluster")


def generateData(
    nIndividuals: int = 100,
    nTime: int = 10,
    nClusters: int = 3,
    seed: int | None = None,
) -> dict[str, object]:
    """
    Convenience wrapper returning a dict with (data, clusters), similar to R examples.
    """

    means = list(np.linspace(0.0, 4.0, nClusters))
    mat, clusters = generateTrajectories(
        nIndividuals=nIndividuals, nTime=nTime, means=means, sd=1.0, seed=seed
    )
    return {"data": mat, "clusters": clusters}


def generateLongData(
    nIndividuals: int = 100,
    nTime: int = 10,
    nClusters: int = 3,
    sd: float = 1.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate simulated trajectories in long format with columns Id/Time/Y.
    """

    means = list(np.linspace(0.0, 4.0, nClusters))
    mat, clusters = generateTrajectories(
        nIndividuals=nIndividuals, nTime=nTime, means=means, sd=sd, seed=seed
    )
    df = (
        mat.reset_index()
        .melt(id_vars=["Id"], var_name="Time", value_name="Y")
        .merge(clusters.reset_index().rename(columns={"index": "Id"}), on="Id", how="left")
    )
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Built-in dataset (approximation of R's latrendData)
# ---------------------------------------------------------------------------

def latrendData(seed: int = 1) -> pd.DataFrame:
    """
    A built-in synthetic longitudinal dataset that closely approximates the
    ``latrendData`` dataset shipped with the upstream R package.

    The R dataset contains 200 trajectories observed at 10 equally-spaced
    time points (0..9), with three latent classes that differ in both
    intercept and slope, plus per-individual random intercepts/slopes
    and observation-level noise.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns ``Id``, ``Time``, ``Y``, ``Class``.
        The ``Class`` column gives the ground-truth cluster assignment
        (values 1, 2, 3) -- matching the R column name.
    """

    rng = np.random.default_rng(seed)
    n = 200
    time_points = np.arange(0, 10, dtype=float)

    # Cluster parameters (intercept, slope) -- roughly matching R's latrendData
    cluster_params = {
        1: {"intercept": -1.0, "slope":  0.5},
        2: {"intercept":  1.5, "slope": -0.2},
        3: {"intercept":  0.0, "slope":  0.1},
    }

    # Balanced assignment
    labels = np.array([1] * 67 + [2] * 67 + [3] * 66)
    rng.shuffle(labels)

    rows = []
    for i, cl in enumerate(labels, start=1):
        p = cluster_params[cl]
        # Random effects
        ri = rng.normal(0, 0.5)      # random intercept
        rs = rng.normal(0, 0.05)     # random slope
        for t in time_points:
            noise = rng.normal(0, 0.3)
            y = (p["intercept"] + ri) + (p["slope"] + rs) * t + noise
            rows.append({"Id": i, "Time": t, "Y": y, "Class": cl})

    df = pd.DataFrame(rows)
    df["Id"] = df["Id"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["Y"] = df["Y"].astype(float)
    df["Class"] = df["Class"].astype(int)
    return df
