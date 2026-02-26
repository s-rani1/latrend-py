"""
Class proportion and posterior probability plots for latrend-py.

Mirrors R latrend's plotClassProportions() and related posterior probability
visualisations, with theme_light() styling and the ggplot2 default palette.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from ..core.model import LCModel
from ._backend import choose_backend, require_plotnine
from ._mpl import get_axes
from ._theme import (
    _cluster_colors,
    apply_mpl_theme,
    scale_fill_latrend,
    theme_latrend,
)


def plotClassProportions(
    model: LCModel,
    *,
    ax: Axes | None = None,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Bar plot of class proportions.

    Mirrors R's output where each cluster is coloured with the ggplot2
    default palette and bars show relative sizes.
    """

    props = model.classProportions()
    n_clusters = len(props)
    backend = choose_backend(backend)

    if backend == "plotnine":
        p9 = require_plotnine()
        df = pd.DataFrame({
            "Cluster": [str(k) for k in props.index],
            "Proportion": props.to_numpy(),
        })
        df["Cluster"] = pd.Categorical(df["Cluster"], categories=df["Cluster"].tolist())
        p = (
            p9.ggplot(df, p9.aes(x="Cluster", y="Proportion", fill="Cluster"))
            + p9.geom_col(show_legend=False, width=0.7)
            + scale_fill_latrend(n_clusters)
            + p9.labs(x="Cluster", y="Proportion")
            + p9.scale_y_continuous(limits=(0, 1))
            + theme_latrend(figure_size=figure_size, base_size=base_size)
        )
        return p

    # matplotlib
    ax = get_axes(ax)
    colors = _cluster_colors(n_clusters)
    labels = [str(i) for i in props.index]
    ax.bar(labels, props.to_numpy(), color=colors, width=0.7)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    apply_mpl_theme(ax, base_size=base_size)
    return ax


def plotClassProbabilities(
    model: LCModel,
    *,
    ax: Axes | None = None,
    bins: int = 20,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Histogram of posterior probabilities (one distribution per class).

    Mirrors the R package's posterior probability distribution plot.
    """

    if model.postprob is None:
        raise ValueError("Model has no postprob matrix")

    n_clusters = model.nClusters()
    backend = choose_backend(backend)

    if backend == "plotnine":
        p9 = require_plotnine()
        pp_long = model.postprob.reset_index(drop=True).melt(
            var_name="Cluster", value_name="Posterior"
        )
        pp_long["Cluster"] = pp_long["Cluster"].astype(str)
        pp_long["Cluster"] = pd.Categorical(pp_long["Cluster"])
        p = (
            p9.ggplot(pp_long, p9.aes(x="Posterior", fill="Cluster"))
            + p9.geom_histogram(bins=bins, alpha=0.45, position="identity")
            + scale_fill_latrend(n_clusters)
            + p9.labs(
                x="Posterior probability",
                y="Count",
                fill="Cluster",
            )
            + theme_latrend(figure_size=figure_size, base_size=base_size)
        )
        return p

    # matplotlib
    ax = get_axes(ax)
    pp = model.postprob.to_numpy(dtype=float)
    colors = _cluster_colors(n_clusters)
    for j in range(n_clusters):
        ax.hist(
            pp[:, j], bins=bins, alpha=0.45,
            label=f"Cluster {j + 1}", color=colors[j],
        )
    ax.set_xlabel("Posterior probability")
    ax.set_ylabel("Count")
    ax.legend(title="Cluster", fontsize=base_size - 2, title_fontsize=base_size - 1)
    apply_mpl_theme(ax, base_size=base_size)
    return ax
