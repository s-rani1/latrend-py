"""
Metric plots for latrend-py.

Mirrors R latrend's plotMetric() function, used for the elbow method
(plotting a metric like silhouette score across different values of k).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from matplotlib.axes import Axes

from ..core.model import LCModels
from ..metrics.cluster import silhouette_score_long
from ._backend import choose_backend, require_plotnine
from ._mpl import get_axes
from ._theme import LATREND_PALETTE, apply_mpl_theme, theme_latrend


def plotMetric(
    models: LCModels,
    metric: str | list[str] = "silhouette",
    *,
    ax: Axes | None = None,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Plot a metric (e.g. silhouette) across a collection of models.

    Mirrors R's ``plotMetric(models, "ASW")`` — a line + point plot of
    the selected metric vs. number of clusters.
    """

    backend = choose_backend(backend)

    metrics = [metric] if isinstance(metric, str) else list(metric)
    if not metrics:
        raise ValueError("metric must be a non-empty string or list of strings")

    rows: list[dict[str, Any]] = []
    xs = []
    for model in models:
        k = model.nClusters()
        xs.append(k)
        for m in metrics:
            value = model.meta.get("metrics", {}).get(m)
            if value is None and m == "silhouette":
                value = silhouette_score_long(model)
                model.meta.setdefault("metrics", {})[m] = value
            rows.append({"Number of clusters": k, "Metric": m, "Value": value})

    if backend == "plotnine":
        p9 = require_plotnine()
        df = pd.DataFrame(rows)
        if len(metrics) == 1:
            only = metrics[0]
            sub = df[df["Metric"] == only]
            p = (
                p9.ggplot(sub, p9.aes(x="Number of clusters", y="Value"))
                + p9.geom_line(color=LATREND_PALETTE[0], size=0.8)
                + p9.geom_point(color=LATREND_PALETTE[0], size=3)
                + p9.labs(x="Number of clusters", y=only)
                + p9.scale_x_continuous(breaks=xs)
                + theme_latrend(figure_size=figure_size, base_size=base_size)
            )
        else:
            p = (
                p9.ggplot(df, p9.aes(x="Number of clusters", y="Value"))
                + p9.geom_line(color=LATREND_PALETTE[0], size=0.8)
                + p9.geom_point(color=LATREND_PALETTE[0], size=3)
                + p9.labs(x="Number of clusters", y="Value")
                + p9.scale_x_continuous(breaks=xs)
                + p9.facet_wrap("~Metric", scales="free_y")
                + theme_latrend(figure_size=figure_size, base_size=base_size)
            )
        return p

    # matplotlib
    ax = get_axes(ax)
    df = pd.DataFrame(rows)
    for i, m in enumerate(metrics):
        sub = df[df["Metric"] == m]
        ax.plot(
            sub["Number of clusters"], sub["Value"],
            marker="o",
            color=LATREND_PALETTE[i % len(LATREND_PALETTE)],
            linewidth=0.8,
            markersize=5,
            label=m if len(metrics) > 1 else None,
        )
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel(metrics[0] if len(metrics) == 1 else "Value")
    ax.set_xticks(xs)
    if len(metrics) > 1:
        ax.legend(title="Metric", fontsize=base_size - 2, title_fontsize=base_size - 1)
    apply_mpl_theme(ax, base_size=base_size)
    return ax
