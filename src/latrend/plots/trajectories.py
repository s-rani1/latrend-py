"""
Trajectory plotting functions for latrend-py.

Closely mirrors the R latrend package's plotTrajectories() and
plotClusterTrajectories() functions, including:
  - ggplot2 theme_light() styling
  - ggplot2 default discrete colour palette (evenly spaced HCL hues)
  - Faceted cluster views
  - Ribbon-based confidence intervals
  - Trajectory overlay on cluster mean plots
  - Cluster proportion labels in legends
"""

from __future__ import annotations

from typing import Any, Callable

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from ..core.model import LCModel
from ._backend import choose_backend, require_plotnine
from ._mpl import get_axes
from ._theme import (
    LATREND_PALETTE,
    _cluster_colors,
    apply_mpl_theme,
    scale_color_latrend,
    scale_fill_latrend,
    theme_latrend,
)


# ---------------------------------------------------------------------------
# Cluster label helpers (mirrors R make.clusterPropLabels / make.clusterSizeLabels)
# ---------------------------------------------------------------------------

def make_clusterPropLabels(model: LCModel) -> dict:
    """Return {cluster_id: 'A (42.0%)'} labels like R's make.clusterPropLabels."""
    props = model.classProportions()
    labels = {}
    for i, (k, v) in enumerate(props.items()):
        letter = chr(ord("A") + i)
        labels[k] = f"{letter} ({v:.0%})"
    return labels


def make_clusterSizeLabels(model: LCModel) -> dict:
    """Return {cluster_id: 'A (n=15)'} labels like R's make.clusterSizeLabels."""
    counts = model.classCounts()
    labels = {}
    for i, (k, v) in enumerate(counts.items()):
        letter = chr(ord("A") + i)
        labels[k] = f"{letter} (n={int(v)})"
    return labels


# ---------------------------------------------------------------------------
# plotTrajectories
# ---------------------------------------------------------------------------

def plotTrajectories(
    x: pd.DataFrame | LCModel,
    *,
    response: str | None = None,
    ax: Axes | None = None,
    byCluster: bool = True,
    alpha: float = 0.25,
    linewidth: float = 0.6,
    color: str | None = None,
    nSample: int | None = None,
    seed: int | None = None,
    facet: bool = False,
    clusterLabeler: Callable[[LCModel], dict] | None = None,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Plot individual trajectories (spaghetti plot).

    Mirrors R's ``plotTrajectories(data, response = "Y")`` and
    ``plotTrajectories(model)``.

    Parameters
    ----------
    x : DataFrame or LCModel
        Long-format data or a fitted model.
    response : str, optional
        Response column name (overrides model's outcome).
    byCluster : bool
        If *True* and clusters are available, colour trajectories by cluster.
    alpha : float
        Line opacity (0-1).
    linewidth : float
        Line width in pt.
    color : str, optional
        Single colour when ``byCluster=False``. Defaults to first palette entry.
    nSample : int, optional
        Random sample of trajectories to display.
    facet : bool
        Facet by cluster (separate panels per cluster).
    clusterLabeler : callable, optional
        Function taking an LCModel and returning ``{cluster_id: label}`` dict.
    """

    backend = choose_backend(backend)
    if color is None:
        color = LATREND_PALETTE[0]

    # ---- Prepare data ----
    if isinstance(x, LCModel):
        df = x.data.copy()
        id_col = x.method.id
        time_col = x.method.time
        y_col = response or x.method.outcome
        if byCluster:
            # Avoid Cluster_x/Cluster_y after merge when source data already has a Cluster column.
            if "Cluster" in df.columns:
                df = df.drop(columns=["Cluster"])
            cluster_map = x.clusters.rename("Cluster").reset_index()
            cluster_map = cluster_map.rename(columns={"index": id_col})
            df = df.merge(cluster_map, on=id_col, how="inner")
            if clusterLabeler is None:
                # Match R-style defaults for model-based cluster labels.
                clusterLabeler = make_clusterPropLabels
            if clusterLabeler is not None:
                label_map = clusterLabeler(x)
                df["Cluster"] = df["Cluster"].map(label_map).fillna(df["Cluster"].astype(str))
    else:
        df = x.copy()
        id_col = "Id"
        time_col = "Time"
        y_col = response or "Y"

    # ---- Sample ----
    if nSample is not None:
        ids = pd.Index(df[id_col].unique())
        if nSample < len(ids):
            rng = np.random.default_rng(seed)
            keep = pd.Index(rng.choice(ids.to_numpy(), size=int(nSample), replace=False))
            df = df[df[id_col].isin(keep)]

    # ================================================================
    # Plotnine backend
    # ================================================================
    if backend == "plotnine":
        p9 = require_plotnine()

        has_cluster = byCluster and "Cluster" in df.columns
        if has_cluster:
            df["Cluster"] = pd.Categorical(df["Cluster"])
            n_clusters = df["Cluster"].nunique()
            p = (
                p9.ggplot(df, p9.aes(x=time_col, y=y_col, group=id_col, color="Cluster"))
                + p9.geom_line(alpha=alpha, size=linewidth)
                + scale_color_latrend(n_clusters)
                + p9.labs(x=time_col, y=y_col, color="Cluster")
                + theme_latrend(figure_size=figure_size, base_size=base_size)
            )
            if facet:
                facet_kwargs = {"ncol": 2} if n_clusters == 4 else {}
                p = p + p9.facet_wrap("~Cluster", **facet_kwargs) + p9.theme(legend_position="none")
        else:
            p = (
                p9.ggplot(df, p9.aes(x=time_col, y=y_col, group=id_col))
                + p9.geom_line(alpha=alpha, size=linewidth, color=color)
                + p9.labs(x=time_col, y=y_col)
                + theme_latrend(figure_size=figure_size, base_size=base_size)
            )
        return p

    # ================================================================
    # Matplotlib backend
    # ================================================================
    ax = get_axes(ax)

    has_cluster = byCluster and "Cluster" in df.columns
    if has_cluster:
        clusters = sorted(df["Cluster"].unique(), key=str)
        n_clusters = len(clusters)
        colors = _cluster_colors(n_clusters)
        color_map = {c: colors[i] for i, c in enumerate(clusters)}
        used = set()
        for (_, _cluster), g in df.groupby([id_col, "Cluster"], sort=False):
            ax.plot(
                g[time_col], g[y_col],
                alpha=alpha, linewidth=linewidth,
                color=color_map.get(_cluster, color),
                label=str(_cluster) if _cluster not in used else None,
            )
            used.add(_cluster)
        ax.legend(title="Cluster", fontsize=base_size - 2, title_fontsize=base_size - 1)
    else:
        for _, g in df.groupby(id_col, sort=False):
            ax.plot(g[time_col], g[y_col], alpha=alpha, linewidth=linewidth, color=color)

    ax.set_xlabel(time_col)
    ax.set_ylabel(y_col)
    apply_mpl_theme(ax, base_size=base_size)
    return ax


# ---------------------------------------------------------------------------
# plotClusterTrajectories
# ---------------------------------------------------------------------------

def plotClusterTrajectories(
    x: pd.DataFrame | LCModel,
    *,
    response: str | None = None,
    cluster: str = "Cluster",
    ax: Axes | None = None,
    ci: bool = False,
    trajectories: bool | str = False,
    linewidth: float = 1.5,
    traj_alpha: float = 0.15,
    traj_linewidth: float = 0.4,
    facet: bool | None = None,
    clusterLabeler: Callable[[LCModel], dict] | None = None,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Plot per-cluster mean trajectories.

    Mirrors R's ``plotClusterTrajectories(model)`` with ``trajectories``,
    ``facet``, and ``ci``/ribbon options.

    Parameters
    ----------
    x : DataFrame or LCModel
        Either a fitted LCModel or a DataFrame with a cluster column.
    response : str, optional
        Name of the response column (default: model's outcome or "Y").
    cluster : str
        Column name with cluster assignments (for DataFrame input).
    ci : bool
        Show 95 % confidence ribbon (mean +/- 1.96 SEM).
    trajectories : bool or str
        If *True*, overlay individual trajectories behind the cluster means.
        String values ``"sd"``, ``"se"``, ``"range"`` control ribbon type.
    facet : bool or None
        Facet by cluster. Default: ``True`` when ``trajectories`` is truthy.
    clusterLabeler : callable, optional
        Function returning ``{cluster_id: label}`` mapping.
    """

    backend = choose_backend(backend)

    # ---- Resolve facet default ----
    if facet is None:
        facet = bool(trajectories)

    # ---- Build long DataFrame with Cluster column ----
    if isinstance(x, LCModel):
        df = x.data.copy()
        id_col = x.method.id
        time_col = x.method.time
        y_col = response or x.method.outcome
        if "Cluster" in df.columns:
            df = df.drop(columns=["Cluster"])
        cluster_map = x.clusters.rename("Cluster").reset_index()
        cluster_map = cluster_map.rename(columns={"index": id_col})
        df = df.merge(cluster_map, on=id_col, how="inner")
        model = x
    else:
        df = x.copy()
        id_col = "Id"
        time_col = "Time"
        y_col = response or "Y"
        if cluster in df.columns and "Cluster" not in df.columns:
            df = df.rename(columns={cluster: "Cluster"})
        model = None

    # ---- Cluster labels ----
    if model is not None and clusterLabeler is None:
        # Matches latrend's default: show A/B/C… with class proportions.
        clusterLabeler = make_clusterPropLabels
    if clusterLabeler is not None and model is not None:
        label_map = clusterLabeler(model)
        df["Cluster"] = df["Cluster"].map(label_map).fillna(df["Cluster"].astype(str))

    # ---- Compute summary stats ----
    grouped = df.groupby(["Cluster", time_col], sort=True)[y_col]
    summary = grouped.mean().rename("mean").reset_index()

    # Ribbon data
    ribbon_type = None
    if ci or (isinstance(trajectories, str) and trajectories in ("se", "sd", "range")):
        ribbon_type = trajectories if isinstance(trajectories, str) else "se"

    if ribbon_type == "se" or ci:
        sem = grouped.sem().rename("sem").reset_index()
        summary = summary.merge(sem, on=["Cluster", time_col], how="left")
        summary["ymin"] = summary["mean"] - 1.96 * summary["sem"]
        summary["ymax"] = summary["mean"] + 1.96 * summary["sem"]
    elif ribbon_type == "sd":
        sd = grouped.std().rename("sd").reset_index()
        summary = summary.merge(sd, on=["Cluster", time_col], how="left")
        summary["ymin"] = summary["mean"] - summary["sd"]
        summary["ymax"] = summary["mean"] + summary["sd"]
    elif ribbon_type == "range":
        rmin = grouped.min().rename("ymin").reset_index()
        rmax = grouped.max().rename("ymax").reset_index()
        summary = summary.merge(rmin, on=["Cluster", time_col], how="left")
        summary = summary.merge(rmax, on=["Cluster", time_col], how="left")

    has_ribbon = "ymin" in summary.columns

    # ---- Cluster ordering ----
    clusters = sorted(summary["Cluster"].unique(), key=str)
    n_clusters = len(clusters)

    summary["Cluster"] = pd.Categorical(summary["Cluster"], categories=clusters)
    df["Cluster"] = pd.Categorical(df["Cluster"], categories=clusters)

    # ================================================================
    # Plotnine backend
    # ================================================================
    if backend == "plotnine":
        p9 = require_plotnine()

        # Overlay individual trajectories
        show_traj = bool(trajectories)
        p = p9.ggplot()
        if show_traj:
            p = p + p9.geom_line(
                data=df,
                mapping=p9.aes(x=time_col, y=y_col, group=id_col, color="Cluster"),
                alpha=traj_alpha,
                size=traj_linewidth,
                show_legend=not facet,
                inherit_aes=False,
            )

        # Ribbon
        if has_ribbon:
            p = p + p9.geom_ribbon(
                data=summary,
                mapping=p9.aes(x=time_col, ymin="ymin", ymax="ymax", fill="Cluster", group="Cluster"),
                alpha=0.2,
                color="none",
                show_legend=not facet,
                inherit_aes=False,
            )

        # Mean line
        if facet:
            # Match the R vignette output: coloured assigned trajectories + black mean per panel.
            p = p + p9.geom_line(
                data=summary,
                mapping=p9.aes(x=time_col, y="mean", group=1),
                color="black",
                size=linewidth,
                inherit_aes=False,
            )
        else:
            p = p + p9.geom_line(
                data=summary,
                mapping=p9.aes(x=time_col, y="mean", color="Cluster", group="Cluster"),
                size=linewidth,
                inherit_aes=False,
            )

        p = p + scale_color_latrend(n_clusters)
        if has_ribbon:
            p = p + scale_fill_latrend(n_clusters)

        p = p + p9.labs(title="Cluster trajectories", x=time_col, y=y_col, color="Cluster", fill="Cluster")
        p = p + theme_latrend(figure_size=figure_size, base_size=base_size)

        if facet:
            facet_kwargs = {"ncol": 2} if n_clusters == 4 else {}
            p = p + p9.facet_wrap("~Cluster", **facet_kwargs) + p9.theme(legend_position="none")

        return p

    # ================================================================
    # Matplotlib backend
    # ================================================================
    ax = get_axes(ax)
    colors = _cluster_colors(n_clusters)
    color_map = {c: colors[i] for i, c in enumerate(clusters)}

    show_traj = bool(trajectories)

    for c in clusters:
        clr = color_map[c]
        sub = summary[summary["Cluster"] == c]

        # Overlay individual trajectories
        if show_traj:
            traj_df = df[df["Cluster"] == c]
            for _, g in traj_df.groupby(id_col, sort=False):
                ax.plot(g[time_col], g[y_col], color=clr, alpha=traj_alpha, linewidth=traj_linewidth)

        # Ribbon
        if has_ribbon:
            ax.fill_between(
                sub[time_col], sub["ymin"], sub["ymax"],
                color=clr, alpha=0.2,
            )

        # Mean line
        ax.plot(sub[time_col], sub["mean"], color=clr, linewidth=linewidth, label=str(c))

    ax.set_xlabel(time_col)
    ax.set_ylabel(y_col)
    ax.legend(title="Cluster", fontsize=base_size - 2, title_fontsize=base_size - 1)
    apply_mpl_theme(ax, base_size=base_size)
    return ax


# ---------------------------------------------------------------------------
# plotFittedTrajectories (convenience alias)
# ---------------------------------------------------------------------------

def plotFittedTrajectories(
    model: LCModel,
    *,
    ax: Axes | None = None,
    ci: bool = False,
    trajectories: bool | str = False,
    linewidth: float = 1.5,
    facet: bool | None = None,
    clusterLabeler: Callable[[LCModel], dict] | None = None,
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Any:
    """
    Compatibility helper: for methods without an explicit fitted trajectory model,
    this falls back to plotting per-cluster mean trajectories.
    """

    return plotClusterTrajectories(
        model,
        ax=ax,
        ci=ci,
        trajectories=trajectories,
        linewidth=linewidth,
        facet=facet,
        clusterLabeler=clusterLabeler,
        backend=backend,
        figure_size=figure_size,
        base_size=base_size,
    )
