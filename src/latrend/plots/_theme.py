"""
Plotting theme for latrend-py.

Mirrors ggplot2 defaults used in the latrend R package vignettes:
  - theme_gray() styling
  - default discrete hue palette (scales::hue_pal / grDevices::hcl)

Both plotnine and matplotlib backends are supported so that plots look
consistent regardless of the available rendering engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._backend import require_plotnine

# ---------------------------------------------------------------------------
# ggplot2 default discrete colour palette (evenly-spaced HCL hues)
# ---------------------------------------------------------------------------
# ggplot2::scale_colour_discrete() uses scales::hue_pal(h = c(15, 375), c = 100, l = 65).
# Note: the palette depends on *n* (it is re-computed to evenly space hues).


def _hcl_to_hex(h: np.ndarray, *, c: float = 100.0, l: float = 65.0) -> list[str]:
    """
    Convert polar LUV (HCL) to sRGB hex.

    Implements the same colour model used by R's grDevices::hcl() (fixup=TRUE),
    which underpins ggplot2's default discrete hue palette.
    """

    h = np.asarray(h, dtype=float)

    # Reference white (D65, 2°) used for sRGB.
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    denom = Xn + 15 * Yn + 3 * Zn
    un = (4 * Xn) / denom
    vn = (9 * Yn) / denom

    # L*u*v* (u*, v* from chroma and hue)
    hr = np.deg2rad(h)
    u = c * np.cos(hr)
    v = c * np.sin(hr)

    L = float(l)
    if L == 0.0:
        X = np.zeros_like(u)
        Y = np.zeros_like(u)
        Z = np.zeros_like(u)
    else:
        up = u / (13 * L) + un
        vp = v / (13 * L) + vn

        # L* -> Y (relative)
        if L > 8:
            Y = Yn * (((L + 16) / 116) ** 3)
        else:
            Y = Yn * (L / 903.3)

        X = Y * (9 * up) / (4 * vp)
        Z = Y * (12 - 3 * up - 20 * vp) / (4 * vp)

    # XYZ -> linear RGB (sRGB, D65)
    M = np.array(
        [
            [3.2406, -1.5372, -0.4986],
            [-0.9689, 1.8758, 0.0415],
            [0.0557, -0.2040, 1.0570],
        ]
    )
    XYZ = np.stack([X, np.full_like(X, Y), Z], axis=0)
    rgb_lin = (M @ XYZ.reshape(3, -1)).reshape(3, *X.shape)

    # Gamma correction + clamp ("fixup")
    a = 0.055
    rgb_lin = np.clip(rgb_lin, 0, None)
    rgb = np.where(
        rgb_lin <= 0.0031308,
        12.92 * rgb_lin,
        (1 + a) * np.power(rgb_lin, 1 / 2.4) - a,
    )
    rgb = np.clip(rgb, 0.0, 1.0)

    rgb255 = np.rint(rgb * 255).astype(int)
    return [
        f"#{r:02X}{g:02X}{b:02X}"
        for r, g, b in zip(rgb255[0].flat, rgb255[1].flat, rgb255[2].flat)
    ]


def ggplot2_hue_palette(
    n: int,
    *,
    h: tuple[float, float] = (15.0, 375.0),
    c: float = 100.0,
    l: float = 65.0,
) -> list[str]:
    """Return the ggplot2 default discrete hue palette for *n* categories."""
    if n <= 0:
        return []
    hues = np.linspace(h[0], h[1], n + 1)[:-1]
    return _hcl_to_hex(hues, c=c, l=l)


LATREND_PALETTE = ggplot2_hue_palette(9)


def _cluster_colors(n: int) -> list[str]:
    """Return *n* cluster colours from the ggplot2 hue palette."""
    return ggplot2_hue_palette(int(n))


# ---------------------------------------------------------------------------
# plotnine theme
# ---------------------------------------------------------------------------

def theme_latrend(*, figure_size: tuple[float, float] = (7, 4), base_size: float = 11) -> Any:
    """
    plotnine theme that closely matches R latrend's default ggplot2 look.

    Base theme  : theme_gray() (ggplot2 default)
    """

    p9 = require_plotnine()

    return p9.theme_gray(base_size=base_size, base_family="Helvetica") + p9.theme(
        figure_size=figure_size,
    )


def scale_color_latrend(n: int | None = None) -> Any:
    """plotnine manual colour scale using the latrend palette."""
    p9 = require_plotnine()
    if n is not None:
        return p9.scale_color_manual(values=ggplot2_hue_palette(int(n)))
    return p9.scale_color_manual(values=LATREND_PALETTE)


def scale_fill_latrend(n: int | None = None) -> Any:
    """plotnine manual fill scale using the latrend palette."""
    p9 = require_plotnine()
    if n is not None:
        return p9.scale_fill_manual(values=ggplot2_hue_palette(int(n)))
    return p9.scale_fill_manual(values=LATREND_PALETTE)


# ---------------------------------------------------------------------------
# matplotlib styling helper
# ---------------------------------------------------------------------------

def apply_mpl_theme(ax, *, base_size: float = 11) -> None:
    """
    Style a matplotlib Axes to approximate theme_gray() from ggplot2.
    """
    import matplotlib as mpl

    ax.set_facecolor("#EBEBEB")
    ax.figure.set_facecolor("white")

    # Grid
    ax.grid(True, which="major", color="white", linewidth=0.6)
    ax.grid(True, which="minor", color="white", linewidth=0.4)
    ax.set_axisbelow(True)

    # Spines (match ggplot2's minimal panel border)
    for spine in ax.spines.values():
        spine.set_color("#EBEBEB")
        spine.set_linewidth(1.0)

    # Ticks
    ax.tick_params(
        axis="both", which="both",
        colors="#333333", labelsize=base_size - 1,
        direction="out", length=3, width=0.4,
    )

    # Labels
    ax.xaxis.label.set_fontsize(base_size)
    ax.xaxis.label.set_color("#333333")
    ax.yaxis.label.set_fontsize(base_size)
    ax.yaxis.label.set_color("#333333")

    if ax.get_title():
        ax.title.set_fontsize(base_size + 2)
        ax.title.set_fontweight("bold")
        ax.title.set_color("#333333")
