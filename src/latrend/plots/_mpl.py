from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def get_axes(ax: Axes | None = None) -> Axes:
    """
    Return an Axes, creating a new Agg-backed Figure when needed.

    Using the Agg canvas avoids hard dependencies on GUI backends in headless/CI environments.
    """

    if ax is not None:
        return ax
    fig = Figure()
    FigureCanvas(fig)
    return fig.add_subplot(1, 1, 1)

