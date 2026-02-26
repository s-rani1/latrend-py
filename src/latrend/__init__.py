from ._version import __version__
from .core.method import LCMethod
from .core.model import LCModel, LCModels
from .core.pipeline import fitLatrendMethod, latrendBatchCluster, latrendCluster, latrendRepCluster
from .core.trajectories import trajectories
from .core.matrix import tsframe, tsframeToMatrix, tsmatrix, tsmatrixToFrame
from .data.simulate import generateData, generateLongData, generateTrajectories, latrendData
from .methods.features import lcMethodFeatures
from .methods.lmkm import lcMethodLMKM
from .methods.random import lcMethodRandom
from .methods.rmethod import lcMethodR
from .backends import is_r_available
from .plots import (
    plotClassProbabilities,
    plotClassProportions,
    plotMetric,
    plotClusterTrajectories,
    plotFittedTrajectories,
    plotTrajectories,
    theme_latrend,
    LATREND_PALETTE,
    make_clusterPropLabels,
    make_clusterSizeLabels,
)
from .report import lcModelReport

__all__ = [
    "__version__",
    # Core types
    "LCMethod",
    "LCModel",
    "LCModels",
    # Data utilities
    "generateTrajectories",
    "generateData",
    "generateLongData",
    "latrendData",
    "trajectories",
    "tsmatrix",
    "tsmatrixToFrame",
    "tsframe",
    "tsframeToMatrix",
    # Methods
    "lcMethodRandom",
    "lcMethodLMKM",
    "lcMethodFeatures",
    "lcMethodR",
    "is_r_available",
    # Pipeline
    "fitLatrendMethod",
    "latrendCluster",
    "latrendBatchCluster",
    "latrendRepCluster",
    # Plots
    "plotTrajectories",
    "plotClusterTrajectories",
    "plotFittedTrajectories",
    "plotMetric",
    "plotClassProportions",
    "plotClassProbabilities",
    "theme_latrend",
    "LATREND_PALETTE",
    "make_clusterPropLabels",
    "make_clusterSizeLabels",
    # Reporting
    "lcModelReport",
]


def __getattr__(name: str):
    """
    Dynamic fallback for lcMethod* constructors that are not yet implemented in Python.

    Example: `lcMethodLcmmGMM(...)` will return an R-backed method object created via lcMethodR().
    """

    if name.startswith("lcMethod") and name not in globals():
        def _factory(**kwargs):
            return lcMethodR(name, **kwargs)

        _factory.__name__ = name
        return _factory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
