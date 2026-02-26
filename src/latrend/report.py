from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.model import LCModel
from .plots.classes import plotClassProbabilities, plotClassProportions
from .plots.trajectories import plotClusterTrajectories


def _save_plot(obj: Any, path: Path) -> None:
    if hasattr(obj, "save"):
        # plotnine ggplot
        obj.save(str(path), dpi=150, verbose=False)
        return
    if hasattr(obj, "figure"):
        # matplotlib Axes
        obj.figure.savefig(path, dpi=150, bbox_inches="tight")
        return
    raise TypeError("Unsupported plot object type for saving")


def lcModelReport(
    model: LCModel,
    out_dir: str | Path,
    *,
    filename: str = "report.md",
    backend: str | None = None,
    figure_size: tuple[float, float] = (7, 4),
    base_size: float = 11,
) -> Path:
    """
    Create a lightweight Markdown report with a few standard plots.

    This is not a 1:1 port of the upstream HTML reporting, but provides a useful
    GitHub-friendly artifact.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = out_dir / "cluster_trajectories.png"
    prop_path = out_dir / "class_proportions.png"
    pp_path = out_dir / "class_probabilities.png"

    p = plotClusterTrajectories(
        model, backend=backend, figure_size=figure_size, base_size=base_size
    )
    _save_plot(p, traj_path)

    p = plotClassProportions(model, backend=backend, figure_size=figure_size, base_size=base_size)
    _save_plot(p, prop_path)

    p = plotClassProbabilities(model, backend=backend, figure_size=figure_size, base_size=base_size)
    _save_plot(p, pp_path)

    report_path = out_dir / filename
    report_path.write_text(
        "\n".join(
            [
                "# lcModel report",
                "",
                f"- Method: `{model.method.getName()}`",
                f"- nClusters: `{model.nClusters()}`",
                "",
                "## Cluster trajectories",
                f"![Cluster trajectories]({traj_path.name})",
                "",
                "## Class proportions",
                f"![Class proportions]({prop_path.name})",
                "",
                "## Posterior probabilities",
                f"![Posterior probabilities]({pp_path.name})",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report_path
