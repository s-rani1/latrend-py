from __future__ import annotations

import pathlib
import sys

import pandas as pd


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    import latrend as lt  # noqa: E402

    data_path = repo_root / "tests" / "data" / "latrend_data.csv"
    assign_path = repo_root / "tests" / "data" / "kml_model4_assignments.csv"

    df = pd.read_csv(data_path).drop(columns=["Unnamed: 0"], errors="ignore")
    assign = pd.read_csv(assign_path)

    clusters = assign.set_index("Id")["Cluster"]
    method = lt.LCMethod(id="Id", time="Time", outcome="Y", name="KML (demo)")
    model = lt.LCModel(method=method, data=df[["Id", "Time", "Y"]], clusters=clusters)

    p = lt.plotClusterTrajectories(
        model,
        trajectories=True,
        backend="plotnine",
        figure_size=(7, 5),
        base_size=11,
    )

    out_path = repo_root / "kml_model4_cluster_trajectories.png"
    p.save(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

