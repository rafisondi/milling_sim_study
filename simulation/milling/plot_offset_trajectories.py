import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milling_path import MillingPath

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_xy_from_path_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"CSV must contain 'x' and 'y' columns: {csv_path}")
    return df[["x", "y"]].to_numpy()


def sample_original_path(path_xy: np.ndarray, n_samples: int) -> np.ndarray:
    milling_path = MillingPath(path_xy, offset=None)
    distances = np.linspace(0.0, milling_path.length(), n_samples)
    return milling_path.points_at_distances(distances).T


def offset_by_normals(path_xy: np.ndarray, offset_mm: float, side: str) -> np.ndarray:
    x = path_xy[:, 0]
    y = path_xy[:, 1]

    tx = np.gradient(x)
    ty = np.gradient(y)
    tangent_norm = np.sqrt(tx * tx + ty * ty)
    tangent_norm = np.maximum(tangent_norm, 1e-12)
    tx = tx / tangent_norm
    ty = ty / tangent_norm

    nx = -ty
    ny = tx

    sign = 1.0 if side == "left" else -1.0
    return np.column_stack((x + sign * offset_mm * nx, y + sign * offset_mm * ny))


def main():
    parser = argparse.ArgumentParser(description="Plot original and left/right offset milling trajectories.")
    parser.add_argument(
        "--path-csv",
        default=str(REPO_ROOT / "data" / "paths" / "Workpiece_long_with_start_milling_path.csv"),
        help="Input path CSV with x,y columns.",
    )
    parser.add_argument(
        "--offset-mm",
        type=float,
        default=0.5,
        help="Parallel offset distance in mm for left/right trajectories.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="Number of sampled points per trajectory.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path. If omitted, opens an interactive plot window.",
    )
    args = parser.parse_args()

    path_csv = Path(args.path_csv)
    if not path_csv.is_absolute():
        path_csv = REPO_ROOT / path_csv

    path_xy = load_xy_from_path_csv(path_csv)

    traj_original = sample_original_path(path_xy, n_samples=args.samples)
    traj_left = offset_by_normals(traj_original, offset_mm=args.offset_mm, side="left")
    traj_right = offset_by_normals(traj_original, offset_mm=args.offset_mm, side="right")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj_left[:, 0], traj_left[:, 1], linewidth=1.3, label=f"Left offset (+{args.offset_mm:.3f} mm)")
    ax.plot(traj_original[:, 0], traj_original[:, 1], linewidth=1.5, label="Original")
    ax.plot(traj_right[:, 0], traj_right[:, 1], linewidth=1.3, label=f"Right offset (+{args.offset_mm:.3f} mm)")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("Milling trajectory offsets")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    sample_idx = np.arange(traj_original.shape[0])
    fig_xy, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_x.plot(sample_idx, traj_left[:, 0], linewidth=1.2, label=f"Left (+{args.offset_mm:.3f} mm)")
    ax_x.plot(sample_idx, traj_original[:, 0], linewidth=1.3, label="Original")
    ax_x.plot(sample_idx, traj_right[:, 0], linewidth=1.2, label=f"Right (+{args.offset_mm:.3f} mm)")
    ax_x.set_ylabel("X [mm]")
    ax_x.set_title("X coordinate vs sample")
    ax_x.grid(True)
    ax_x.legend()

    ax_y.plot(sample_idx, traj_left[:, 1], linewidth=1.2, label=f"Left (+{args.offset_mm:.3f} mm)")
    ax_y.plot(sample_idx, traj_original[:, 1], linewidth=1.3, label="Original")
    ax_y.plot(sample_idx, traj_right[:, 1], linewidth=1.2, label=f"Right (+{args.offset_mm:.3f} mm)")
    ax_y.set_xlabel("Sample index")
    ax_y.set_ylabel("Y [mm]")
    ax_y.set_title("Y coordinate vs sample")
    ax_y.grid(True)
    ax_y.legend()
    fig_xy.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        out_xy = out.with_name(f"{out.stem}_xy_components{out.suffix}")
        fig_xy.savefig(out_xy, dpi=200)
        print(f"Saved XY plot to: {out}")
        print(f"Saved x/y components plot to: {out_xy}")
        return

    plt.show()


if __name__ == "__main__":
    main()
