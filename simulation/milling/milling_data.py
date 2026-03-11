from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from GeopmetryStandalone import Area2D
from milling_path import MillingPath


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
SETTINGS_DIR = EXPERIMENTS_DIR / "settings"
WORKPIECE_WIDTH_Y_MM = 150.0


def find_latest_run_dir(base_dir: Path = EXPERIMENTS_DIR) -> Path:
    run_dirs = [
        path
        for path in base_dir.iterdir()
        if path.is_dir() and (path / "experiment_data.csv").exists() and path.name[:8].isdigit()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with experiment_data.csv found in: {base_dir}")
    return sorted(run_dirs)[-1]


def resolve_run_dir(run_name: str | Path | None) -> Path:
    if run_name is None:
        return find_latest_run_dir()

    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        run_dir = EXPERIMENTS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def resolve_run_paths(run_name: str | Path, role_hint: str | None = None) -> tuple[Path, Path, Path | None, Path]:
    run_dir = resolve_run_dir(run_name)
    exp_csv = run_dir / "experiment_data.csv"
    if not exp_csv.exists():
        raise FileNotFoundError(f"Missing experiment_data.csv in run directory: {run_dir}")

    trace_candidates: list[Path] = []
    if role_hint:
        trace_candidates = sorted(run_dir.glob(f"path_trace_{role_hint}.csv"))
    if not trace_candidates:
        trace_candidates = sorted(run_dir.glob("path_trace*.csv"))
    trace_csv = trace_candidates[0] if trace_candidates else None

    settings_hash = run_dir.name.split("__")[1] if "__" in run_dir.name else ""
    params_json = SETTINGS_DIR / settings_hash / "params.json"
    return run_dir, exp_csv, trace_csv, params_json


def load_json_dict(path: Path, *, missing_ok: bool = False) -> dict:
    if missing_ok and not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run_params(run_name: str | Path, *, missing_ok: bool = False) -> dict:
    _, _, _, params_json = resolve_run_paths(run_name)
    return load_json_dict(params_json, missing_ok=missing_ok)


def load_trace_df(run_name: str | Path, role_hint: str | None = None) -> pd.DataFrame:
    run_dir, _, trace_csv, _ = resolve_run_paths(run_name, role_hint=role_hint)
    if trace_csv is None or not trace_csv.exists():
        raise FileNotFoundError(f"Missing path trace CSV in run directory: {run_dir}")
    return pd.read_csv(trace_csv)


def load_run_df(
    run_name: str | Path,
    *,
    role_hint: str | None = None,
    required_cols: set[str] | None = None,
    include_xy: bool = False,
) -> pd.DataFrame:
    _, exp_csv, trace_csv, _ = resolve_run_paths(run_name, role_hint=role_hint)
    df = pd.read_csv(exp_csv)

    if required_cols:
        missing = required_cols.difference(df.columns)
        if missing:
            raise ValueError(f"{exp_csv} missing columns: {sorted(missing)}")

    if trace_csv is None or not trace_csv.exists():
        df["s_mm"] = np.arange(len(df), dtype=float)
        return df

    trace_df = pd.read_csv(trace_csv)
    trace_time = trace_df["time_s"].to_numpy(dtype=float) if "time_s" in trace_df.columns else None
    sample_time = df["time_s"].to_numpy(dtype=float) if "time_s" in df.columns else None

    def assign_trace_column(dest_col: str, source_col: str) -> None:
        if source_col not in trace_df.columns:
            return
        values = trace_df[source_col].to_numpy(dtype=float)
        if len(trace_df) == len(df):
            df[dest_col] = values
            return
        if trace_time is not None and sample_time is not None:
            df[dest_col] = np.interp(sample_time, trace_time, values)

    assign_trace_column("s_mm", "position_along_path_mm")
    if "s_mm" not in df.columns:
        df["s_mm"] = np.arange(len(df), dtype=float)

    if include_xy:
        assign_trace_column("x_mm", "tool_center_x_mm")
        assign_trace_column("y_mm", "tool_center_y_mm")

    return df


def load_workpiece_vertices_from_pickle(pickle_path: str | Path) -> np.ndarray:
    workpiece_area = Area2D.load_from_disk(pickle_path)
    boundary_points_vec = workpiece_area.get_boundary_points()
    boundary_points = np.array([[p.xyz_in_array()[0], p.xyz_in_array()[2]] for p in boundary_points_vec])
    return boundary_points.T


def generate_chafli_trajectory(
    path_csv: str | Path,
    *,
    axial_cutting_depth_mm: float,
    feed_speed_mm_s: float,
    dt: float,
    path_offset_mm: float | None = None,
    path_offset_side: str = "left",
) -> tuple[np.ndarray, np.ndarray]:
    milling_path_df = pd.read_csv(path_csv, index_col=0)
    milling_path = MillingPath(
        milling_path_df.to_numpy(),
        axial_cutting_dept=axial_cutting_depth_mm,
        feed_curve=feed_speed_mm_s,
        offset=path_offset_mm,
        offset_side=path_offset_side,
    )

    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []

    while pos_on_path <= milling_path.length():
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += feed_speed_mm_s * dt

    return np.asarray(trajectory_local), np.asarray(pos_on_path_log)


def planned_waypoints_full_with_start(x_offset_mm: float = 0.0, y_offset_mm: float = 0.0) -> np.ndarray:
    path_coordinates = np.array(
        [
            [-20.0, 7.0],
            [0.0, 7.0],
            [20.0, 8.0],
            [30.0, 6.5],
            [40.0, 7.5],
            [65.0, 0.0],
            [80.0, 1.5],
            [85.0, 3.5],
            [92.0, 4.0],
            [98.0, 2.0],
            [105.0, -4.0],
            [105.0, -18.0],
            [95.0, -25.0],
            [70.0, -27.0],
            [40.0, -28.0],
            [40.0, -30.0],
            [45.0, -31.0],
            [85.0, -33.0],
            [93.0, -40.0],
            [95.0, -50.0],
            [95.0, -60.0],
        ],
        dtype=float,
    )
    path_start_coordinates = np.array(
        [
            [5.0, -190.0],
            [5.0, -150.0],
            [5.0, -100.0],
            [5.0, -50.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [3.0, 23.0],
            [-16.0, 30.0],
            [-30.0, 21.0],
            [-30.0, 12.0],
        ],
        dtype=float,
    )

    mirrored = np.array([1.0, -1.0]) * np.flip(path_coordinates, axis=0) - np.array([0.0, WORKPIECE_WIDTH_Y_MM])
    waypoints = np.concatenate([path_start_coordinates, path_coordinates, mirrored], axis=0)
    waypoints[:, 0] += float(x_offset_mm)
    waypoints[:, 1] += float(y_offset_mm)
    return waypoints


def project_waypoints_to_s(waypoint_xy: np.ndarray, trace_df: pd.DataFrame) -> np.ndarray:
    if {"x_mm", "y_mm"}.issubset(trace_df.columns):
        trace_xy = trace_df[["x_mm", "y_mm"]].to_numpy(dtype=float)
    elif {"tool_center_x_mm", "tool_center_y_mm"}.issubset(trace_df.columns):
        trace_xy = trace_df[["tool_center_x_mm", "tool_center_y_mm"]].to_numpy(dtype=float)
    else:
        raise ValueError("Trace dataframe must contain either x_mm/y_mm or tool_center_x_mm/tool_center_y_mm.")

    if "s_mm" in trace_df.columns:
        trace_s = trace_df["s_mm"].to_numpy(dtype=float)
    elif "position_along_path_mm" in trace_df.columns:
        trace_s = trace_df["position_along_path_mm"].to_numpy(dtype=float)
    else:
        raise ValueError("Trace dataframe must contain either s_mm or position_along_path_mm.")

    if len(waypoint_xy) == 0 or len(trace_xy) == 0:
        return np.array([], dtype=float)

    s_i = np.zeros(len(waypoint_xy), dtype=float)
    for i, wp in enumerate(waypoint_xy):
        d2 = np.sum((trace_xy - wp) ** 2, axis=1)
        s_i[i] = trace_s[int(np.argmin(d2))]

    return np.maximum.accumulate(s_i)
