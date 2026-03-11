import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
SETTINGS_DIR = EXPERIMENTS_DIR / "settings"
GRADIENTS_DIR = EXPERIMENTS_DIR / "gradients"

WORKPIECE_WIDTH_Y_MM = 150.0
MAX_WAYPOINT_LABELS = 52

# Defaults from the latest offset simulation batch.
DEFAULT_ORIGIN_RUN = "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000"
DEFAULT_LEFT_RUN = "20260306_221950__c636af8ee28e__rpm833p3__feed40p000__fs10000"
DEFAULT_RIGHT_RUN = "20260306_232207__a443bc2674a4__rpm833p3__feed40p000__fs10000"


def load_run(run_folder: str) -> tuple[pd.DataFrame, dict]:
    exp_dir = EXPERIMENTS_DIR / run_folder
    params_hash = run_folder.split("__")[1]
    params_path = SETTINGS_DIR / params_hash / "params.json"
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    exp_csv = exp_dir / "experiment_data.csv"
    df = pd.read_csv(exp_csv)

    required = {"time_s", "tool_orientation", "fmill_x_N", "fmill_y_N"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{exp_csv} missing columns: {sorted(missing)}")

    trace_candidates = sorted(exp_dir.glob("path_trace*.csv"))
    if trace_candidates:
        trace_df = pd.read_csv(trace_candidates[0])
        if "position_along_path_mm" in trace_df.columns and len(trace_df) == len(df):
            df["s_mm"] = trace_df["position_along_path_mm"].to_numpy(dtype=float)
        elif {"time_s", "position_along_path_mm"}.issubset(trace_df.columns):
            df["s_mm"] = np.interp(
                df["time_s"].to_numpy(dtype=float),
                trace_df["time_s"].to_numpy(dtype=float),
                trace_df["position_along_path_mm"].to_numpy(dtype=float),
            )
        else:
            df["s_mm"] = np.arange(len(df), dtype=float)
    else:
        df["s_mm"] = np.arange(len(df), dtype=float)

    return df, params


def load_trace_df(run_folder: str) -> pd.DataFrame:
    exp_dir = EXPERIMENTS_DIR / run_folder
    trace_candidates = sorted(exp_dir.glob("path_trace*.csv"))
    
    trace_df = pd.read_csv(trace_candidates[0])
    required = {"position_along_path_mm", "tool_center_x_mm", "tool_center_y_mm"}
    
    return trace_df


def full_revolution_average(df: pd.DataFrame) -> pd.DataFrame:
    angle = np.unwrap(df["tool_orientation"].to_numpy(dtype=float))
    rev_idx = np.floor((angle - angle[0]) / (2.0 * np.pi)).astype(int)
    rev_idx = rev_idx - rev_idx.min()

    return (
        df.assign(rev_idx=rev_idx)
        .groupby("rev_idx", as_index=False)
        .agg(
            time_s=("time_s", "mean"),
            s_mm=("s_mm", "mean"),
            fx_avg_N=("fmill_x_N", "mean"),
            fy_avg_N=("fmill_y_N", "mean"),
            n_samples=("rev_idx", "size"),
        )
    )


def align_multiple_binned_by_s(revavg_by_name: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    processed = {}
    for name, df in revavg_by_name.items():
        processed[name] = df.sort_values("s_mm").groupby("s_mm", as_index=False).mean(numeric_only=True)

    s_min = max(float(d["s_mm"].min()) for d in processed.values())
    s_max = min(float(d["s_mm"].max()) for d in processed.values())
    if s_max <= s_min:
        raise ValueError("No overlapping s-range in binned trajectories.")

    n_common = min(len(d) for d in processed.values())

    s_common = np.linspace(s_min, s_max, n_common)
    aligned = {}
    for name, d in processed.items():
        aligned[name] = pd.DataFrame(
            {
                "s_mm": s_common,
                "fx_avg_N": np.interp(s_common, d["s_mm"].to_numpy(), d["fx_avg_N"].to_numpy()),
                "fy_avg_N": np.interp(s_common, d["s_mm"].to_numpy(), d["fy_avg_N"].to_numpy()),
            }
        )
        
    return s_common, aligned


def interpolate_trace_xy_to_s(origin_trace_df: pd.DataFrame, s_common: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tr = origin_trace_df.sort_values("position_along_path_mm").groupby("position_along_path_mm", as_index=False).mean(
        numeric_only=True
    )
    s_trace = tr["position_along_path_mm"].to_numpy(dtype=float)
    x_trace = tr["tool_center_x_mm"].to_numpy(dtype=float)
    y_trace = tr["tool_center_y_mm"].to_numpy(dtype=float)
    x = np.interp(s_common, s_trace, x_trace)
    y = np.interp(s_common, s_trace, y_trace)
    return x, y


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
    path_coordinates_mirrored_y = np.array([1.0, -1.0]) * np.flip(path_coordinates, axis=0) - np.array(
        [0.0, WORKPIECE_WIDTH_Y_MM]
    )
    path_coordinates_full_with_start = np.concatenate(
        [path_start_coordinates, path_coordinates, path_coordinates_mirrored_y], axis=0
    )
    path_coordinates_full_with_start[:, 0] += float(x_offset_mm)
    path_coordinates_full_with_start[:, 1] += float(y_offset_mm)
    return path_coordinates_full_with_start


def project_waypoints_to_origin_s(waypoint_xy: np.ndarray, df_trace_origin: pd.DataFrame) -> np.ndarray:
    trace_xy = df_trace_origin[["tool_center_x_mm", "tool_center_y_mm"]].to_numpy(dtype=float)
    trace_s = df_trace_origin["position_along_path_mm"].to_numpy(dtype=float)
    if len(waypoint_xy) == 0 or len(trace_xy) == 0:
        return np.array([], dtype=float)

    s_i = np.zeros(len(waypoint_xy), dtype=float)
    for i, wp in enumerate(waypoint_xy):
        d2 = np.sum((trace_xy - wp) ** 2, axis=1)
        s_i[i] = trace_s[int(np.argmin(d2))]
    return np.maximum.accumulate(s_i)


def add_waypoint_axis(
    ax: plt.Axes,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
    max_labels: int = MAX_WAYPOINT_LABELS,
):
    for s_i in waypoint_s:
        ax.axvline(s_i, color="k", linewidth=0.7, alpha=0.12)

    if len(waypoint_s) == 0:
        return

    if len(waypoint_s) <= max_labels:
        label_idx = np.arange(len(waypoint_s), dtype=int)
    else:
        label_idx = np.linspace(0, len(waypoint_s) - 1, max_labels, dtype=int)
        label_idx = np.unique(label_idx)

    tick_pos = waypoint_s[label_idx]
    tick_lbl = [f"S{i}" for i in waypoint_ids[label_idx]]

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(tick_pos)
    secax.set_xticklabels(tick_lbl, rotation=90, fontsize=7)
    secax.set_xlabel("Path waypoints S_i")


def save_gradient_outputs(
    origin_run: str,
    left_run: str,
    right_run: str,
    offset_mm: float,
    origin_params: dict,
    left_params: dict,
    right_params: dict,
    directional_df: pd.DataFrame,
    jacobian_df: pd.DataFrame,
):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = GRADIENTS_DIR / f"{stamp}__origin_{origin_run.split('__')[0]}__left_{left_run.split('__')[0]}__right_{right_run.split('__')[0]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    directional_path = out_dir / "gradients_s.csv"
    jacobian_path = out_dir / "gradients_xy.csv"
    directional_df.to_csv(directional_path, index=False)
    jacobian_df.to_csv(jacobian_path, index=False)

    metadata_dir = out_dir / "mini-metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    def run_file_manifest(run_folder: str, run_params: dict) -> dict:
        exp_dir = EXPERIMENTS_DIR / run_folder
        params_hash = run_folder.split("__")[1]
        trace_candidates = sorted(exp_dir.glob("path_trace*.csv"))
        return {
            "run_folder_name": run_folder,
            "experiment_dir": str(exp_dir),
            "experiment_data_csv": str(exp_dir / "experiment_data.csv"),
            "path_trace_csv": str(trace_candidates[0]) if trace_candidates else None,
            "params_json": str(SETTINGS_DIR / params_hash / "params.json"),
            "trajectory_variant": run_params.get("trajectory_variant", None),
        }

    metadata = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "offset_mm": float(offset_mm),
        "runs": {
            "origin": run_file_manifest(origin_run, origin_params),
            "left": run_file_manifest(left_run, left_params),
            "right": run_file_manifest(right_run, right_params),
        },
        "files": {
            "gradients_s_csv": directional_path.name,
            "gradients_xy_csv": jacobian_path.name,
        },
    }
    with open(metadata_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved gradients_s to: {directional_path}")
    print(f"Saved gradients_xy to: {jacobian_path}")
    print(f"Saved metadata file to: {metadata_dir / 'metadata.json'}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Load origin/left/right offset runs, bin force per revolution, and compute chain-rule force Jacobian."
        )
    )
    parser.add_argument("--origin-run", default=DEFAULT_ORIGIN_RUN, help="Origin run folder name.")
    parser.add_argument("--left-run", default=DEFAULT_LEFT_RUN, help="Left run folder name.")
    parser.add_argument("--right-run", default=DEFAULT_RIGHT_RUN, help="Right run folder name.")
    parser.add_argument(
        "--offset-mm",
        type=float,
        default=None,
        help="Offset per side in mm. If omitted, uses params['trajectory_offset_mm'].",
    )
    args = parser.parse_args()

    origin_df, origin_params = load_run(args.origin_run)
    left_df, left_params = load_run(args.left_run)
    right_df, right_params = load_run(args.right_run)
    origin_trace_df = load_trace_df(args.origin_run)

    print(f"Loaded origin run: {args.origin_run} (variant={origin_params.get('trajectory_variant', 'origin')})")
    print(f"Loaded left   run: {args.left_run} (variant={left_params.get('trajectory_variant', 'left')})")
    print(f"Loaded right  run: {args.right_run} (variant={right_params.get('trajectory_variant', 'right')})")

    offset_mm = args.offset_mm
    if offset_mm is None:
        offset_mm = float(left_params.get("trajectory_offset_mm", np.nan))
        if not np.isfinite(offset_mm) or offset_mm <= 0:
            offset_mm = float(right_params.get("trajectory_offset_mm", np.nan))
    if not np.isfinite(offset_mm) or offset_mm <= 0:
        raise ValueError("Could not determine valid offset. Pass --offset-mm explicitly.")

    origin_rev = full_revolution_average(origin_df)
    left_rev = full_revolution_average(left_df)
    right_rev = full_revolution_average(right_df)
    print(f"Revolution bins: origin={len(origin_rev)}, left={len(left_rev)}, right={len(right_rev)}")
    print(f"Offset used for normal derivative: {offset_mm:.6f} mm")

    s_common, aligned = align_multiple_binned_by_s({"origin": origin_rev, "left": left_rev, "right": right_rev})
    origin_rev_aligned = aligned["origin"]
    left_rev_aligned = aligned["left"]
    right_rev_aligned = aligned["right"]

    edge_order = 2 if len(s_common) > 2 else 1

    # Directional derivatives of binned average forces.
    dfx_ds = np.gradient(origin_rev_aligned["fx_avg_N"].to_numpy(), s_common, edge_order=edge_order)
    dfy_ds = np.gradient(origin_rev_aligned["fy_avg_N"].to_numpy(), s_common, edge_order=edge_order)
    dfx_dn = (left_rev_aligned["fx_avg_N"].to_numpy() - right_rev_aligned["fx_avg_N"].to_numpy()) / (2.0 * offset_mm)
    dfy_dn = (left_rev_aligned["fy_avg_N"].to_numpy() - right_rev_aligned["fy_avg_N"].to_numpy()) / (2.0 * offset_mm)

    # Local path frame (t, n) from origin trajectory geometry.
    x_s, y_s = interpolate_trace_xy_to_s(origin_trace_df, s_common)
    tx = np.gradient(x_s, s_common, edge_order=edge_order)
    ty = np.gradient(y_s, s_common, edge_order=edge_order)
    tnorm = np.maximum(np.hypot(tx, ty), 1e-12)
    tx = tx / tnorm
    ty = ty / tnorm
    nx = -ty
    ny = tx

    # Chain rule: grad(F) = t * dF/ds + n * dF/dn.
    dfx_dx = tx * dfx_ds + nx * dfx_dn
    dfx_dy = ty * dfx_ds + ny * dfx_dn
    dfy_dx = tx * dfy_ds + nx * dfy_dn
    dfy_dy = ty * dfy_ds + ny * dfy_dn

    directional_df = pd.DataFrame(
        {
            "s_mm": s_common,
            "x_mm": x_s,
            "y_mm": y_s,
            "tx": tx,
            "ty": ty,
            "nx": nx,
            "ny": ny,
            "dFx_ds_N_per_mm": dfx_ds,
            "dFy_ds_N_per_mm": dfy_ds,
            "dFx_dn_N_per_mm": dfx_dn,
            "dFy_dn_N_per_mm": dfy_dn,
        }
    )
    jacobian_df = pd.DataFrame(
        {
            "s_mm": s_common,
            "dFx_dx_N_per_mm": dfx_dx,
            "dFx_dy_N_per_mm": dfx_dy,
            "dFy_dx_N_per_mm": dfy_dx,
            "dFy_dy_N_per_mm": dfy_dy,
        }
    )

    save_gradient_outputs(
        origin_run=args.origin_run,
        left_run=args.left_run,
        right_run=args.right_run,
        offset_mm=offset_mm,
        origin_params=origin_params,
        left_params=left_params,
        right_params=right_params,
        directional_df=directional_df,
        jacobian_df=jacobian_df,
    )

    waypoint_xy = planned_waypoints_full_with_start(
        x_offset_mm=float(origin_params.get("path_x_offset_mm", 0.0)),
        y_offset_mm=float(origin_params.get("path_y_offset_mm", 0.0)),
    )
    waypoint_s = project_waypoints_to_origin_s(waypoint_xy, origin_trace_df)
    waypoint_ids = np.arange(1, len(waypoint_s) + 1, dtype=int)

    fig_traj, ax_traj = plt.subplots(figsize=(8, 7))
    if {"tool_center_x_mm", "tool_center_y_mm"}.issubset(origin_trace_df.columns):
        ax_traj.plot(origin_trace_df["tool_center_x_mm"], origin_trace_df["tool_center_y_mm"], label="Origin", linewidth=1.2)
    if {"tool_center_x_mm", "tool_center_y_mm"}.issubset(left_df.columns):
        ax_traj.plot(left_df["tool_center_x_mm"], left_df["tool_center_y_mm"], label="Left", linewidth=1.2)
    if {"tool_center_x_mm", "tool_center_y_mm"}.issubset(right_df.columns):
        ax_traj.plot(right_df["tool_center_x_mm"], right_df["tool_center_y_mm"], label="Right", linewidth=1.2)
    ax_traj.set_xlabel("X [mm]")
    ax_traj.set_ylabel("Y [mm]")
    ax_traj.set_title("Offset trajectories (origin/left/right)")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend()
    fig_traj.tight_layout()

    fig_binned, axes_binned = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes_binned[0].plot(origin_rev["s_mm"], origin_rev["fx_avg_N"], label="Fx origin (avg/rev)", linewidth=1.2)
    axes_binned[0].plot(left_rev["s_mm"], left_rev["fx_avg_N"], label="Fx left (avg/rev)", linewidth=1.2)
    axes_binned[0].plot(right_rev["s_mm"], right_rev["fx_avg_N"], label="Fx right (avg/rev)", linewidth=1.2)
    axes_binned[0].set_ylabel("Fx avg/rev [N]")
    axes_binned[0].set_title("Force binned per spindle revolution")
    add_waypoint_axis(axes_binned[0], waypoint_s, waypoint_ids)
    axes_binned[0].grid(True)
    axes_binned[0].legend()

    axes_binned[1].plot(origin_rev["s_mm"], origin_rev["fy_avg_N"], label="Fy origin (avg/rev)", linewidth=1.2)
    axes_binned[1].plot(left_rev["s_mm"], left_rev["fy_avg_N"], label="Fy left (avg/rev)", linewidth=1.2)
    axes_binned[1].plot(right_rev["s_mm"], right_rev["fy_avg_N"], label="Fy right (avg/rev)", linewidth=1.2)
    axes_binned[1].set_xlabel("Path coordinate s [mm]")
    axes_binned[1].set_ylabel("Fy avg/rev [N]")
    add_waypoint_axis(axes_binned[1], waypoint_s, waypoint_ids)
    axes_binned[1].grid(True)
    axes_binned[1].legend()
    fig_binned.tight_layout()

    fig_dir, axes_dir = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes_dir[0].plot(s_common, dfx_ds, label="d(Fx_avg)/ds", linewidth=1.2)
    axes_dir[0].plot(s_common, dfx_dn, label="d(Fx_avg)/dn", linewidth=1.2)
    axes_dir[0].set_ylabel("Directional grad [N/mm]")
    axes_dir[0].set_title("Directional gradients of binned average forces")
    add_waypoint_axis(axes_dir[0], waypoint_s, waypoint_ids)
    axes_dir[0].grid(True)
    axes_dir[0].legend()

    axes_dir[1].plot(s_common, dfy_ds, label="d(Fy_avg)/ds", linewidth=1.2)
    axes_dir[1].plot(s_common, dfy_dn, label="d(Fy_avg)/dn", linewidth=1.2)
    axes_dir[1].set_xlabel("Path coordinate s [mm]")
    axes_dir[1].set_ylabel("Directional grad [N/mm]")
    add_waypoint_axis(axes_dir[1], waypoint_s, waypoint_ids)
    axes_dir[1].grid(True)
    axes_dir[1].legend()
    fig_dir.tight_layout()

    fig_jac, axes_jac = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_jac[0, 0].plot(s_common, dfx_dx, linewidth=1.2, label="dFx/dx")
    axes_jac[0, 0].set_ylabel("N/mm")
    axes_jac[0, 0].set_title("dFx/dx")
    add_waypoint_axis(axes_jac[0, 0], waypoint_s, waypoint_ids)
    axes_jac[0, 0].grid(True)
    axes_jac[0, 0].legend()

    axes_jac[0, 1].plot(s_common, dfx_dy, linewidth=1.2, label="dFx/dy")
    axes_jac[0, 1].set_title("dFx/dy")
    add_waypoint_axis(axes_jac[0, 1], waypoint_s, waypoint_ids)
    axes_jac[0, 1].grid(True)
    axes_jac[0, 1].legend()

    axes_jac[1, 0].plot(s_common, dfy_dx, linewidth=1.2, label="dFy/dx")
    axes_jac[1, 0].set_xlabel("Path coordinate s [mm]")
    axes_jac[1, 0].set_ylabel("N/mm")
    axes_jac[1, 0].set_title("dFy/dx")
    add_waypoint_axis(axes_jac[1, 0], waypoint_s, waypoint_ids)
    axes_jac[1, 0].grid(True)
    axes_jac[1, 0].legend()

    axes_jac[1, 1].plot(s_common, dfy_dy, linewidth=1.2, label="dFy/dy")
    axes_jac[1, 1].set_xlabel("Path coordinate s [mm]")
    axes_jac[1, 1].set_title("dFy/dy")
    add_waypoint_axis(axes_jac[1, 1], waypoint_s, waypoint_ids)
    axes_jac[1, 1].grid(True)
    axes_jac[1, 1].legend()
    fig_jac.suptitle("Force Jacobian matrix J(s) = dF/d[x, y]")
    fig_jac.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
