import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulation.milling.milling_data import (
    generate_chafli_trajectory,
    planned_waypoints_full_with_start,
    project_waypoints_to_s,
)


EXPERIMENT_DIR = Path("data/experiments")
SETTINGS_DIR = EXPERIMENT_DIR / "settings"
MAX_WAYPOINT_LABELS = 52


def build_experiment_description(params, run_id):
    rpm = params.get("rpm", None)
    feed = params.get("feed_speed_mm_s", None)
    dt = params.get("dt", None)
    ap = params.get("axial_cutting_depth_mm", None)
    z = params.get("z_teeth", None)
    fz = params.get("feed_per_tooth_mm", None)

    fs = None
    if dt is not None and dt > 0:
        fs = 1.0 / dt

    parts = [f"run {run_id}"]
    if rpm is not None:
        parts.append(f"rpm={rpm:.1f}")
    if z is not None:
        parts.append(f"z={z}")
    if ap is not None:
        parts.append(f"ap={ap:.1f} mm")
    if feed is not None:
        parts.append(f"vf={feed:.2f} mm/s")
    if fz is not None:
        parts.append(f"fz={fz:.4f} mm/tooth")
    if fs is not None:
        parts.append(f"fs={fs:.0f} Hz")
    return " | ".join(parts)


def compute_revolution_average_from_angle(t, angle_rad, signal_2d):
    t = np.asarray(t).reshape(-1)
    angle_rad = np.asarray(angle_rad).reshape(-1)
    signal_2d = np.asarray(signal_2d)

    if signal_2d.ndim != 2:
        raise ValueError(f"signal_2d must be 2D, got shape {signal_2d.shape}")
    if len(t) != len(angle_rad) or len(t) != signal_2d.shape[0]:
        raise ValueError("t, angle_rad, and signal_2d must have matching first dimension")

    angle_unwrapped = np.unwrap(angle_rad)
    rev_index = np.floor((angle_unwrapped - angle_unwrapped[0]) / (2.0 * np.pi)).astype(int)

    signal_revavg = np.full_like(signal_2d, np.nan, dtype=float)
    t_rev = []
    fx_rev = []
    fy_rev = []
    n_rev = []

    for rev in np.unique(rev_index):
        mask = rev_index == rev
        if not np.any(mask):
            continue

        mean_xy = np.mean(signal_2d[mask, :], axis=0)
        signal_revavg[mask, :] = mean_xy
        t_rev.append(float(np.mean(t[mask])))
        fx_rev.append(float(mean_xy[0]))
        fy_rev.append(float(mean_xy[1]))
        n_rev.append(int(np.count_nonzero(mask)))

    df_rev = pd.DataFrame(
        {
            "rev_idx": np.arange(len(t_rev)),
            "time_s": t_rev,
            "fx_avg_N": fx_rev,
            "fy_avg_N": fy_rev,
            "n_samples": n_rev,
        }
    )
    return signal_revavg, rev_index, df_rev


def list_experiments():
    print("\nAvailable experiments:\n")
    if not EXPERIMENT_DIR.exists():
        print(f"{EXPERIMENT_DIR} does not exist")
        return
    for p in sorted(EXPERIMENT_DIR.iterdir()):
        if p.is_dir() and p.name != "settings":
            print(p.name)


def load_experiment(run_id: str):
    matches = [p for p in EXPERIMENT_DIR.iterdir() if p.is_dir() and run_id in p.name]
    if not matches:
        raise RuntimeError(f"No experiment found for run_id '{run_id}'")
    if len(matches) > 1:
        print("Multiple matches:")
        for m in matches:
            print(" ", m.name)
        raise RuntimeError("run_id not unique")

    exp_dir = matches[0]
    parts = exp_dir.name.split("__")
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected experiment folder format: {exp_dir.name}")
    settings_hash = parts[1]

    experiment_csv = exp_dir / "experiment_data.csv"
    path_trace_csv = exp_dir / "path_trace.csv"
    params_path = SETTINGS_DIR / settings_hash / "params.json"

    if not experiment_csv.exists():
        raise FileNotFoundError(experiment_csv)
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    df_exp = pd.read_csv(experiment_csv)
    df_trace = pd.read_csv(path_trace_csv) if path_trace_csv.exists() else None
    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    return df_exp, df_trace, params, exp_dir


def get_trajectory_xy_mm(df_exp, df_trace):
    if df_trace is not None and {"tool_center_x_mm", "tool_center_y_mm"}.issubset(df_trace.columns):
        return (
            df_trace["tool_center_x_mm"].to_numpy(),
            df_trace["tool_center_y_mm"].to_numpy(),
        )
    if {"tool_center_nominal_x_mm", "tool_center_nominal_y_mm"}.issubset(df_exp.columns):
        return (
            df_exp["tool_center_nominal_x_mm"].to_numpy(),
            df_exp["tool_center_nominal_y_mm"].to_numpy(),
        )
    raise RuntimeError("No trajectory XY columns found in path_trace.csv or experiment_data.csv")


def load_precomputed_feed_curve(csv_path: str | Path) -> np.ndarray:
    feed_curve_df = pd.read_csv(csv_path)
    feed_curve = feed_curve_df[["x_support", "feed_curve"]].to_numpy(dtype=float)
    order = np.argsort(feed_curve[:, 0], kind="stable")
    feed_curve = feed_curve[order]
    unique_support_mask = np.ones(len(feed_curve), dtype=bool)
    unique_support_mask[1:] = np.diff(feed_curve[:, 0]) > 0.0
    return feed_curve[unique_support_mask]


def reconstruct_expected_trajectory(params: dict) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    required = {"milling_path_csv", "axial_cutting_depth_mm", "feed_speed_mm_s", "dt"}
    if not required.issubset(params):
        return None, None

    feed_curve = None
    if params.get("use_precomputed_feed_profile") and params.get("feed_profile_csv"):
        feed_curve = load_precomputed_feed_curve(params["feed_profile_csv"])

    trajectory_local_mm, pos_on_path_mm = generate_chafli_trajectory(
        params["milling_path_csv"],
        axial_cutting_depth_mm=float(params["axial_cutting_depth_mm"]),
        feed_speed_mm_s=float(params["feed_speed_mm_s"]),
        dt=float(params["dt"]),
        feed_curve=feed_curve,
        min_feed_speed_mm_s=float(params.get("min_feed_speed_mm_s", 1e-6)),
        final_pos_on_path_mm=float(params["trajectory_final_s_mm"]) if params.get("trajectory_final_s_mm") is not None else None,
        path_offset_mm=float(params["path_offset_mm"]) if params.get("path_offset_mm") is not None else None,
        path_offset_side=str(params.get("path_offset_side", "left")),
    )
    return trajectory_local_mm, pos_on_path_mm


def get_realized_tool_xy_mm(df_exp, df_trace):
    if {"tool_center_actual_x_mm", "tool_center_actual_y_mm"}.issubset(df_exp.columns):
        return (
            df_exp["tool_center_actual_x_mm"].to_numpy(),
            df_exp["tool_center_actual_y_mm"].to_numpy(),
        )
    if df_trace is not None and {"tool_center_actual_x_mm", "tool_center_actual_y_mm"}.issubset(df_trace.columns):
        return (
            df_trace["tool_center_actual_x_mm"].to_numpy(),
            df_trace["tool_center_actual_y_mm"].to_numpy(),
        )
    return None, None


def get_waypoint_times_s(trace_source: pd.DataFrame | None, params):
    if trace_source is None:
        return np.array([], dtype=float), np.array([], dtype=int), np.empty((0, 2), dtype=float)

    required_xy_cols = {"tool_center_x_mm", "tool_center_y_mm"}
    required_s_cols = {"position_along_path_mm", "time_s"}
    if not (required_xy_cols.issubset(trace_source.columns) and required_s_cols.issubset(trace_source.columns)):
        return np.array([], dtype=float), np.array([], dtype=int), np.empty((0, 2), dtype=float)

    waypoint_xy = planned_waypoints_full_with_start(
        x_offset_mm=float(params.get("path_x_offset_mm", 0.0)),
        y_offset_mm=float(params.get("path_y_offset_mm", 0.0)),
    )
    waypoint_s = project_waypoints_to_s(waypoint_xy, trace_source)
    waypoint_ids = np.arange(1, len(waypoint_s) + 1, dtype=int)

    trace_base = (
        trace_source[["position_along_path_mm", "time_s"]]
        .sort_values("position_along_path_mm")
        .groupby("position_along_path_mm", as_index=False)
        .mean(numeric_only=True)
    )
    trace_s = trace_base["position_along_path_mm"].to_numpy(dtype=float)
    trace_t = trace_base["time_s"].to_numpy(dtype=float)
    waypoint_t = np.interp(waypoint_s, trace_s, trace_t)
    return waypoint_t, waypoint_ids, waypoint_xy


def add_waypoint_time_axis(ax: plt.Axes, waypoint_t: np.ndarray, waypoint_ids: np.ndarray, max_labels: int = MAX_WAYPOINT_LABELS):
    for t_i in waypoint_t:
        ax.axvline(t_i, color="k", linewidth=0.7, alpha=0.12)

    if len(waypoint_t) == 0:
        return

    if len(waypoint_t) <= max_labels:
        label_idx = np.arange(len(waypoint_t), dtype=int)
    else:
        label_idx = np.unique(np.linspace(0, len(waypoint_t) - 1, max_labels, dtype=int))

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(waypoint_t[label_idx])
    secax.set_xticklabels([f"S{i}" for i in waypoint_ids[label_idx]], rotation=90, fontsize=7)
    secax.set_xlabel("Path waypoints S_i")


def main():
    parser = argparse.ArgumentParser(description="Inspect saved trajectory-milling experiments")
    parser.add_argument("--run-id", help="Run id (timestamp part of experiment folder)")
    parser.add_argument("--list", action="store_true", help="List experiments")
    parser.add_argument(
        "--plot-realized-tool-history",
        action="store_true",
        help="Overlay realized tool XY history (tool_center_actual_x_mm/y_mm) in the trajectory plot",
    )
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.run_id is None:
        parser.error("Provide --run-id or --list")

    df_exp, df_trace, params, exp_dir = load_experiment(args.run_id)

    print("\nLoaded experiment:")
    print(exp_dir)
    print("\nParameters:")
    for k, v in params.items():
        print(f"{k:30s} : {v}")

    if "time_s" not in df_exp.columns:
        raise RuntimeError("CSV missing column 'time_s'")

    t = df_exp["time_s"].to_numpy()
    nominal_traj_mm, nominal_s_mm = reconstruct_expected_trajectory(params)
    if nominal_traj_mm is not None and len(nominal_traj_mm) == len(t):
        x_mm = nominal_traj_mm[:, 0]
        y_mm = nominal_traj_mm[:, 1]
        nominal_trace = pd.DataFrame(
            {
                "time_s": t,
                "position_along_path_mm": nominal_s_mm,
                "tool_center_x_mm": x_mm,
                "tool_center_y_mm": y_mm,
            }
        )
    else:
        x_mm, y_mm = get_trajectory_xy_mm(df_exp, df_trace)
        nominal_trace = df_trace if df_trace is not None else df_exp

    waypoint_t, waypoint_ids, waypoint_xy = get_waypoint_times_s(nominal_trace, params)
    desc = build_experiment_description(params, args.run_id)

    # XY trajectory
    plt.figure(figsize=(8, 8))
    plt.plot(x_mm, y_mm, linewidth=1.2, label="Expected tool trajectory (MillingPath)")
    if len(waypoint_xy) > 0:
        plt.scatter(waypoint_xy[:, 0], waypoint_xy[:, 1], s=14, alpha=0.7, label="Planned waypoints")

    if args.plot_realized_tool_history:
        x_real_mm, y_real_mm = get_realized_tool_xy_mm(df_exp, df_trace)
        if x_real_mm is not None and y_real_mm is not None:
            plt.plot(x_real_mm, y_real_mm, linewidth=1.0, alpha=0.9, label="Tool center history (realized)")
        else:
            print(
                "Warning: '--plot-realized-tool-history' requested, but realized XY columns "
                "(tool_center_actual_x_mm/y_mm) were not found."
            )

    plt.scatter(x_mm[0], y_mm[0], s=35, marker="o", label="Start")
    plt.scatter(x_mm[-1], y_mm[-1], s=35, marker="x", label="End")
    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("XY tool trajectory\n" + desc)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if args.plot_realized_tool_history:
        x_real_mm, y_real_mm = get_realized_tool_xy_mm(df_exp, df_trace)
        if x_real_mm is not None and y_real_mm is not None:
            fig_xy_time, axes_xy_time = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

            axes_xy_time[0].plot(t, x_mm, linewidth=1.2, alpha=0.85, label="x nominal")
            axes_xy_time[0].plot(t, x_real_mm, linewidth=1.0, alpha=0.9, label="x realized")
            axes_xy_time[0].set_ylabel("X [mm]")
            axes_xy_time[0].set_title("Tool center X/Y vs time")
            add_waypoint_time_axis(axes_xy_time[0], waypoint_t, waypoint_ids)
            axes_xy_time[0].grid(True)
            axes_xy_time[0].legend()

            axes_xy_time[1].plot(t, y_mm, linewidth=1.2, alpha=0.85, label="y nominal")
            axes_xy_time[1].plot(t, y_real_mm, linewidth=1.0, alpha=0.9, label="y realized")
            axes_xy_time[1].set_xlabel("Time [s]")
            axes_xy_time[1].set_ylabel("Y [mm]")
            add_waypoint_time_axis(axes_xy_time[1], waypoint_t, waypoint_ids)
            axes_xy_time[1].grid(True)
            axes_xy_time[1].legend()

            fig_xy_time.suptitle(desc)
            fig_xy_time.tight_layout()

            fig_xy_err, axes_xy_err = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ex_mm = x_real_mm - x_mm
            ey_mm = y_real_mm - y_mm

            axes_xy_err[0].plot(t, ex_mm, linewidth=1.1, label="x error = x_realized - x_nominal")
            axes_xy_err[0].set_ylabel("X error [mm]")
            axes_xy_err[0].set_title("TCP position error vs nominal trajectory")
            add_waypoint_time_axis(axes_xy_err[0], waypoint_t, waypoint_ids)
            axes_xy_err[0].grid(True)
            axes_xy_err[0].legend()

            axes_xy_err[1].plot(t, ey_mm, linewidth=1.1, label="y error = y_realized - y_nominal")
            axes_xy_err[1].set_xlabel("Time [s]")
            axes_xy_err[1].set_ylabel("Y error [mm]")
            add_waypoint_time_axis(axes_xy_err[1], waypoint_t, waypoint_ids)
            axes_xy_err[1].grid(True)
            axes_xy_err[1].legend()

            fig_xy_err.suptitle(desc)
            fig_xy_err.tight_layout()

    plotted_fx = "fmill_x_N" in df_exp.columns
    plotted_fy = "fmill_y_N" in df_exp.columns

    if "tool_orientation" in df_exp.columns and plotted_fx and plotted_fy:
        angle = df_exp["tool_orientation"].to_numpy()
        fmill = np.column_stack([df_exp["fmill_x_N"].to_numpy(), df_exp["fmill_y_N"].to_numpy()])
        fmill_revavg, _, _ = compute_revolution_average_from_angle(t, angle, fmill)
    else:
        fmill_revavg = None

    if not (plotted_fx or plotted_fy):
        raise RuntimeError("No milling force columns found (expected fmill_x_N / fmill_y_N)")

    if plotted_fx:
        plt.figure(figsize=(11, 5))
        plt.plot(t, df_exp["fmill_x_N"], label="Fx continuous", alpha=0.85)
        if fmill_revavg is not None:
            plt.plot(
                t,
                fmill_revavg[:, 0],
                "-.",
                linewidth=1.8,
                label="Fx avg (full-rotation bins)",
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Force X [N]")
        plt.title("Force X vs time\n" + desc)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    if plotted_fy:
        plt.figure(figsize=(11, 5))
        plt.plot(t, df_exp["fmill_y_N"], label="Fy continuous", alpha=0.85)
        if fmill_revavg is not None:
            plt.plot(
                t,
                fmill_revavg[:, 1],
                "-.",
                linewidth=1.8,
                label="Fy avg (full-rotation bins)",
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Force Y [N]")
        plt.title("Force Y vs time\n" + desc)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
