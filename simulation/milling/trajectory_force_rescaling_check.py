import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milling_data import EXPERIMENTS_DIR, load_json_dict, load_run_df, planned_waypoints_full_with_start, project_waypoints_to_s, resolve_run_paths

DEFAULT_ORIGIN_RUN = "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000"
DEFAULT_TARGET_RUN = "20260308_123323__9cc66a5310ea__rpm833p3__feed60p000__fs10000"
TARGET_ROLE = "left"  # "left" or "right"
SHOW_PLOTS = True
MAX_WAYPOINT_LABELS = 52
REL_ERR_MIN_MEAS_N = 0.5

def full_revolution_average(df: pd.DataFrame) -> pd.DataFrame:
    angle = np.unwrap(df["tool_orientation"].to_numpy(dtype=float))
    rev_idx = np.floor((angle - angle[0]) / (2.0 * np.pi)).astype(int)
    rev_idx -= rev_idx.min()

    return (
        df.assign(rev_idx=rev_idx)
        .groupby("rev_idx", as_index=False)
        .agg(
            time_s=("time_s", "mean"),
            s_mm=("s_mm", "mean"),
            x_mm=("x_mm", "mean"),
            y_mm=("y_mm", "mean"),
            fx_avg_N=("fmill_x_N", "mean"),
            fy_avg_N=("fmill_y_N", "mean"),
        )
    )


def interp_on_s(df: pd.DataFrame, s_common: np.ndarray, cols: list[str]) -> pd.DataFrame:
    base = df.sort_values("s_mm").groupby("s_mm", as_index=False).mean(numeric_only=True)
    out = pd.DataFrame({"s_mm": s_common})
    s_src = base["s_mm"].to_numpy()
    for col in cols:
        out[col] = np.interp(s_common, s_src, base[col].to_numpy())
    return out


def common_s(origin_rev: pd.DataFrame, target_rev: pd.DataFrame) -> np.ndarray:
    s_min = max(origin_rev["s_mm"].min(), target_rev["s_mm"].min())
    s_max = min(origin_rev["s_mm"].max(), target_rev["s_mm"].max())
    n = min(len(origin_rev), len(target_rev))
    return np.linspace(s_min, s_max, n)


def rmse_mae(meas: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    err = meas - pred
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def load_params(path: Path) -> dict:
    return load_json_dict(path)


def feed_per_tooth(params: dict) -> float:
    if "feed_per_tooth_mm" in params:
        return float(params["feed_per_tooth_mm"])

    if {"rpm", "feed_speed_mm_s", "z_teeth"}.issubset(params):
        rpm = float(params["rpm"])
        feed_mm_s = float(params["feed_speed_mm_s"])
        teeth = int(params["z_teeth"])
        rev_per_s = rpm / 60.0
        return feed_mm_s / (rev_per_s * teeth)

    rpm = float(params["spindle_speed_rpm"])
    feed_mm_per_min = float(params["feed_rate_mm_per_min"])
    tooth_count = int(params["tool_tooth_count"])
    return feed_mm_per_min / (rpm * tooth_count)


def print_run_params(origin_name: str, target_name: str, origin_params: dict, target_params: dict):
    origin_fz = feed_per_tooth(origin_params)
    target_fz = feed_per_tooth(target_params)

    print("\nOrigin run params")
    print(f"run_name: {origin_name}")
    print(f"  rpm: {origin_params.get('rpm')}")
    print(f"  feed_speed_mm_s: {origin_params.get('feed_speed_mm_s')}")
    print(f"  z_teeth: {origin_params.get('z_teeth')}")
    print(f"  trajectory_variant: {origin_params.get('trajectory_variant')}")
    print(f"  path_offset_mm: {origin_params.get('path_offset_mm')}")
    print(f"  feed_per_tooth_mm: {origin_fz:.6f}")

    print("\nTarget run params")
    print(f"run_name: {target_name}")
    print(f"  rpm: {target_params.get('rpm')}")
    print(f"  feed_speed_mm_s: {target_params.get('feed_speed_mm_s')}")
    print(f"  z_teeth: {target_params.get('z_teeth')}")
    print(f"  trajectory_variant: {target_params.get('trajectory_variant')}")
    print(f"  path_offset_mm: {target_params.get('path_offset_mm')}")
    print(f"  feed_per_tooth_mm: {target_fz:.6f}")

    print("\nFeed-per-tooth ratio")
    print(f"target / origin: {target_fz / origin_fz:.6f}")

def add_waypoint_axis(ax: plt.Axes, waypoint_s: np.ndarray, waypoint_ids: np.ndarray):
    for s_i in waypoint_s:
        ax.axvline(s_i, color="k", linewidth=0.7, alpha=0.12)

    if len(waypoint_s) == 0:
        return

    if len(waypoint_s) <= MAX_WAYPOINT_LABELS:
        label_idx = np.arange(len(waypoint_s), dtype=int)
    else:
        label_idx = np.unique(np.linspace(0, len(waypoint_s) - 1, MAX_WAYPOINT_LABELS, dtype=int))

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(waypoint_s[label_idx])
    secax.set_xticklabels([f"S{i}" for i in waypoint_ids[label_idx]], rotation=90, fontsize=7)
    secax.set_xlabel("Path waypoints S_i")


def plot_force_rescaling(
    s_common: np.ndarray,
    fx_target: np.ndarray,
    fy_target: np.ndarray,
    fx_origin: np.ndarray,
    fy_origin: np.ndarray,
    fx_origin_scaled: np.ndarray,
    fy_origin_scaled: np.ndarray,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(s_common, fx_target, label="Fx target")
    axes[0].plot(s_common, fx_origin, label="Fx origin")
    axes[0].plot(s_common, fx_origin_scaled, "--", label="Fx origin rescaled by fz")
    axes[0].set_ylabel("Fx [N]")
    add_waypoint_axis(axes[0], waypoint_s, waypoint_ids)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(s_common, fy_target, label="Fy target")
    axes[1].plot(s_common, fy_origin, label="Fy origin")
    axes[1].plot(s_common, fy_origin_scaled, "--", label="Fy origin rescaled by fz")
    axes[1].set_xlabel("s [mm]")
    axes[1].set_ylabel("Fy [N]")
    add_waypoint_axis(axes[1], waypoint_s, waypoint_ids)
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Raw force comparison with feed-per-tooth rescaling")
    fig.tight_layout()


def plot_force_normalized(
    s_common: np.ndarray,
    fx_target_per_fz: np.ndarray,
    fy_target_per_fz: np.ndarray,
    fx_origin_per_fz: np.ndarray,
    fy_origin_per_fz: np.ndarray,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(s_common, fx_target_per_fz, label="Fx target / fz_target")
    axes[0].plot(s_common, fx_origin_per_fz, label="Fx origin / fz_origin")
    axes[0].set_ylabel("Fx / fz [N / (mm/tooth)]")
    add_waypoint_axis(axes[0], waypoint_s, waypoint_ids)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(s_common, fy_target_per_fz, label="Fy target / fz_target")
    axes[1].plot(s_common, fy_origin_per_fz, label="Fy origin / fz_origin")
    axes[1].set_xlabel("s [mm]")
    axes[1].set_ylabel("Fy / fz [N / (mm/tooth)]")
    add_waypoint_axis(axes[1], waypoint_s, waypoint_ids)
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Force comparison normalized by feed per tooth")
    fig.tight_layout()


def plot_rescaling_error(
    s_common: np.ndarray,
    fx_target: np.ndarray,
    fy_target: np.ndarray,
    fx_est: np.ndarray,
    fy_est: np.ndarray,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
):
    err_fx = fx_target - fx_est
    err_fy = fy_target - fy_est
    err_mag = np.hypot(err_fx, err_fy)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(s_common, err_fx, label="Fx error = Fx_target - Fx_est")
    axes[0].set_ylabel("Fx error [N]")
    add_waypoint_axis(axes[0], waypoint_s, waypoint_ids)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(s_common, err_fy, label="Fy error = Fy_target - Fy_est")
    axes[1].set_ylabel("Fy error [N]")
    add_waypoint_axis(axes[1], waypoint_s, waypoint_ids)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(s_common, err_mag, label="|error|")
    axes[2].set_xlabel("s [mm]")
    axes[2].set_ylabel("|error| [N]")
    add_waypoint_axis(axes[2], waypoint_s, waypoint_ids)
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle("Rescaling estimation error (target - origin rescaled)")
    fig.tight_layout()


def plot_rescaling_relative_error(
    s_common: np.ndarray,
    fx_target: np.ndarray,
    fy_target: np.ndarray,
    fx_est: np.ndarray,
    fy_est: np.ndarray,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
):
    err_mag = np.hypot(fx_target - fx_est, fy_target - fy_est)
    f_meas_mag = np.hypot(fx_target, fy_target)

    rel_err = np.full_like(f_meas_mag, np.nan, dtype=float)
    valid = f_meas_mag >= REL_ERR_MIN_MEAS_N
    rel_err[valid] = err_mag[valid] / f_meas_mag[valid]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s_common, 100.0 * rel_err, label="Relative error |F| [%]")
    ax.set(
        xlabel="s [mm]",
        ylabel="Relative error [%]",
        title=f"Rescaling relative error (|F_meas| >= {REL_ERR_MIN_MEAS_N} N)",
    )
    add_waypoint_axis(ax, waypoint_s, waypoint_ids)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def main():
    if TARGET_ROLE not in ("left", "right"):
        raise ValueError(f"TARGET_ROLE must be 'left' or 'right', got: {TARGET_ROLE}")

    origin_dir, _, _, origin_params = resolve_run_paths(DEFAULT_ORIGIN_RUN, role_hint="origin")
    target_dir, _, _, target_params = resolve_run_paths(DEFAULT_TARGET_RUN, role_hint=TARGET_ROLE)
    origin_name = origin_dir.name
    target_name = target_dir.name

    origin_raw = load_run_df(origin_name, role_hint="origin", include_xy=True)
    target_raw = load_run_df(target_name, role_hint=TARGET_ROLE, include_xy=True)

    origin_params_dict = load_params(origin_params)
    target_params_dict = load_params(target_params)
    print_run_params(origin_name, target_name, origin_params_dict, target_params_dict)

    origin_fz = feed_per_tooth(origin_params_dict)
    target_fz = feed_per_tooth(target_params_dict)
    scale_origin_to_target = target_fz / origin_fz

    origin_rev = full_revolution_average(origin_raw)
    target_rev = full_revolution_average(target_raw)

    s_common = common_s(origin_rev, target_rev)
    origin_i = interp_on_s(origin_rev, s_common, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    target_i = interp_on_s(target_rev, s_common, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])

    fx_origin = origin_i["fx_avg_N"].to_numpy()
    fy_origin = origin_i["fy_avg_N"].to_numpy()
    fx_target = target_i["fx_avg_N"].to_numpy()
    fy_target = target_i["fy_avg_N"].to_numpy()

    fx_origin_scaled = fx_origin * scale_origin_to_target
    fy_origin_scaled = fy_origin * scale_origin_to_target

    fx_origin_per_fz = fx_origin / origin_fz
    fy_origin_per_fz = fy_origin / origin_fz
    fx_target_per_fz = fx_target / target_fz
    fy_target_per_fz = fy_target / target_fz

    fx_rmse_raw, fx_mae_raw = rmse_mae(fx_target, fx_origin)
    fy_rmse_raw, fy_mae_raw = rmse_mae(fy_target, fy_origin)
    fx_rmse_scaled, fx_mae_scaled = rmse_mae(fx_target, fx_origin_scaled)
    fy_rmse_scaled, fy_mae_scaled = rmse_mae(fy_target, fy_origin_scaled)
    fx_rmse_norm, fx_mae_norm = rmse_mae(fx_target_per_fz, fx_origin_per_fz)
    fy_rmse_norm, fy_mae_norm = rmse_mae(fy_target_per_fz, fy_origin_per_fz)

    print("\nRaw force comparison")
    print(f"Fx RMSE={fx_rmse_raw:.4f}, MAE={fx_mae_raw:.4f}")
    print(f"Fy RMSE={fy_rmse_raw:.4f}, MAE={fy_mae_raw:.4f}")

    print("\nOrigin force rescaled to target feed per tooth")
    print(f"scale factor: {scale_origin_to_target:.6f}")
    print(f"Fx RMSE={fx_rmse_scaled:.4f}, MAE={fx_mae_scaled:.4f}")
    print(f"Fy RMSE={fy_rmse_scaled:.4f}, MAE={fy_mae_scaled:.4f}")

    print("\nForce normalized by feed per tooth")
    print(f"Fx/fz RMSE={fx_rmse_norm:.4f}, MAE={fx_mae_norm:.4f}")
    print(f"Fy/fz RMSE={fy_rmse_norm:.4f}, MAE={fy_mae_norm:.4f}")

    waypoint_xy = planned_waypoints_full_with_start(
        x_offset_mm=float(origin_params_dict.get("path_x_offset_mm", 0.0)),
        y_offset_mm=float(origin_params_dict.get("path_y_offset_mm", 0.0)),
    )
    waypoint_s = project_waypoints_to_s(waypoint_xy, origin_raw)
    waypoint_ids = np.arange(1, len(waypoint_s) + 1, dtype=int)

    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
    ax_traj.plot(origin_i["x_mm"], origin_i["y_mm"], label="Origin (rev-avg)")
    ax_traj.plot(target_i["x_mm"], target_i["y_mm"], label=f"Target {TARGET_ROLE} (rev-avg)")
    ax_traj.set(xlabel="x [mm]", ylabel="y [mm]", title="Trajectory comparison on common s")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend()
    fig_traj.tight_layout()

    plot_force_rescaling(
        s_common=s_common,
        fx_target=fx_target,
        fy_target=fy_target,
        fx_origin=fx_origin,
        fy_origin=fy_origin,
        fx_origin_scaled=fx_origin_scaled,
        fy_origin_scaled=fy_origin_scaled,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )

    plot_force_normalized(
        s_common=s_common,
        fx_target_per_fz=fx_target_per_fz,
        fy_target_per_fz=fy_target_per_fz,
        fx_origin_per_fz=fx_origin_per_fz,
        fy_origin_per_fz=fy_origin_per_fz,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )

    plot_rescaling_error(
        s_common=s_common,
        fx_target=fx_target,
        fy_target=fy_target,
        fx_est=fx_origin_scaled,
        fy_est=fy_origin_scaled,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )
    plot_rescaling_relative_error(
        s_common=s_common,
        fx_target=fx_target,
        fy_target=fy_target,
        fx_est=fx_origin_scaled,
        fy_est=fy_origin_scaled,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
