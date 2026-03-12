import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from milling_data import (
    EXPERIMENTS_DIR,
    binned_average,
    load_json_dict,
    load_run_df,
    planned_waypoints_full_with_start,
    project_waypoints_to_s,
    resolve_run_paths,
    samples_per_revolution,
)

GRADIENT_DIR = EXPERIMENTS_DIR / "gradients" / "20260307_123434__origin_20260306_211831__left_20260306_221950__right_20260306_232207"

DEFAULT_ORIGIN_RUN = "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000"
DEFAULT_TARGET_RUN = "20260307_151514__1b6c818b8446__rpm833p3__feed40p000__fs10000"
# DEFAULT_TARGET_RUN = "20260308_220538__b19f751fa5fe__rpm833p3__feed40p000__fs10000" 
# DEFAULT_TARGET_RUN = "20260308_190829__1d608e3cdf47__rpm833p3__feed40p000__fs10000"
TARGET_ROLE = "left"  # "left" or "right"
TARGET_OFFSET_MM = 0.3
SHOW_PLOTS = True
REL_ERR_MIN_MEAS_N = 1.0
MAX_WAYPOINT_LABELS = 52


def load_gradient_bundle(gradient_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gxy_path = gradient_dir / "gradients_xy.csv"
    gs_path = gradient_dir / "gradients_s.csv"

    gxy = pd.read_csv(gxy_path)
    gs = pd.read_csv(gs_path)
    return gxy, gs

def full_revolution_average(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    bin_size = samples_per_revolution(params)
    return binned_average(
        df,
        bin_size=bin_size,
        cols=["fmill_x_N", "fmill_y_N"],
        extra_cols=["s_mm", "x_mm", "y_mm"],
    ).rename(columns={"fmill_x_N": "fx_avg_N", "fmill_y_N": "fy_avg_N"})


def interp_on_s(df: pd.DataFrame, s_common: np.ndarray, cols: list[str]) -> pd.DataFrame:
    base = df.sort_values("s_mm").groupby("s_mm", as_index=False).mean(numeric_only=True)
    out = pd.DataFrame({"s_mm": s_common})
    for col in cols:
        out[col] = np.interp(s_common, base["s_mm"].to_numpy(), base[col].to_numpy())
    return out


def common_s(origin_rev: pd.DataFrame, target_rev: pd.DataFrame, gxy: pd.DataFrame) -> np.ndarray:
    s_min = max(origin_rev["s_mm"].min(), target_rev["s_mm"].min(), gxy["s_mm"].min())
    s_max = min(origin_rev["s_mm"].max(), target_rev["s_mm"].max(), gxy["s_mm"].max())
    n = min(len(origin_rev), len(target_rev), len(gxy))
    return np.linspace(s_min, s_max, n)


def rmse_mae(meas: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    err = meas - pred
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def load_params(path: Path) -> dict:
    return load_json_dict(path, missing_ok=True)


def print_run_params(origin_name: str, target_name: str, origin_params: dict, target_params: dict):
    print("\nOrigin run params")
    print(f"run_name: {origin_name}")
    for key in sorted(origin_params.keys()):
        print(f"  {key}: {origin_params[key]}")

    print("\nTarget run params")
    print(f"run_name: {target_name}")
    for key in sorted(target_params.keys()):
        print(f"  {key}: {target_params[key]}")


def resolve_offset_mm() -> float:
    offset = float(TARGET_OFFSET_MM)
    if offset <= 0:
        raise ValueError(f"TARGET_OFFSET_MM must be > 0, got {TARGET_OFFSET_MM}")
    return offset

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


def plot_force_comparison(
    title: str,
    pred_label: str,
    s_common: np.ndarray,
    fx_meas: np.ndarray,
    fy_meas: np.ndarray,
    fx_pred: np.ndarray,
    fy_pred: np.ndarray,
    fx_origin: np.ndarray,
    fy_origin: np.ndarray,
    err_mag: np.ndarray,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    axes[0].plot(s_common, fx_meas, label="Fx measured target")
    axes[0].plot(s_common, fx_pred, "--", label=f"Fx pred ({pred_label})")
    axes[0].plot(s_common, fx_origin, alpha=0.8, label="Fx origin")
    axes[0].set_ylabel("Fx [N]")
    add_waypoint_axis(axes[0], waypoint_s, waypoint_ids)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(s_common, fy_meas, label="Fy measured target")
    axes[1].plot(s_common, fy_pred, "--", label=f"Fy pred ({pred_label})")
    axes[1].plot(s_common, fy_origin, alpha=0.8, label="Fy origin")
    axes[1].set_ylabel("Fy [N]")
    add_waypoint_axis(axes[1], waypoint_s, waypoint_ids)
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(s_common, err_mag, label=f"|error| {pred_label}")
    axes[2].set(xlabel="s [mm]", ylabel="|error| [N]")
    add_waypoint_axis(axes[2], waypoint_s, waypoint_ids)
    axes[2].grid(True)
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()


def plot_relative_error(label: str, s_common: np.ndarray, rel: np.ndarray, waypoint_s: np.ndarray, waypoint_ids: np.ndarray):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s_common, 100.0 * rel, label=f"RAE |F| {label} [%]")
    ax.set(
        xlabel="s [mm]",
        ylabel="Relative abs error [%]",
        title=f"Relative error {label} (|F_meas| >= {REL_ERR_MIN_MEAS_N} N)",
    )
    add_waypoint_axis(ax, waypoint_s, waypoint_ids)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()


def main():
    if TARGET_ROLE not in ("left", "right"):
        raise ValueError(f"TARGET_ROLE must be 'left' or 'right', got: {TARGET_ROLE}")

    gxy, gs = load_gradient_bundle(GRADIENT_DIR)

    origin_dir, _, _, origin_params = resolve_run_paths(DEFAULT_ORIGIN_RUN, role_hint="origin")
    target_dir, _, _, target_params = resolve_run_paths(DEFAULT_TARGET_RUN, role_hint=TARGET_ROLE)
    origin_name = origin_dir.name
    target_name = target_dir.name

    origin_raw = load_run_df(origin_name, role_hint="origin", include_xy=True)
    target_raw = load_run_df(target_name, role_hint=TARGET_ROLE, include_xy=True)
    origin_params_dict = load_params(origin_params)
    target_params_dict = load_params(target_params)
    print_run_params(origin_name, target_name, origin_params_dict, target_params_dict)

    origin_rev = full_revolution_average(origin_raw, origin_params_dict)
    target_rev = full_revolution_average(target_raw, target_params_dict)

    s_common = common_s(origin_rev, target_rev, gxy)

    origin_i = interp_on_s(origin_rev, s_common, ["time_s", "x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    target_i = interp_on_s(target_rev, s_common, ["time_s", "x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    gxy_i = interp_on_s(gxy, s_common, ["dFx_dx_N_per_mm", "dFx_dy_N_per_mm", "dFy_dx_N_per_mm", "dFy_dy_N_per_mm"])
    gs_i = interp_on_s(gs, s_common, ["dFx_dn_N_per_mm", "dFy_dn_N_per_mm"])

    dx_mm = target_i["x_mm"].to_numpy() - origin_i["x_mm"].to_numpy()
    dy_mm = target_i["y_mm"].to_numpy() - origin_i["y_mm"].to_numpy()

    fx_origin = origin_i["fx_avg_N"].to_numpy()
    fy_origin = origin_i["fy_avg_N"].to_numpy()
    fx_meas = target_i["fx_avg_N"].to_numpy()
    fy_meas = target_i["fy_avg_N"].to_numpy()

    fx_pred_xy = fx_origin + gxy_i["dFx_dx_N_per_mm"].to_numpy() * dx_mm + gxy_i["dFx_dy_N_per_mm"].to_numpy() * dy_mm
    fy_pred_xy = fy_origin + gxy_i["dFy_dx_N_per_mm"].to_numpy() * dx_mm + gxy_i["dFy_dy_N_per_mm"].to_numpy() * dy_mm

    offset_mm = resolve_offset_mm()
    sign = 1.0 if TARGET_ROLE == "left" else -1.0
    fx_pred_n = fx_origin + sign * gs_i["dFx_dn_N_per_mm"].to_numpy() * offset_mm
    fy_pred_n = fy_origin + sign * gs_i["dFy_dn_N_per_mm"].to_numpy() * offset_mm

    fx_rmse_xy, fx_mae_xy = rmse_mae(fx_meas, fx_pred_xy)
    fy_rmse_xy, fy_mae_xy = rmse_mae(fy_meas, fy_pred_xy)
    fx_rmse_n, fx_mae_n = rmse_mae(fx_meas, fx_pred_n)
    fy_rmse_n, fy_mae_n = rmse_mae(fy_meas, fy_pred_n)

    dnorm = np.hypot(dx_mm, dy_mm)
    mean_signed_normal = float(sign * np.nanmean(dnorm))

    print(f"Gradient bundle: {GRADIENT_DIR}")
    print(f"Origin run: {origin_name}")
    print(f"Target run ({TARGET_ROLE}): {target_name}")
    print(f"Aligned points: {len(s_common)}")
    print(f"Configured offset: {offset_mm:.6f} mm")
    print(f"Observed mean signed displacement: {mean_signed_normal:.6f} mm")
    print(f"XY prediction errors: Fx RMSE={fx_rmse_xy:.4f}, MAE={fx_mae_xy:.4f} | Fy RMSE={fy_rmse_xy:.4f}, MAE={fy_mae_xy:.4f}")
    print(f"Normal prediction errors: Fx RMSE={fx_rmse_n:.4f}, MAE={fx_mae_n:.4f} | Fy RMSE={fy_rmse_n:.4f}, MAE={fy_mae_n:.4f}")

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

    err_xy_mag = np.hypot(fx_meas - fx_pred_xy, fy_meas - fy_pred_xy)
    err_n_mag = np.hypot(fx_meas - fx_pred_n, fy_meas - fy_pred_n)

    plot_force_comparison(
        title="First-order expansion using XY Jacobian gradients",
        pred_label="XY Jacobian",
        s_common=s_common,
        fx_meas=fx_meas,
        fy_meas=fy_meas,
        fx_pred=fx_pred_xy,
        fy_pred=fy_pred_xy,
        fx_origin=fx_origin,
        fy_origin=fy_origin,
        err_mag=err_xy_mag,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )
    plot_force_comparison(
        title="First-order expansion using normal s-gradient",
        pred_label="normal gradient",
        s_common=s_common,
        fx_meas=fx_meas,
        fy_meas=fy_meas,
        fx_pred=fx_pred_n,
        fy_pred=fy_pred_n,
        fx_origin=fx_origin,
        fy_origin=fy_origin,
        err_mag=err_n_mag,
        waypoint_s=waypoint_s,
        waypoint_ids=waypoint_ids,
    )

    f_meas_mag = np.hypot(fx_meas, fy_meas)
    rel_xy = np.full_like(f_meas_mag, np.nan, dtype=float)
    rel_n = np.full_like(f_meas_mag, np.nan, dtype=float)
    valid = f_meas_mag >= REL_ERR_MIN_MEAS_N
    rel_xy[valid] = err_xy_mag[valid] / f_meas_mag[valid]
    rel_n[valid] = err_n_mag[valid] / f_meas_mag[valid]

    plot_relative_error("XY Jacobian", s_common, rel_xy, waypoint_s, waypoint_ids)
    plot_relative_error("normal gradient", s_common, rel_n, waypoint_s, waypoint_ids)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
