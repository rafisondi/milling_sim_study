import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
GRADIENT_DIR = EXPERIMENTS_DIR / "gradients" / "20260307_123434__origin_20260306_211831__left_20260306_221950__right_20260306_232207"

DEFAULT_ORIGIN_RUN = "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000"
# DEFAULT_TARGET_RUN = "20260308_123323__9cc66a5310ea__rpm833p3__feed60p000__fs10000"
# DEFAULT_TARGET_RUN = "20260308_190829__1d608e3cdf47__rpm833p3__feed40p000__fs10000"
DEFAULT_TARGET_RUN = "20260309_004050__f7a1e02cd781__rpm1666p7__feed40p000__fs10000"
TARGET_ROLE = "left"  # "left" or "right"
TARGET_OFFSET_MM = 0.3
SHOW_PLOTS = True
REL_ERR_MIN_MEAS_N = 1.0
WORKPIECE_WIDTH_Y_MM = 150.0
MAX_WAYPOINT_LABELS = 52
ENABLE_OSCILLATION_FORCE_PLOT = True


def load_gradient_bundle(gradient_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    gxy_path = gradient_dir / "gradients_xy.csv"
    gs_path = gradient_dir / "gradients_s.csv"

    gxy = pd.read_csv(gxy_path)
    gs = pd.read_csv(gs_path)
    return gxy, gs


def resolve_run_paths(role: str, run_name: str) -> tuple[str, Path, Path, Path]:
    exp_dir = EXPERIMENTS_DIR / run_name
    exp_csv = exp_dir / "experiment_data.csv"
    trace_candidates = sorted(exp_dir.glob(f"path_trace_{role}.csv")) or sorted(exp_dir.glob("path_trace*.csv"))
    trace_csv = trace_candidates[0] if trace_candidates else exp_dir / f"path_trace_{role}.csv"
    settings_hash = run_name.split("__")[1]
    params_json = EXPERIMENTS_DIR / "settings" / settings_hash / "params.json"
    return run_name, exp_csv, trace_csv, params_json


def load_run(exp_csv: Path, trace_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(exp_csv)
    tr = pd.read_csv(trace_csv)

    if len(tr) == len(df):
        df["s_mm"] = tr["position_along_path_mm"].to_numpy(dtype=float)
        df["x_mm"] = tr["tool_center_x_mm"].to_numpy(dtype=float)
        df["y_mm"] = tr["tool_center_y_mm"].to_numpy(dtype=float)
        return df

    t = df["time_s"].to_numpy(dtype=float)
    df["s_mm"] = np.interp(t, tr["time_s"].to_numpy(dtype=float), tr["position_along_path_mm"].to_numpy(dtype=float))
    df["x_mm"] = np.interp(t, tr["time_s"].to_numpy(dtype=float), tr["tool_center_x_mm"].to_numpy(dtype=float))
    df["y_mm"] = np.interp(t, tr["time_s"].to_numpy(dtype=float), tr["tool_center_y_mm"].to_numpy(dtype=float))
    return df


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
    return json.load(open(path, encoding="utf-8"))


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
    for key in sorted(origin_params.keys()):
        print(f"  {key}: {origin_params[key]}")
    print(f"  feed_per_tooth_mm: {origin_fz}")

    print("\nTarget run params")
    print(f"run_name: {target_name}")
    for key in sorted(target_params.keys()):
        print(f"  {key}: {target_params[key]}")
    print(f"  feed_per_tooth_mm: {target_fz}")

    print("\nFeed-per-tooth ratio")
    print(f"target / origin: {target_fz / origin_fz}")


def to_arrays(df: pd.DataFrame, cols: list[str]) -> list[np.ndarray]:
    return [df[col].to_numpy(dtype=float) for col in cols]


def first_order_xy_prediction(
    fx0: np.ndarray,
    fy0: np.ndarray,
    dfx_dx: np.ndarray,
    dfx_dy: np.ndarray,
    dfy_dx: np.ndarray,
    dfy_dy: np.ndarray,
    dx_mm: np.ndarray,
    dy_mm: np.ndarray,
    origin_fz: float,
    target_fz: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = target_fz / origin_fz
    fx = (fx0 + dfx_dx * dx_mm + dfx_dy * dy_mm) * scale
    fy = (fy0 + dfy_dx * dx_mm + dfy_dy * dy_mm) * scale
    return fx, fy


def signed_normal_displacement_mm(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    dx_mm: np.ndarray,
    dy_mm: np.ndarray,
) -> np.ndarray:
    tx = np.gradient(x_ref)
    ty = np.gradient(y_ref)
    t_norm = np.hypot(tx, ty)
    t_norm[t_norm == 0.0] = 1.0
    tx /= t_norm
    ty /= t_norm
    # Left normal of the path tangent.
    nx = -ty
    ny = tx
    return dx_mm * nx + dy_mm * ny


def relative_error_curve(fx_meas: np.ndarray, fy_meas: np.ndarray, err_mag: np.ndarray, min_meas_n: float) -> np.ndarray:
    f_meas_mag = np.hypot(fx_meas, fy_meas)
    rel = np.full_like(f_meas_mag, np.nan, dtype=float)
    valid = f_meas_mag >= min_meas_n
    rel[valid] = err_mag[valid] / f_meas_mag[valid]
    return rel


def nominal_actual_xy(df: pd.DataFrame, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = {
        "tool_center_nominal_x_mm",
        "tool_center_nominal_y_mm",
        "tool_center_actual_x_mm",
        "tool_center_actual_y_mm",
    }
    if cols.issubset(df.columns):
        x_nom, y_nom, x_act, y_act = to_arrays(
            df.loc[mask],
            ["tool_center_nominal_x_mm", "tool_center_nominal_y_mm", "tool_center_actual_x_mm", "tool_center_actual_y_mm"],
        )
        return x_nom, y_nom, x_act, y_act

    x_nom, y_nom = to_arrays(df.loc[mask], ["x_mm", "y_mm"])
    return x_nom, y_nom, x_nom, y_nom


def planned_waypoints_full_with_start(x_offset_mm: float = 0.0, y_offset_mm: float = 0.0) -> np.ndarray:
    path_coordinates = np.array(
        [
            [-20.0, 7.0], [0.0, 7.0], [20.0, 8.0], [30.0, 6.5], [40.0, 7.5], [65.0, 0.0], [80.0, 1.5],
            [85.0, 3.5], [92.0, 4.0], [98.0, 2.0], [105.0, -4.0], [105.0, -18.0], [95.0, -25.0], [70.0, -27.0],
            [40.0, -28.0], [40.0, -30.0], [45.0, -31.0], [85.0, -33.0], [93.0, -40.0], [95.0, -50.0], [95.0, -60.0],
        ],
        dtype=float,
    )
    path_start_coordinates = np.array(
        [
            [5.0, -190.0], [5.0, -150.0], [5.0, -100.0], [5.0, -50.0], [5.0, 0.0],
            [5.0, 5.0], [3.0, 23.0], [-16.0, 30.0], [-30.0, 21.0], [-30.0, 12.0],
        ],
        dtype=float,
    )

    mirrored = np.array([1.0, -1.0]) * np.flip(path_coordinates, axis=0) - np.array([0.0, WORKPIECE_WIDTH_Y_MM])
    waypoints = np.concatenate([path_start_coordinates, path_coordinates, mirrored], axis=0)
    waypoints[:, 0] += float(x_offset_mm)
    waypoints[:, 1] += float(y_offset_mm)
    return waypoints


def project_waypoints_to_origin_s(waypoint_xy: np.ndarray, origin_raw: pd.DataFrame) -> np.ndarray:
    trace_xy = origin_raw[["x_mm", "y_mm"]].to_numpy(dtype=float)
    trace_s = origin_raw["s_mm"].to_numpy(dtype=float)

    s_i = np.zeros(len(waypoint_xy), dtype=float)
    for i, wp in enumerate(waypoint_xy):
        d2 = np.sum((trace_xy - wp) ** 2, axis=1)
        s_i[i] = trace_s[int(np.argmin(d2))]

    return np.maximum.accumulate(s_i)


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


def plot_time_force_with_oscillation_estimate(
    t_raw: np.ndarray,
    fx_raw: np.ndarray,
    fy_raw: np.ndarray,
    fx_est: np.ndarray,
    fy_est: np.ndarray,
    t_rev: np.ndarray,
    fx_rev: np.ndarray,
    fy_rev: np.ndarray,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(t_raw, fx_raw, alpha=0.55, label="Fx raw target")
    axes[0].plot(t_raw, fx_est, linewidth=1.3, label="Fx estimated from actual tool position")
    axes[0].plot(t_rev, fx_rev, "k.", markersize=3, label="Fx target rev-avg")
    axes[0].set_ylabel("Fx [N]")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(t_raw, fy_raw, alpha=0.55, label="Fy raw target")
    axes[1].plot(t_raw, fy_est, linewidth=1.3, label="Fy estimated from actual tool position")
    axes[1].plot(t_rev, fy_rev, "k.", markersize=3, label="Fy target rev-avg")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("Fy [N]")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Raw force vs oscillation-based estimated force")
    fig.tight_layout()


def main():
    offset_mm = float(TARGET_OFFSET_MM)

    gxy, gs = load_gradient_bundle(GRADIENT_DIR)
    origin_name, origin_exp, origin_trace, origin_params = resolve_run_paths("origin", DEFAULT_ORIGIN_RUN)
    target_name, target_exp, target_trace, target_params = resolve_run_paths(TARGET_ROLE, DEFAULT_TARGET_RUN)

    origin_raw = load_run(origin_exp, origin_trace)
    target_raw = load_run(target_exp, target_trace)
    origin_params_dict = load_params(origin_params)
    target_params_dict = load_params(target_params)
    print_run_params(origin_name, target_name, origin_params_dict, target_params_dict)

    origin_fz = feed_per_tooth(origin_params_dict)
    target_fz = feed_per_tooth(target_params_dict)
    scale_fz = target_fz / origin_fz

    origin_rev = full_revolution_average(origin_raw)
    target_rev = full_revolution_average(target_raw)
    s_common = common_s(origin_rev, target_rev, gxy)

    origin_i = interp_on_s(origin_rev, s_common, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    target_i = interp_on_s(target_rev, s_common, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    gxy_i = interp_on_s(gxy, s_common, ["dFx_dx_N_per_mm", "dFx_dy_N_per_mm", "dFy_dx_N_per_mm", "dFy_dy_N_per_mm"])
    gs_i = interp_on_s(gs, s_common, ["dFx_dn_N_per_mm", "dFy_dn_N_per_mm"])

    ox, oy, fx_origin, fy_origin = to_arrays(origin_i, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    tx, ty, fx_meas, fy_meas = to_arrays(target_i, ["x_mm", "y_mm", "fx_avg_N", "fy_avg_N"])
    dfx_dx, dfx_dy, dfy_dx, dfy_dy = to_arrays(gxy_i, ["dFx_dx_N_per_mm", "dFx_dy_N_per_mm", "dFy_dx_N_per_mm", "dFy_dy_N_per_mm"])
    dfx_dn, dfy_dn = to_arrays(gs_i, ["dFx_dn_N_per_mm", "dFy_dn_N_per_mm"])

    dx_mm = tx - ox
    dy_mm = ty - oy
    dn_signed_mm = signed_normal_displacement_mm(ox, oy, dx_mm, dy_mm)
    fx_pred_xy, fy_pred_xy = first_order_xy_prediction(
        fx_origin, fy_origin, dfx_dx, dfx_dy, dfy_dx, dfy_dy, dx_mm, dy_mm, origin_fz, target_fz
    )
    fx_pred_n = (fx_origin + dfx_dn * dn_signed_mm) * scale_fz
    fy_pred_n = (fy_origin + dfy_dn * dn_signed_mm) * scale_fz

    fx_rmse_xy, fx_mae_xy = rmse_mae(fx_meas, fx_pred_xy)
    fy_rmse_xy, fy_mae_xy = rmse_mae(fy_meas, fy_pred_xy)
    fx_rmse_n, fx_mae_n = rmse_mae(fx_meas, fx_pred_n)
    fy_rmse_n, fy_mae_n = rmse_mae(fy_meas, fy_pred_n)

    print(f"Gradient bundle: {GRADIENT_DIR}")
    print(f"Origin run: {origin_name}")
    print(f"Target run ({TARGET_ROLE}): {target_name}")
    print(f"Aligned points: {len(s_common)}")
    print(f"Configured nominal offset (unused in signed-normal mode): {offset_mm:.6f} mm")
    print(f"Observed mean signed normal displacement: {float(np.nanmean(dn_signed_mm)):.6f} mm")
    print(f"Observed std signed normal displacement: {float(np.nanstd(dn_signed_mm)):.6f} mm")
    print(f"Applied fz normalization/rescaling: origin={origin_fz:.6f}, target={target_fz:.6f}, ratio={scale_fz:.6f}")
    print(f"XY prediction errors: Fx RMSE={fx_rmse_xy:.4f}, MAE={fx_mae_xy:.4f} | Fy RMSE={fy_rmse_xy:.4f}, MAE={fy_mae_xy:.4f}")
    print(f"Normal prediction errors: Fx RMSE={fx_rmse_n:.4f}, MAE={fx_mae_n:.4f} | Fy RMSE={fy_rmse_n:.4f}, MAE={fy_mae_n:.4f}")

    waypoint_xy = planned_waypoints_full_with_start(
        x_offset_mm=float(origin_params_dict.get("path_x_offset_mm", 0.0)),
        y_offset_mm=float(origin_params_dict.get("path_y_offset_mm", 0.0)),
    )
    waypoint_s = project_waypoints_to_origin_s(waypoint_xy, origin_raw)
    waypoint_ids = np.arange(1, len(waypoint_s) + 1, dtype=int)

    fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
    ax_traj.plot(ox, oy, label="Origin (rev-avg)")
    ax_traj.plot(tx, ty, label=f"Target {TARGET_ROLE} (rev-avg)")
    ax_traj.set(xlabel="x [mm]", ylabel="y [mm]", title="Trajectory comparison on common s")
    ax_traj.axis("equal")
    ax_traj.grid(True)
    ax_traj.legend()
    fig_traj.tight_layout()

    err_xy_mag = np.hypot(fx_meas - fx_pred_xy, fy_meas - fy_pred_xy)
    err_n_mag = np.hypot(fx_meas - fx_pred_n, fy_meas - fy_pred_n)

    plot_force_comparison(
        title="First-order expansion using XY Jacobian gradients (origin normalized by fz, then rescaled to target fz)",
        pred_label="XY Jacobian + fz scaling",
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
        title="First-order expansion using normal s-gradient (origin normalized by fz, then rescaled to target fz)",
        pred_label="normal gradient + fz scaling",
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

    rel_xy = relative_error_curve(fx_meas, fy_meas, err_xy_mag, REL_ERR_MIN_MEAS_N)
    rel_n = relative_error_curve(fx_meas, fy_meas, err_n_mag, REL_ERR_MIN_MEAS_N)
    plot_relative_error("XY Jacobian", s_common, rel_xy, waypoint_s, waypoint_ids)
    plot_relative_error("normal gradient", s_common, rel_n, waypoint_s, waypoint_ids)

    if ENABLE_OSCILLATION_FORCE_PLOT:
        s_raw, t_raw_full, fx_raw_full, fy_raw_full = to_arrays(target_raw, ["s_mm", "time_s", "fmill_x_N", "fmill_y_N"])
        valid_raw = (s_raw >= max(origin_raw["s_mm"].min(), gxy["s_mm"].min())) & (s_raw <= min(origin_raw["s_mm"].max(), gxy["s_mm"].max()))
        t_raw = t_raw_full[valid_raw]
        s_eval = s_raw[valid_raw]
        fx_raw = fx_raw_full[valid_raw]
        fy_raw = fy_raw_full[valid_raw]
        x_nom, y_nom, x_act, y_act = nominal_actual_xy(target_raw, valid_raw)

        origin_path_i = interp_on_s(origin_raw[["s_mm", "x_mm", "y_mm"]], s_eval, ["x_mm", "y_mm"])
        origin_force_i = interp_on_s(origin_rev, s_eval, ["fx_avg_N", "fy_avg_N"])
        gxy_raw_i = interp_on_s(gxy, s_eval, ["dFx_dx_N_per_mm", "dFx_dy_N_per_mm", "dFy_dx_N_per_mm", "dFy_dy_N_per_mm"])

        ox_raw, oy_raw = to_arrays(origin_path_i, ["x_mm", "y_mm"])
        fx0_raw, fy0_raw = to_arrays(origin_force_i, ["fx_avg_N", "fy_avg_N"])
        dfx_dx_raw, dfx_dy_raw, dfy_dx_raw, dfy_dy_raw = to_arrays(
            gxy_raw_i, ["dFx_dx_N_per_mm", "dFx_dy_N_per_mm", "dFy_dx_N_per_mm", "dFy_dy_N_per_mm"]
        )

        fx_est, fy_est = first_order_xy_prediction(
            fx0_raw,
            fy0_raw,
            dfx_dx_raw,
            dfx_dy_raw,
            dfy_dx_raw,
            dfy_dy_raw,
            x_act - ox_raw,
            y_act - oy_raw,
            origin_fz,
            target_fz,
        )
        dx_osc_mm = x_act - x_nom
        dy_osc_mm = y_act - y_nom
        fx_rmse_raw_xy, fx_mae_raw_xy = rmse_mae(fx_raw, fx_est)
        fy_rmse_raw_xy, fy_mae_raw_xy = rmse_mae(fy_raw, fy_est)

        print("\nRaw-time estimate with robot oscillation")
        print(f"Aligned raw points: {len(t_raw)}")
        print(f"Mean oscillation dx,dy [mm]: {np.mean(dx_osc_mm):.6f}, {np.mean(dy_osc_mm):.6f}")
        print(f"Std oscillation dx,dy [mm]: {np.std(dx_osc_mm):.6f}, {np.std(dy_osc_mm):.6f}")
        print(f"Raw XY estimate errors: Fx RMSE={fx_rmse_raw_xy:.4f}, MAE={fx_mae_raw_xy:.4f} | Fy RMSE={fy_rmse_raw_xy:.4f}, MAE={fy_mae_raw_xy:.4f}")

        t_rev, fx_rev, fy_rev = to_arrays(target_rev, ["time_s", "fx_avg_N", "fy_avg_N"])
        plot_time_force_with_oscillation_estimate(t_raw, fx_raw, fy_raw, fx_est, fy_est, t_rev, fx_rev, fy_rev)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
