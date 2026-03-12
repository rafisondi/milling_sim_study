import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import (
    zero_order_directional_coefficients,
    zero_order_force_analytical,
)

DATASET_CSV = Path(
    "data/experiments/20260310_095500__8a5a09f72c82__rpm3333p3__feed10p000__fs20000/experiment_data.csv"
)
SETTINGS_BASE_DIR = Path("data/experiments/settings")
WORKPIECE = np.array([[0, 0], [50, 0], [50, 50], [0, 50]]).T
SHOW_PLOT = True


def binned_average(df: pd.DataFrame, bin_size: int, cols, time_col="time_s") -> pd.DataFrame:
    n_bins = len(df) // bin_size
    if n_bins <= 0:
        raise ValueError(f"Not enough samples ({len(df)}) for bin_size={bin_size}")

    df = df.iloc[: n_bins * bin_size].copy()
    arr = df[[time_col, *cols]].to_numpy(dtype=float).reshape(n_bins, bin_size, len(cols) + 1)
    avg = np.nanmean(arr, axis=1)

    out = pd.DataFrame(avg, columns=[time_col, *cols])
    out["rev_idx"] = np.arange(n_bins, dtype=int)
    out["n_samples"] = bin_size
    return out[[time_col, "rev_idx", "n_samples", *cols]]


def compute_binned_average_signal(time_s: np.ndarray, signal_2d: np.ndarray, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(
        {
            "time_s": time_s,
            "c0": signal_2d[:, 0],
            "c1": signal_2d[:, 1],
        }
    )
    averaged = binned_average(df, bin_size=bin_size, cols=["c0", "c1"])
    return averaged["time_s"].to_numpy(dtype=float), averaged[["c0", "c1"]].to_numpy(dtype=float)


def extract_settings_hash(csv_path):
    return csv_path.parent.name.split("__")[1]


def load_params(csv_path, base_dir):
    hs = extract_settings_hash(csv_path)
    p = base_dir / hs / "params.json"
    with open(p, "r") as f:
        return json.load(f)


def load_csv_columns(csv_path):
    with open(csv_path, "r") as f:
        names = f.readline().strip().split(",")
    arr = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return {names[i]: arr[:, i] for i in range(len(names))}


def estimate_radial_immersion(center_mm, tool_R):
    x = center_mm[:, 0]
    y = center_mm[:, 1]
    xR = WORKPIECE[0, -1]
    yT = WORKPIECE[1, -1]
    b = -y + tool_R + yT
    v = -(x - xR)
    mask = (v <= 0) & (b > 0) & (b < 2 * tool_R)
    if np.any(mask):
        return float(np.mean(b[mask])), mask
    return 0.0, np.zeros_like(b, dtype=bool)


def radial_immersion_per_sample(center_mm, tool_R):
    y = center_mm[:, 1]
    yT = WORKPIECE[1, -1]
    ae = -y + tool_R + yT
    return np.clip(ae, 0.0, 2.0 * tool_R)


def entry_exit_from_radial_immersion(ae, R):
    ae = float(np.clip(ae, 0.0, 2.0 * R))
    arg = np.clip(1.0 - ae / R, -1.0, 1.0)
    phi_st = np.pi - np.arccos(arg)
    phi_ex = np.pi
    return phi_st, phi_ex


def rmse_masked(true, pred, mask):
    e = pred[mask] - true[mask]
    if e.size == 0:
        return np.array([np.nan, np.nan])
    return np.sqrt(np.mean(e**2, axis=0))


def angle_in_sector(phi, phi_st, phi_ex):
    phi = phi % (2.0 * np.pi)
    phi_st = phi_st % (2.0 * np.pi)
    phi_ex = phi_ex % (2.0 * np.pi)

    if phi_st <= phi_ex:
        return phi_st <= phi <= phi_ex
    return (phi >= phi_st) or (phi <= phi_ex)


def instantaneous_directional_cutting_matrix(
    orientation,
    phi_st,
    phi_ex,
    z_teeth,
    axial_depth_mm,
    tooth_period_s,
    feed_dir,
    Kt,
    Kr,
):
    B = np.zeros((2, 2), dtype=float)
    feed_dir = np.asarray(feed_dir, dtype=float).reshape(2)
    feed_dir = feed_dir / np.linalg.norm(feed_dir)

    for j in range(int(z_teeth)):
        phi_j = (orientation + 2.0 * np.pi * j / z_teeth) % (2.0 * np.pi)

        if not angle_in_sector(phi_j, phi_st, phi_ex):
            continue

        dphi = np.array(
            [
                -Kt * np.cos(phi_j) - Kr * np.sin(phi_j),
                Kt * np.sin(phi_j) - Kr * np.cos(phi_j),
            ],
            dtype=float,
        ).reshape(2, 1)
        pphi = (np.sin(phi_j) * feed_dir).reshape(1, 2)
        B += axial_depth_mm * tooth_period_s * (dphi @ pphi)

    return B


def corrected_average_directional_matrix(a, N, phi_st, phi_ex, Ktc, Krc):
    c0 = (N * a) / (8.0 * np.pi)
    dphi = phi_ex - phi_st
    dsin2p = np.sin(2.0 * phi_ex) - np.sin(2.0 * phi_st)
    dcos2p = np.cos(2.0 * phi_ex) - np.cos(2.0 * phi_st)

    dxx = c0 * (Ktc * dcos2p - Krc * (2.0 * dphi - dsin2p))
    dyx = c0 * (Ktc * (2.0 * dphi - dsin2p) + Krc * dcos2p)
    dxy = c0 * (Ktc * (2.0 * dphi + dsin2p) - Krc * dcos2p)
    dyy = c0 * (Krc * (2.0 * dphi + dsin2p) + Ktc * dcos2p)

    return np.array([[dxx, dxy], [dyx, dyy]], dtype=float)


params = load_params(DATASET_CSV, SETTINGS_BASE_DIR)
cols = load_csv_columns(DATASET_CSV)

t = cols["time_s"]
x_hist = np.column_stack([cols["x_x_m"], cols["x_y_m"]])
fmill = np.column_stack([cols["fmill_x_N"], cols["fmill_y_N"]])
fanalyt = np.column_stack([cols["fanalyt_x_N"], cols["fanalyt_y_N"]])

tool_center_actual = np.column_stack(
    [cols["tool_center_actual_x_mm"], cols["tool_center_actual_y_mm"]]
)
tool_center_nominal = np.column_stack(
    [cols["tool_center_nominal_x_mm"], cols["tool_center_nominal_y_mm"]]
)
tool_orientation = cols["tool_orientation"]

# Structural perturbation velocity in world frame [mm/s].
udot_world = np.gradient(x_hist, t, axis=0) * 1000.0

feed_path = tool_center_nominal[-1] - tool_center_nominal[0]
feed_dir = feed_path / np.linalg.norm(feed_path)

mproc = milling_workpiece(WORKPIECE)
mproc.diameter_end_mill = float(params["tool_diameter_mm"])
mproc.number_of_teeth = int(params["z_teeth"])
mproc.radius_tool = float(params["tool_radius_mm"])
mproc.slice_height = float(params["axial_cutting_depth_mm"])

Kt = float(mproc.cutting_force_coefficient_Ktc)
Kr = float(mproc.cutting_force_coefficient_Krc)
tp = 1.0 / (params["rev_per_s"] * mproc.number_of_teeth)

ae, steady_mask = estimate_radial_immersion(tool_center_actual, mproc.radius_tool)

# Average model for revolution-binned comparison.
phi_st_avg, phi_ex_avg = entry_exit_from_radial_immersion(ae, mproc.radius_tool)
A0x, A0y = zero_order_directional_coefficients(
    a=mproc.slice_height,
    N=mproc.number_of_teeth,
    phi_st=phi_st_avg,
    phi_ex=phi_ex_avg,
    Ktc=Kt,
    Krc=Kr,
)
a0_vector = np.array([A0x, A0y], dtype=float)

Dbar = corrected_average_directional_matrix(
    a=mproc.slice_height,
    N=float(mproc.number_of_teeth),
    phi_st=float(phi_st_avg),
    phi_ex=float(phi_ex_avg),
    Ktc=Kt,
    Krc=Kr,
)

f0 = zero_order_force_analytical(
    c=float(params["feed_per_tooth_mm"]),
    a=mproc.slice_height,
    N=float(mproc.number_of_teeth),
    phi_st=float(phi_st_avg),
    phi_ex=float(phi_ex_avg),
    Ktc=Kt,
    Krc=Kr,
)

fzero = np.zeros_like(fmill)
fpert_avg = np.zeros_like(fmill)
fest_avg = np.zeros_like(fmill)

fzero[steady_mask] = f0
fpert_avg[steady_mask] = tp * (udot_world[steady_mask] @ Dbar.T)
fest_avg[steady_mask] = fzero[steady_mask] + fpert_avg[steady_mask]

# Instantaneous model for raw comparison.
fpert_inst = np.zeros_like(fmill)
fest_inst = fanalyt.copy()
inst_mask = np.zeros(len(t), dtype=bool)

for i in range(len(t)):
    ae_i = radial_immersion_per_sample(tool_center_actual[i : i + 1], mproc.radius_tool)[0]
    phi_st_i, phi_ex_i = entry_exit_from_radial_immersion(ae_i, mproc.radius_tool)

    engaged = (phi_ex_i - phi_st_i) > 0.0
    if not engaged:
        continue

    inst_mask[i] = True
    orientation_i = tool_orientation[i] + np.pi / 2.0
    B_i = instantaneous_directional_cutting_matrix(
        orientation=orientation_i,
        phi_st=phi_st_i,
        phi_ex=phi_ex_i,
        z_teeth=mproc.number_of_teeth,
        axial_depth_mm=mproc.slice_height,
        tooth_period_s=tp,
        feed_dir=feed_dir,
        Kt=Kt,
        Kr=Kr,
    )
    fpert_inst[i] = B_i @ udot_world[i]
    fest_inst[i] = fanalyt[i] + fpert_inst[i]

rmse_f0 = rmse_masked(fmill, fzero, steady_mask)
rmse_fest_avg = rmse_masked(fmill, fest_avg, steady_mask)
rmse_fpert_avg = rmse_masked(fmill - fzero, fpert_avg, steady_mask)
rmse_fanalyt = rmse_masked(fmill, fanalyt, inst_mask)
rmse_fest_inst = rmse_masked(fmill, fest_inst, inst_mask)
rmse_fpert_inst = rmse_masked(fmill - fanalyt, fpert_inst, inst_mask)

fmill_ss = np.where(steady_mask[:, None], fmill, np.nan)
fzero_ss = np.where(steady_mask[:, None], fzero, np.nan)
fpert_avg_ss = np.where(steady_mask[:, None], fpert_avg, np.nan)
fest_avg_ss = np.where(steady_mask[:, None], fest_avg, np.nan)
fanalyt_ss = np.where(inst_mask[:, None], fanalyt, np.nan)
fest_inst_ss = np.where(inst_mask[:, None], fest_inst, np.nan)
x_ss = np.where(steady_mask[:, None], x_hist, np.nan)
udot_ss = np.where(steady_mask[:, None], udot_world, np.nan)

bin_size = int(round(1.0 / max(float(params["rev_per_s"]) * float(params["dt"]), 1e-12)))
t_rev, fmill_rev = compute_binned_average_signal(t, fmill_ss, bin_size)
_, fzero_rev = compute_binned_average_signal(t, fzero_ss, bin_size)
_, fpert_avg_rev = compute_binned_average_signal(t, fpert_avg_ss, bin_size)
_, fest_avg_rev = compute_binned_average_signal(t, fest_avg_ss, bin_size)
_, fanalyt_rev = compute_binned_average_signal(t, fanalyt_ss, bin_size)
_, fest_inst_rev = compute_binned_average_signal(t, fest_inst_ss, bin_size)
_, x_rev = compute_binned_average_signal(t, x_ss, bin_size)
_, udot_rev = compute_binned_average_signal(t, udot_ss, bin_size)

steady_mask_rev = np.isfinite(fmill_rev).all(axis=1) & np.isfinite(fzero_rev).all(axis=1)
inst_mask_rev = np.isfinite(fmill_rev).all(axis=1) & np.isfinite(fest_inst_rev).all(axis=1) & np.isfinite(fanalyt_rev).all(axis=1)

rmse_f0_rev = rmse_masked(fmill_rev, fzero_rev, steady_mask_rev)
rmse_fest_avg_rev = rmse_masked(fmill_rev, fest_avg_rev, steady_mask_rev)
rmse_fpert_avg_rev = rmse_masked(fmill_rev - fzero_rev, fpert_avg_rev, steady_mask_rev)
rmse_fanalyt_rev = rmse_masked(fmill_rev, fanalyt_rev, inst_mask_rev)
rmse_fest_inst_rev = rmse_masked(fmill_rev, fest_inst_rev, inst_mask_rev)

print("Averaged zero-order coefficients a0:", a0_vector)
print("Corrected averaged directional matrix Dbar:\n", Dbar)
print("Check Dbar[:, 0] - a0:", Dbar[:, 0] - a0_vector)
print("Tooth passing period tp [s]:", tp)
print("RMSE raw vs F0:", rmse_f0)
print("RMSE raw vs F_est_avg = F0 + tp * Dbar * u_dot:", rmse_fest_avg)
print("RMSE raw delta vs tp * Dbar * u_dot:", rmse_fpert_avg)
print("RMSE raw vs fanalyt:", rmse_fanalyt)
print("RMSE raw vs fanalyt + B(t) * u_dot:", rmse_fest_inst)
print("RMSE raw delta vs B(t) * u_dot:", rmse_fpert_inst)
print("RMSE rev-avg vs F0:", rmse_f0_rev)
print("RMSE rev-avg vs F_est_avg:", rmse_fest_avg_rev)
print("RMSE rev-avg delta:", rmse_fpert_avg_rev)
print("RMSE rev-avg vs fanalyt:", rmse_fanalyt_rev)
print("RMSE rev-avg vs instantaneous estimate:", rmse_fest_inst_rev)

if SHOW_PLOT:
    labels = ["Fx", "Fy"]

    fig_inspect, ax_inspect = plt.subplots(figsize=(11, 6))
    ax_inspect.plot(t, fmill[:, 0], label="Fx simulated", alpha=0.8)
    ax_inspect.plot(t, fmill[:, 1], label="Fy simulated", alpha=0.8)
    ax_inspect.plot(t, fanalyt[:, 0], "--", label="Fx analytical", linewidth=1.5)
    ax_inspect.plot(t, fanalyt[:, 1], "--", label="Fy analytical", linewidth=1.5)
    ax_inspect.plot(t, fzero[:, 0], ":", label="Fx zero-order", linewidth=2.0)
    ax_inspect.plot(t, fzero[:, 1], ":", label="Fy zero-order", linewidth=2.0)
    ax_inspect.plot(t_rev, fmill_rev[:, 0], "-.", linewidth=2.0, label="Fx simulated (avg/rev)")
    ax_inspect.plot(t_rev, fmill_rev[:, 1], "-.", linewidth=2.0, label="Fy simulated (avg/rev)")
    ax_inspect.set_xlabel("Time [s]")
    ax_inspect.set_ylabel("Force [N]")
    ax_inspect.set_title("Milling force signals")
    ax_inspect.grid(True)
    ax_inspect.legend()
    fig_inspect.tight_layout()

    fig_raw, ax_raw = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    for i in range(2):
        ax_raw[i].plot(t, fmill[:, i], label="measured raw", linewidth=1.0)
        ax_raw[i].plot(t, fanalyt[:, i], ":", label="analytical raw", linewidth=1.1)
        ax_raw[i].plot(t, fest_inst[:, i], "-.", label="instantaneous estimate", linewidth=1.2)
        ax_raw[i].set_ylabel(f"{labels[i]} [N]")
        ax_raw[i].grid(alpha=0.3)
    ax_raw[-1].set_xlabel("time [s]")
    ax_raw[0].legend(loc="upper left")
    fig_raw.suptitle("Raw measured forces vs instantaneous perturbation model")
    fig_raw.tight_layout()

    fig_rev, ax_rev = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    for i in range(2):
        ax_rev[i].plot(t_rev, fmill_rev[:, i], label="measured avg/rev", linewidth=1.8)
        ax_rev[i].plot(t_rev, fzero_rev[:, i], "--", label="zero-order avg/rev", linewidth=1.5)
        ax_rev[i].plot(t_rev, fest_avg_rev[:, i], "-.", label="averaged estimate", linewidth=1.5)
        ax_rev[i].plot(t_rev, fest_inst_rev[:, i], ":", label="instantaneous estimate avg/rev", linewidth=1.3)
        ax_rev[i].set_ylabel(f"{labels[i]} [N]")
        ax_rev[i].grid(alpha=0.3)
    ax_rev[-1].set_xlabel("time [s]")
    ax_rev[0].legend(loc="upper left")
    fig_rev.suptitle("Revolution-binned force comparison")
    fig_rev.tight_layout()

    fig_u, ax_u = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    for i in range(2):
        ax_u[i].plot(t, udot_world[:, i], color="0.75", linewidth=0.8, label="u_dot raw")
        ax_u[i].plot(t_rev, udot_rev[:, i], color="0.15", linewidth=1.4, label="u_dot avg/rev")
        ax_u[i].set_ylabel(f"u_dot_{'xy'[i]} [mm/s]")
        ax_u[i].grid(alpha=0.3)
    ax_u[-1].set_xlabel("time [s]")
    ax_u[0].legend(loc="upper left")
    fig_u.suptitle("Robot perturbation velocity")
    fig_u.tight_layout()

    fig_x, ax_x = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    for i in range(2):
        ax_x[i].plot(t, x_hist[:, i] * 1e3, color="0.75", linewidth=0.8, label="oscillation raw")
        ax_x[i].plot(t_rev, x_rev[:, i] * 1e3, color="0.15", linewidth=1.4, label="oscillation avg/rev")
        ax_x[i].set_ylabel(f"x_{'xy'[i]} [mm]")
        ax_x[i].grid(alpha=0.3)
    ax_x[-1].set_xlabel("time [s]")
    ax_x[0].legend(loc="upper left")
    fig_x.suptitle("Robot oscillation")
    fig_x.tight_layout()

    plt.show()
