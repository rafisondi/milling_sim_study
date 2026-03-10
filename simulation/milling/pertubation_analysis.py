import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import (
    zero_order_directional_coefficients,
    zero_order_force_analytical,
)

DATASET_CSV = Path(
    "data/experiments/20260306_152249__ef4201c3b017__rpm833p3__feed20p000__fs10000/experiment_data.csv"
)
SETTINGS_BASE_DIR = Path("data/experiments/settings")
WORKPIECE = np.array([[0, 0], [50, 0], [50, 50], [0, 50]]).T
SHOW_PLOT = True


def compute_revolution_average_from_angle(angle_rad, signal_2d):
    angle_rad = angle_rad.reshape(-1)
    signal_2d = signal_2d.reshape(len(signal_2d), 2)
    ang = np.unwrap(angle_rad)
    rev_idx = np.floor((ang - ang[0]) / (2 * np.pi)).astype(int)
    out = np.full_like(signal_2d, np.nan)
    for r in np.unique(rev_idx):
        m = rev_idx == r
        out[m] = np.nanmean(signal_2d[m], axis=0)
    return out, rev_idx


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


def entry_exit_from_radial_immersion(ae, R):
    ae = float(np.clip(ae, 0, 2 * R))
    arg = np.clip(1 - ae / R, -1, 1)
    phi_st = np.pi - np.arccos(arg)
    phi_ex = np.pi
    return phi_st, phi_ex


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


def rmse_masked(true, pred, mask):
    e = pred[mask] - true[mask]
    if e.size == 0:
        return np.array([np.nan, np.nan])
    return np.sqrt(np.mean(e**2, axis=0))


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


# World-frame structural velocity (mm/s)
xdot_world = np.gradient(x_hist, t, axis=0) * 1000.0

# Infer the feed direction from the nominal tool-center path.
feed_path = tool_center_nominal[-1] - tool_center_nominal[0]
feed_path_norm = np.linalg.norm(feed_path)
if feed_path_norm <= 0.0:
    raise ValueError("Nominal tool-center path has zero length; cannot infer feed direction.")
feed_dir = feed_path / feed_path_norm


mproc = milling_workpiece(WORKPIECE)
mproc.diameter_end_mill = float(params["tool_diameter_mm"])
mproc.number_of_teeth = int(params["z_teeth"])
mproc.radius_tool = float(params["tool_radius_mm"])
mproc.slice_height = float(params["axial_cutting_depth_mm"])

ae, steady_mask = estimate_radial_immersion(tool_center_actual, mproc.radius_tool)
phi_st, phi_ex = entry_exit_from_radial_immersion(ae, mproc.radius_tool)

# Directional coefficients
a0x, a0y = zero_order_directional_coefficients(
    a=mproc.slice_height,
    N=mproc.number_of_teeth,
    phi_st=phi_st,
    phi_ex=phi_ex,
    Ktc=mproc.cutting_force_coefficient_Ktc,
    Krc=mproc.cutting_force_coefficient_Krc,
)
a0_vector = np.array([a0x, a0y], dtype=float)

# Tooth passing period
tp = 1.0 / (params["rev_per_s"] * mproc.number_of_teeth)

# Dynamic feed-per-tooth variation from structural velocity projected
# onto the feed direction. The sin(phi) term is already embedded in a0.
delta_c = tp * (xdot_world @ feed_dir)

# Zero-order force
f0 = zero_order_force_analytical(
    c=float(params["feed_per_tooth_mm"]),
    a=mproc.slice_height,
    N=float(mproc.number_of_teeth),
    phi_st=float(phi_st),
    phi_ex=float(phi_ex),
    Ktc=mproc.cutting_force_coefficient_Ktc,
    Krc=mproc.cutting_force_coefficient_Krc,
)

# Build predictions
fzero = np.zeros_like(fmill)
fpert = np.zeros_like(fmill)
fest = np.zeros_like(fmill)

fzero[steady_mask] = f0
fpert[steady_mask] = delta_c[steady_mask, None] * a0_vector
fest[steady_mask] = fzero[steady_mask] + fpert[steady_mask]


rmse_f0 = rmse_masked(fmill, fzero, steady_mask)
rmse_fest = rmse_masked(fmill, fest, steady_mask)
rmse_delta = rmse_masked(fmill - fzero, fpert, steady_mask)

# Revolution-binned averages using tool orientation
fmill_ss = np.where(steady_mask[:, None], fmill, np.nan)
fzero_ss = np.where(steady_mask[:, None], fzero, np.nan)
fest_ss = np.where(steady_mask[:, None], fest, np.nan)
x_ss = np.where(steady_mask[:, None], x_hist, np.nan)

fmill_rev, rev_idx = compute_revolution_average_from_angle(tool_orientation, fmill_ss)
fzero_rev, _ = compute_revolution_average_from_angle(tool_orientation, fzero_ss)
fest_rev, _ = compute_revolution_average_from_angle(tool_orientation, fest_ss)
x_rev, _ = compute_revolution_average_from_angle(tool_orientation, x_ss)

rmse_f0_rev = rmse_masked(fmill_rev, fzero_rev, steady_mask)
rmse_fest_rev = rmse_masked(fmill_rev, fest_rev, steady_mask)
rmse_delta_rev = rmse_masked(fmill_rev - fzero_rev, fest_rev - fzero_rev, steady_mask)

print("RMSE raw vs F0:", rmse_f0)
print("RMSE raw vs F_est:", rmse_fest)
print("RMSE delta vs projected feed perturbation:", rmse_delta)
print("RMSE rev-avg vs F0:", rmse_f0_rev)
print("RMSE rev-avg vs F_est:", rmse_fest_rev)
print("RMSE rev-avg delta:", rmse_delta_rev)


if SHOW_PLOT:
    labels = ["Fx", "Fy"]
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    for i in range(2):
        ax[i].plot(t, fmill_rev[:, i], label="measured avg/rev", linewidth=1.8)
        ax[i].plot(t, fzero_rev[:, i], "--", label="zero-order avg/rev", linewidth=1.5)
        ax[i].plot(t, fest_rev[:, i], "-.", label="estimate avg/rev", linewidth=1.5)
        ax[i].set_ylabel(f"{labels[i]} [N]")
        ax[i].grid(alpha=0.3)

    ax[-1].set_xlabel("time [s]")
    ax[0].legend(loc="upper left")
    fig.suptitle("Revolution-binned force comparison")
    fig.tight_layout()

    fig_x, ax_x = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
    for i in range(2):
        ax_x[i].plot(t, x_hist[:, i] * 1e3, color="0.75", linewidth=0.8, label="oscillation raw")
        ax_x[i].plot(t, x_rev[:, i] * 1e3, color="0.15", linewidth=1.4, label="oscillation avg/rev")
        ax_x[i].set_ylabel(f"x_{'xy'[i]} [mm]")
        ax_x[i].grid(alpha=0.3)
    ax_x[-1].set_xlabel("time [s]")
    ax_x[0].legend(loc="upper left")
    fig_x.suptitle("Robot oscillation (raw and revolution-binned)")
    fig_x.tight_layout()
    plt.show()
