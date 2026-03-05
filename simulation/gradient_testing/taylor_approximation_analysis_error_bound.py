import math as m
import numpy as np
import matplotlib.pyplot as plt

from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *


def make_time_vector(rev_per_min: float, delta_deg: float, T_end: float) -> tuple[np.ndarray, float]:
    """Return time vector t and dt consistent with angular step delta_deg at spindle speed rev_per_min."""
    rot_step = delta_deg * (2 * m.pi / 360.0)  # rad per sample
    steps_per_rev = (2 * m.pi) / rot_step      # samples per revolution
    sample_freq = steps_per_rev * (rev_per_min / 60.0)  # samples per second
    dt = 1.0 / sample_freq

    # Robust t vector: exactly N samples up to <= T_end
    N = int(np.floor(T_end / dt)) + 1
    t = np.arange(N) * dt
    return t, dt


def tool_center_from_ae(wp_p4: np.ndarray, tool_radius: float, ae0: float) -> np.ndarray:
    """Baseline tool center position at the corner wp_p4 for a given radial engagement ae0."""
    offset_tool_y = tool_radius - ae0
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(max(tool_radius**2 - offset_tool_y**2, 0.0))
    else:
        offset_tool_x = -tool_radius

    return np.array([wp_p4[0] + offset_tool_x, wp_p4[1] + offset_tool_y], dtype=float)


if __name__ == "__main__":
    # ---------------------------
    # Workpiece geometry (rectangle)
    # ---------------------------
    wp_p1 = np.array([0.0, 0.0])
    wp_p2 = np.array([1000.0, 0.0])
    wp_p3 = np.array([1000.0, 76.0])
    wp_p4 = np.array([0.0, 76.0])

    xy_workpiece = np.array([wp_p1, wp_p2, wp_p3, wp_p4]).T  # shape (2,4)

    # ---------------------------
    # Tooling parameters
    # ---------------------------
    tool_diameter = 20.0
    tool_radius = tool_diameter / 2.0
    axial_cutting_depth = 1.0
    N_tool_teeth = 1

    # ---------------------------
    # Cutting coefficients
    # ---------------------------
    Ktc = 1930.4
    Krc = 1159.6

    # ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000.0
    feed_per_tooth = 0.15
    delta_deg = 1.0
    T_end = 0.5

    omega = -rev_per_min * 2 * m.pi / 60.0  # rad/s (sign = your convention)
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * N_tool_teeth  # mm/s

    t, dt = make_time_vector(rev_per_min, delta_deg, T_end)
    N = t.size

    # ---------------------------
    # Baseline engagement and dy sweep
    # ---------------------------
    ae0 = 7.0
    xy_tool_center_0 = tool_center_from_ae(wp_p4, tool_radius, ae0)

    dy_list = np.arange(-0.5, 0.5 + 1e-12, 0.1)
    M = dy_list.size

    # Storage
    Fx_true = np.zeros((M, N))
    Fy_true = np.zeros((M, N))
    dFxdy_true = np.zeros((M, N))
    dFydy_true = np.zeros((M, N))

    Fx0_t = np.zeros(N)
    Fy0_t = np.zeros(N)
    dFxdy0_t = np.zeros(N)
    dFydy0_t = np.zeros(N)

    # Precompute x-shift over time
    x_shift = feed_speed * t

    # ----------------------------
    # Main loops
    # ----------------------------
    for it in range(N):
        # Base dy=0
        r0 = xy_tool_center_0 + np.array([x_shift[it], 0.0])

        b0 = -r0[1] + tool_radius + xy_workpiece[1, -1]
        v0 = -(r0[0] - xy_workpiece[0, -1])

        phi_st0, phi_ex0 = compute_entry_exit_angles_downmilling(
            r_xy_current=r0,
            xy_workpiece=xy_workpiece,
            radius_tool=tool_radius,
            diameter_end_mill=tool_diameter,
        )

        F0 = np.squeeze(zero_order_force_analytical(
            feed_per_tooth, axial_cutting_depth, N_tool_teeth,
            phi_st0, phi_ex0, Ktc, Krc
        ))
        Fx0_t[it], Fy0_t[it] = F0[0], F0[1]

        dF0 = zero_order_dF(
            feed_per_tooth, axial_cutting_depth,
            b0, v0, tool_diameter, N_tool_teeth,
            phi_st0, phi_ex0, Ktc, Krc
        )
        dFxdy0_t[it] = dF0[0, 1]
        dFydy0_t[it] = dF0[1, 1]

        # Sweep dy
        for k, dy in enumerate(dy_list):
            r = xy_tool_center_0 + np.array([x_shift[it], dy])

            b = -r[1] + tool_radius + xy_workpiece[1, -1]
            v = -(r[0] - xy_workpiece[0, -1])

            phi_st, phi_ex = compute_entry_exit_angles_downmilling(
                r_xy_current=r,
                xy_workpiece=xy_workpiece,
                radius_tool=tool_radius,
                diameter_end_mill=tool_diameter,
            )

            Favg = np.squeeze(zero_order_force_analytical(
                feed_per_tooth, axial_cutting_depth, N_tool_teeth,
                phi_st, phi_ex, Ktc, Krc
            ))
            Fx_true[k, it], Fy_true[k, it] = Favg[0], Favg[1]

            dF = zero_order_dF(
                feed_per_tooth, axial_cutting_depth,
                b, v, tool_diameter, N_tool_teeth,
                phi_st, phi_ex, Ktc, Krc
            )
            dFxdy_true[k, it] = dF[0, 1]
            dFydy_true[k, it] = dF[1, 1]

    # ----------------------------
    # First-order Taylor about dy=0
    # ----------------------------
    Fx_tay = Fx0_t[None, :] + dFxdy0_t[None, :] * dy_list[:, None]
    Fy_tay = Fy0_t[None, :] + dFydy0_t[None, :] * dy_list[:, None]

    # Errors
    Fx_err = Fx_true - Fx_tay
    Fy_err = Fy_true - Fy_tay

    # ----------------------------
    # Relative error (FIXED normalization)
    # ----------------------------
    eps = 1e-6
    Fx_rel_err = np.abs(Fx_err) / (np.abs(Fx_true) + eps)
    Fy_rel_err = np.abs(Fy_err) / (np.abs(Fy_true) + eps)

    # (Alternative: normalize by vector magnitude)
    # Fmag_true = np.sqrt(Fx_true**2 + Fy_true**2)
    # Fx_rel_err = np.abs(Fx_err) / (Fmag_true + eps)
    # Fy_rel_err = np.abs(Fy_err) / (Fmag_true + eps)

    # ----------------------------
    # SMAPE (fine as-is)
    # ----------------------------
    Fx_smape = 2.0 * np.abs(Fx_err) / (np.abs(Fx_true) + np.abs(Fx_tay) + eps)
    Fy_smape = 2.0 * np.abs(Fy_err) / (np.abs(Fy_true) + np.abs(Fy_tay) + eps)
    Fx_smape_pct = 100.0 * Fx_smape
    Fy_smape_pct = 100.0 * Fy_smape

    # ----------------------------
    # Domain-normalized remainder |R|/dy^2  (safe mask usage)
    # ----------------------------
    dy_abs = np.abs(dy_list)
    mask = dy_abs > 0.0

    Ex = np.full((M, N), np.nan)
    Ey = np.full((M, N), np.nan)

    Ex[mask, :] = np.abs(Fx_err[mask, :]) / (dy_abs[mask, None] ** 2)
    Ey[mask, :] = np.abs(Fy_err[mask, :]) / (dy_abs[mask, None] ** 2)

    # ----------------------------
    # Optional sanity check: analytic dF/dy at 0 vs finite-difference slope
    # ----------------------------
    # Pick h from your dy grid if available
    h = 0.1
    if np.any(np.isclose(dy_list, -h)) and np.any(np.isclose(dy_list, +h)):
        k_m = int(np.where(np.isclose(dy_list, -h))[0][0])
        k_p = int(np.where(np.isclose(dy_list, +h))[0][0])
        dFxdy_fd = (Fx_true[k_p, :] - Fx_true[k_m, :]) / (2.0 * h)
        dFydy_fd = (Fy_true[k_p, :] - Fy_true[k_m, :]) / (2.0 * h)

        plt.figure(figsize=(10, 4))
        plt.plot(t, dFxdy0_t, label="dFx/dy analytic @0")
        plt.plot(t, dFxdy_fd, "--", label="dFx/dy FD (h=0.1)")
        plt.grid(True, alpha=0.3)
        plt.xlabel("Time [s]")
        plt.ylabel("dFx/dy [N/mm]")
        plt.title("Derivative sanity check")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # Plots (keep legends manageable)
    # ----------------------------
    show_all_labels = False  # flip to True if you really want all legend entries

    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for k, dy in enumerate(dy_list):
        lab_true = f"dy={dy:+.1f} true" if show_all_labels else None
        lab_tay  = f"dy={dy:+.1f} Tay" if show_all_labels else None

        axs[0].plot(t, Fx_true[k, :], linewidth=1, label=lab_true)
        axs[0].plot(t, Fx_tay[k, :], linewidth=1, linestyle="--", alpha=0.9, label=lab_tay)

        axs[1].plot(t, Fy_true[k, :], linewidth=1, label=lab_true)
        axs[1].plot(t, Fy_tay[k, :], linewidth=1, linestyle="--", alpha=0.9, label=lab_tay)

    axs[0].set_ylabel("Fx [N]")
    axs[0].set_title(f"Taylor about dy=0 (ae={ae0:.1f} mm): Fx(t)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Fy [N]")
    axs[1].set_title(f"Taylor about dy=0 (ae={ae0:.1f} mm): Fy(t)")
    for ax in axs:
        ax.grid(True, alpha=0.3)

    if show_all_labels:
        axs[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        axs[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    # Error plots (absolute, relative, SMAPE, remainder)
    # Absolute error
    figE, axsE = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, dy in enumerate(dy_list):
        axsE[0].plot(t, Fx_err[k, :], linewidth=1)
        axsE[1].plot(t, Fy_err[k, :], linewidth=1)
    axsE[0].set_ylabel("Fx_true - Fx_Tay [N]")
    axsE[0].set_title("Taylor absolute error vs time (Fx)")
    axsE[1].set_xlabel("Time [s]")
    axsE[1].set_ylabel("Fy_true - Fy_Tay [N]")
    axsE[1].set_title("Taylor absolute error vs time (Fy)")
    axsE[0].grid(True, alpha=0.3)
    axsE[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Relative error (fixed)
    figR, axsR = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, dy in enumerate(dy_list):
        axsR[0].plot(t, Fx_rel_err[k, :], linewidth=1)
        axsR[1].plot(t, Fy_rel_err[k, :], linewidth=1)
    axsR[0].set_ylabel("Rel err Fx [-]")
    axsR[0].set_title("Taylor relative error vs time (Fx)")
    axsR[1].set_xlabel("Time [s]")
    axsR[1].set_ylabel("Rel err Fy [-]")
    axsR[1].set_title("Taylor relative error vs time (Fy)")
    axsR[0].grid(True, alpha=0.3)
    axsR[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # SMAPE
    figS, axsS = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, dy in enumerate(dy_list):
        axsS[0].plot(t, Fx_smape_pct[k, :], linewidth=1)
        axsS[1].plot(t, Fy_smape_pct[k, :], linewidth=1)
    axsS[0].set_ylabel("SMAPE Fx [%]")
    axsS[0].set_title("Taylor SMAPE vs time (Fx)")
    axsS[1].set_xlabel("Time [s]")
    axsS[1].set_ylabel("SMAPE Fy [%]")
    axsS[1].set_title("Taylor SMAPE vs time (Fy)")
    axsS[0].grid(True, alpha=0.3)
    axsS[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Mean SMAPE vs dy
    Fx_smape_mean = Fx_smape_pct.mean(axis=1)
    Fy_smape_mean = Fy_smape_pct.mean(axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(dy_list, Fx_smape_mean, marker="o", label="Mean SMAPE Fx [%]")
    plt.plot(dy_list, Fy_smape_mean, marker="s", label="Mean SMAPE Fy [%]")
    plt.xlabel("dy [mm]")
    plt.ylabel("Mean SMAPE over time [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Remainder normalized by dy^2
    figQ, axsQ = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, dy in enumerate(dy_list):
        if np.isclose(dy, 0.0):
            continue
        axsQ[0].plot(t, Ex[k, :], linewidth=1)
        axsQ[1].plot(t, Ey[k, :], linewidth=1)
    axsQ[0].set_ylabel(r"$|R_x|/dy^2$ [N/mm$^2$]")
    axsQ[0].set_title("Domain-normalized remainder vs time (Fx)")
    axsQ[1].set_xlabel("Time [s]")
    axsQ[1].set_ylabel(r"$|R_y|/dy^2$ [N/mm$^2$]")
    axsQ[1].set_title("Domain-normalized remainder vs time (Fy)")
    axsQ[0].grid(True, alpha=0.3)
    axsQ[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    Ex_mean = np.nanmean(Ex, axis=1)
    Ey_mean = np.nanmean(Ey, axis=1)
    plt.figure(figsize=(8, 4))
    plt.plot(dy_list, Ex_mean, marker="o", label=r"Mean $|R_x|/dy^2$")
    plt.plot(dy_list, Ey_mean, marker="s", label=r"Mean $|R_y|/dy^2$")
    plt.xlabel("dy [mm]")
    plt.ylabel(r"Mean normalized remainder [N/mm$^2$]")
    plt.title("Mean second-order Taylor remainder (domain-normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
