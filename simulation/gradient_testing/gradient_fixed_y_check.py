import math as m
import numpy as np
import matplotlib.pyplot as plt

from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *


if __name__ == "__main__":
    # ---------------------------
    # Workpiece geometry
    # ---------------------------
    wp_p1 = np.array([0.0, 0.0])
    wp_p2 = np.array([1000.0, 0.0])
    wp_p3 = np.array([1000.0, 76.0])
    wp_p4 = np.array([0.0, 76.0])

    xy_workpiece = np.array([wp_p1, wp_p2, wp_p3, wp_p4]).T  # (2,4)

    # ---------------------------
    # Tooling parameters
    # ---------------------------
    tool_diameter = 20.0  # [mm]
    tool_radius = tool_diameter / 2.0
    radial_engagement = 8.0  # [mm] <-- KEEP THIS FIXED
    axial_cutting_depth = 1.0  # [mm]
    N_tool_teeth = 1

    # ---------------------------
    # Cutting coefficients
    # ---------------------------
    Ktc = 1930.4
    Krc = 1159.6

    # ---------------------------
    # Baseline tool center for this ae (fixed y)
    # ---------------------------
    offset_tool_y = tool_radius - radial_engagement
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(max(tool_radius**2 - offset_tool_y**2, 0.0))
    else:
        offset_tool_x = -tool_radius

    xy_tool_center_0 = np.array([
        wp_p4[0] + offset_tool_x,
        wp_p4[1] + offset_tool_y
    ], dtype=float)

    # ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000.0
    feed_per_tooth = 0.15  # [mm/tooth]
    delta_deg = 1.0
    rot_angle_per_time_step = delta_deg * (2 * m.pi / 360.0)
    T_end = 0.15 # [s]

    omega = -rev_per_min * 2 * m.pi / 60.0  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * N_tool_teeth  # mm/s

    # ---------------------------
    # Evaluation: sweep in dx (NOT dy)
    # ---------------------------
    dx_limits = [-2.0, 2.0]   # [mm] sweep around baseline x at fixed time T_end
    delta_x = 0.01            # [mm]
    x_iter = np.arange(dx_limits[0], dx_limits[1] + 1e-12, delta_x)
    x_len = len(x_iter)

    Fx_vals_avg = np.zeros(x_len)
    Fy_vals_avg = np.zeros(x_len)

    Fx0 = None
    Fy0 = None
    dFx_dx0 = None
    dFy_dx0 = None

    # Fixed dy so radial engagement stays fixed
    dy = 0.0

    # Baseline feed shift at T_end
    x_base = feed_speed * T_end

    for i in range(x_len):
        dx = x_iter[i]

        # Tool center: move in x only
        r_xy_current = xy_tool_center_0 + np.array([x_base + dx, dy])

        # Geometry terms (b depends on y; v depends on x)
        b = -r_xy_current[1] + tool_radius + xy_workpiece[1, -1]
        v = -(r_xy_current[0] - xy_workpiece[0, -1])

        phi_st, phi_ex = compute_entry_exit_angles_downmilling(
            r_xy_current=r_xy_current,
            xy_workpiece=xy_workpiece,
            radius_tool=tool_radius,
            diameter_end_mill=tool_diameter
        )

        res = zero_order_force_analytical(
            feed_per_tooth,
            axial_cutting_depth,
            N_tool_teeth,
            phi_st,
            phi_ex,
            Ktc,
            Krc
        )
        res = np.squeeze(res)
        Fx_vals_avg[i] = res[0]
        Fy_vals_avg[i] = res[1]

        # Grab linearization data at dx=0
        if abs(dx) < 1e-12:
            dF = zero_order_dF(
                feed_per_tooth,
                axial_cutting_depth,
                b,
                v,
                tool_diameter,
                N_tool_teeth,
                phi_st,
                phi_ex,
                Ktc,
                Krc
            )

            Fx0 = Fx_vals_avg[i]
            Fy0 = Fy_vals_avg[i]
            dFx_dx0 = dF[0, 0]  # <-- derivative w.r.t x
            dFy_dx0 = dF[1, 0]  # <-- derivative w.r.t x

    # Fallback if dx=0 not hit exactly
    if Fx0 is None:
        i0 = int(np.argmin(np.abs(x_iter)))
        dx0 = x_iter[i0]

        r_xy_0 = xy_tool_center_0 + np.array([x_base + dx0, dy])
        b0 = -r_xy_0[1] + tool_radius + xy_workpiece[1, -1]
        v0 = -(r_xy_0[0] - xy_workpiece[0, -1])

        phi_st0, phi_ex0 = compute_entry_exit_angles_downmilling(
            r_xy_current=r_xy_0,
            xy_workpiece=xy_workpiece,
            radius_tool=tool_radius,
            diameter_end_mill=tool_diameter
        )

        F0 = np.squeeze(zero_order_force_analytical(
            feed_per_tooth, axial_cutting_depth, N_tool_teeth,
            phi_st0, phi_ex0, Ktc, Krc
        ))
        Fx0, Fy0 = F0[0], F0[1]

        dF0 = zero_order_dF(
            feed_per_tooth, axial_cutting_depth, b0, v0,
            tool_diameter, N_tool_teeth, phi_st0, phi_ex0, Ktc, Krc
        )
        dFx_dx0 = dF0[0, 0]
        dFy_dx0 = dF0[1, 0]

    # First-order Taylor in dx
    Fx_lin = Fx0 + dFx_dx0 * x_iter
    Fy_lin = Fy0 + dFy_dx0 * x_iter

    # ---------------------------
    # Plots: Force vs dx and linearization
    # ---------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(x_iter, Fx_vals_avg, marker="+", linestyle="None", label=f"Fx (avg) at T={T_end}")
    axs[0].plot(x_iter, Fx_lin, linestyle="--", linewidth=2, label="Fx linearization at dx=0")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Milling Forces vs x-offset (dy fixed to keep radial engagement constant)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(x_iter, Fy_vals_avg, marker="+", linestyle="None", label=f"Fy (avg) at T={T_end}")
    axs[1].plot(x_iter, Fy_lin, linestyle="--", linewidth=2, label="Fy linearization at dx=0")
    axs[1].set_xlabel("Step dx [mm]")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Error vs dx
    fig1, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(x_iter, Fx_vals_avg - Fx_lin, marker="+", linestyle="None", label=f"Fx - Fx_lin at T={T_end}")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Linearization error vs dx")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(x_iter, Fy_vals_avg - Fy_lin, marker="+", linestyle="None", label=f"Fy - Fy_lin at T={T_end}")
    axs[1].set_xlabel("Step dx [mm]")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ============================================================
    # Central difference derivative at dx=0 using different step sizes H
    # ============================================================
    H_list = np.unique(np.concatenate([
        np.geomspace(2.0, 0.01, 30),
        np.array([2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01])
    ]))
    H_list = np.sort(H_list)[::-1]

    # Interpolate forces at +/-H in x
    Fx_plus  = np.interp(+H_list, x_iter, Fx_vals_avg)
    Fx_minus = np.interp(-H_list, x_iter, Fx_vals_avg)
    Fy_plus  = np.interp(+H_list, x_iter, Fy_vals_avg)
    Fy_minus = np.interp(-H_list, x_iter, Fy_vals_avg)

    dFx_num_H = (Fx_plus - Fx_minus) / (2.0 * H_list)
    dFy_num_H = (Fy_plus - Fy_minus) / (2.0 * H_list)

    err_dFx = dFx_num_H - dFx_dx0
    err_dFy = dFy_num_H - dFy_dx0

    abs_err_dFx = np.abs(err_dFx)
    abs_err_dFy = np.abs(err_dFy)
    abs_err_vec = np.sqrt(err_dFx**2 + err_dFy**2)

    # 1) Estimated derivative vs step size H
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(H_list, dFx_num_H, marker="o", linestyle="None", label="Central diff estimate")
    axs[0].axhline(dFx_dx0, linestyle="--", linewidth=2, label="Analytical dFx/dx at dx=0", c="r")
    axs[0].set_ylabel("dFx/dx [N/mm]")
    axs[0].set_title("Derivative at x0 using different central-difference step sizes H")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(H_list, dFy_num_H, marker="o", linestyle="None", label="Central diff estimate")
    axs[1].axhline(dFy_dx0, linestyle="--", linewidth=2, label="Analytical dFy/dx at dx=0", c="r")
    axs[1].set_xlabel("Step size H [mm]")
    axs[1].set_ylabel("dFy/dx [N/mm]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    plt.tight_layout()
    plt.show()

    # 2) Absolute error vs step size H
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(H_list, abs_err_dFx, marker="o", linestyle="None", label="|Δ dFx/dx|")
    ax.plot(H_list, abs_err_dFy, marker="s", linestyle="None", label="|Δ dFy/dx|")
    ax.plot(H_list, abs_err_vec, marker="^", linestyle="None", label="||Δ dF/dx||")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Step size H [mm]")
    ax.set_ylabel("Absolute error [N/mm]")
    ax.set_title("Absolute error vs step size H (central diff at x0)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()
