
import math as m
import numpy as np
import time
import matplotlib.pyplot as plt

# Import necessary funcs / classes
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *


if __name__ == "__main__":
    # ---------------------------
    # Workpiece geometry
    
    # 4 ------ 3
    # |        |
    # |        |
    # 1 ------ 2
    
    # ---------------------------
    
    wp_p1 = np.array([0,0])
    wp_p2 = np.array([1000,0])
    wp_p3 = np.array([1000,76])
    wp_p4 = np.array([0,76])
    
    xy_workpiece = np.array([
        wp_p1,
        wp_p2,
        wp_p3,
        wp_p4
    ]).T
    
    # ---------------------------
    # Tooling parameters
    # ---------------------------
    
    tool_diameter = 20.0  # [mm]
    tool_radius = tool_diameter / 2.0
    radial_engagement = 8.0# [mm]
    axial_cutting_depth = 1.0  # [mm]
    N_tool_teeth = 1
    
    # ---------------------------
    # Cutting coefficients
    # ---------------------------
    
    Ktc = 1930.4
    Krc = 1159.6
    # Kac = 200.6
    
    
    offset_tool_y = tool_radius - radial_engagement
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    else:
        offset_tool_x = -tool_radius
    
    xy_tool_center_0 = np.array([wp_p4[0] + offset_tool_x,
                                 wp_p4[1] + offset_tool_y])  # Start above and left of workpiece corner

    #   ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000
    feed_per_tooth = 0.15 # 0.15 [mm/tooth]
    delta_deg = 1.0 # [deg]
    rot_angle_per_time_step = delta_deg * (2*m.pi / 360)
    T_end =  0.1 # [s]
    
    # # ---------------------------
    # # Process speeds
    # # ---------------------------
    omega = -rev_per_min * 2 * m.pi / 60  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * N_tool_teeth # mm/s
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60
    
    #---------------------------
    # Evaluation Parameters
    #---------------------------

# ---- y-offset sweep around baseline ----
h_limits = [-2.0, 2.0]     # [mm]
delta_h = 0.01             # [mm]
h_iter = np.arange(h_limits[0], h_limits[1] + 1e-12, delta_h)
h_len = len(h_iter)

# ---- step-size list for the derivative test at y0 ----
H_list = np.unique(np.concatenate([
    np.geomspace(2.0, 0.01, 30),
    np.array([2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01])
]))
H_list = np.sort(H_list)[::-1]  # 2.0 -> 0.01

# ---- radial engagement sweep ----
ae_list = np.arange(1.0, 10.0 + 1e-12, 1.0)  # [mm]

# store abs error curves: shape (n_ae, n_H)
abs_err_dFx_all = np.zeros((len(ae_list), len(H_list)))
abs_err_dFy_all = np.zeros((len(ae_list), len(H_list)))

# store analytical gradients at y0 for each ae (for relative error normalization)
dFx_dy0_all = np.zeros(len(ae_list))
dFy_dy0_all = np.zeros(len(ae_list))

# store rel error curves: shape (n_ae, n_H)
rel_err_dFx_all = np.zeros((len(ae_list), len(H_list)))
rel_err_dFy_all = np.zeros((len(ae_list), len(H_list)))


for k, ae in enumerate(ae_list):

    # ---------------------------
    # Build baseline tool center for this radial engagement ae
    # same geometry logic you used:
    # offset_tool_y = R - ae
    # offset_tool_x = -sqrt(R^2 - offset_tool_y^2) if offset_tool_y>0 else -R
    # ---------------------------
    offset_tool_y = tool_radius - ae
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    else:
        offset_tool_x = -tool_radius

    xy_tool_center_0 = np.array([
        wp_p4[0] + offset_tool_x,
        wp_p4[1] + offset_tool_y
    ])

    # ---------------------------
    # Sweep h around dy=0 at fixed x(T_end)
    # ---------------------------
    Fx_vals_avg = np.zeros(h_len)
    Fy_vals_avg = np.zeros(h_len)

    Fx0 = None
    Fy0 = None
    dFx_dy0 = None
    dFy_dy0 = None

    for i in range(h_len):
        dx = 0.0
        dy = h_iter[i]

        r_xy_current = xy_tool_center_0 + np.array([feed_speed * T_end + dx, dy])

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

        # analytical gradient at nominal point h=0
        if abs(h_iter[i]) < 1e-12:
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
            dFx_dy0 = dF[0, 1]
            dFy_dy0 = dF[1, 1]

    # fallback in case floating grid misses exact 0
    if dFx_dy0 is None:
        i0 = int(np.argmin(np.abs(h_iter)))
        dy0 = h_iter[i0]
        r_xy_0 = xy_tool_center_0 + np.array([feed_speed * T_end, dy0])

        b0 = -r_xy_0[1] + tool_radius + xy_workpiece[1, -1]
        v0 = -(r_xy_0[0] - xy_workpiece[0, -1])

        phi_st0, phi_ex0 = compute_entry_exit_angles_downmilling(
            r_xy_current=r_xy_0,
            xy_workpiece=xy_workpiece,
            radius_tool=tool_radius,
            diameter_end_mill=tool_diameter
        )

        dF0 = zero_order_dF(
            feed_per_tooth,
            axial_cutting_depth,
            b0,
            v0,
            tool_diameter,
            N_tool_teeth,
            phi_st0,
            phi_ex0,
            Ktc,
            Krc
        )

        Fx0 = Fx_vals_avg[i0]
        Fy0 = Fy_vals_avg[i0]
        dFx_dy0 = dF0[0, 1]
        dFy_dy0 = dF0[1, 1]

    # ---------------------------
    # Central-difference-at-y0 with step size H
    # dF/dy|0 ≈ (F(+H)-F(-H)) / (2H)
    # ---------------------------
    Fx_plus  = np.interp(+H_list, h_iter, Fx_vals_avg)
    Fx_minus = np.interp(-H_list, h_iter, Fx_vals_avg)
    Fy_plus  = np.interp(+H_list, h_iter, Fy_vals_avg)
    Fy_minus = np.interp(-H_list, h_iter, Fy_vals_avg)

    dFx_num_H = (Fx_plus - Fx_minus) / (2.0 * H_list)
    dFy_num_H = (Fy_plus - Fy_minus) / (2.0 * H_list)
    
    # keep analytical gradients for this ae
    dFx_dy0_all[k] = dFx_dy0
    dFy_dy0_all[k] = dFy_dy0

    # relative error (guard against tiny analytical gradients)
    eps_rel = 1e-9  # N/mm (tune if needed)
    rel_err_dFx_all[k, :] = np.abs(dFx_num_H - dFx_dy0) / np.maximum(np.abs(dFx_dy0), eps_rel)
    rel_err_dFy_all[k, :] = np.abs(dFy_num_H - dFy_dy0) / np.maximum(np.abs(dFy_dy0), eps_rel)


    abs_err_dFx_all[k, :] = np.abs(dFx_num_H - dFx_dy0)
    abs_err_dFy_all[k, :] = np.abs(dFy_num_H - dFy_dy0)

    print(f"ae={ae:.1f} mm done. dFx_dy0={dFx_dy0:.3g}, dFy_dy0={dFy_dy0:.3g}")

# ============================================================
# Final plot: abs error vs step size H for all ae
# two subplots: Fx and Fy
# ============================================================

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for k, ae in enumerate(ae_list):
    axs[0].plot(H_list, abs_err_dFx_all[k, :], marker="o", linestyle="None", label=f"ae={ae:.0f}mm")
    axs[1].plot(H_list, abs_err_dFy_all[k, :], marker="o", linestyle="None", label=f"ae={ae:.0f}mm")

axs[0].set_xscale("log")
axs[0].set_yscale("log")
axs[1].set_xscale("log")
axs[1].set_yscale("log")

axs[0].set_ylabel("|error in dFx/dy| [N/mm]")
axs[0].set_title("Central-difference step-size study at y0: absolute error vs H / Varying radial engagement")

axs[1].set_xlabel("Step size H [mm]")
axs[1].set_ylabel("|error in dFy/dy| [N/mm]")

axs[0].grid(True, alpha=0.3, which="both")
axs[1].grid(True, alpha=0.3, which="both")

# legend can get big; put it outside
axs[0].legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
axs[1].legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

plt.tight_layout()
plt.show()

# ============================================================
# Final plot: relative error vs step size H for all ae
# ============================================================

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for k, ae in enumerate(ae_list):
    axs[0].plot(H_list, rel_err_dFx_all[k, :], marker="o", linestyle="None", label=f"ae={ae:.0f}mm")
    axs[1].plot(H_list, rel_err_dFy_all[k, :], marker="o", linestyle="None", label=f"ae={ae:.0f}mm")

axs[0].set_xscale("log")
axs[1].set_xscale("log")

# y scale: log is often useful; if you get zeros, use symlog or add epsilon
axs[0].set_yscale("log")
axs[1].set_yscale("log")

axs[0].set_ylabel("Relative error in dFx/dy [-]")
axs[0].set_title("Central-difference step-size study at y0: relative error vs H / Varying radial engagement")

axs[1].set_xlabel("Step size H [mm]")
axs[1].set_ylabel("Relative error in dFy/dy [-]")

axs[0].grid(True, alpha=0.3, which="both")
axs[1].grid(True, alpha=0.3, which="both")

axs[0].legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))
axs[1].legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.02, 1.0))

plt.tight_layout()
plt.show()
