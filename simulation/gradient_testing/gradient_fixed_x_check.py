
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
    
    h_limits = [-2.0, 2.0]     # [mm] sweep around baseline tool center y
    delta_h = 0.01             # [mm]
    h_iter = np.arange(h_limits[0], h_limits[1] + 1e-12, delta_h)
    h_len = len(h_iter)

    Fx_vals_avg = np.zeros(h_len)
    Fy_vals_avg = np.zeros(h_len)
    
    Fx0 = None
    Fy0 = None
    dFx_dy0 = None
    dFy_dy0 = None

    for i in range(h_len):
        dx = 0.0
        dy = h_iter[i]  # <-- offset only (NOT absolute y)

        r_xy_current = xy_tool_center_0 + np.array([feed_speed * T_end + dx, dy])
    
        
        # Current tool center
        # r_xy_current = xy_tool_center_0 + np.array([feed_speed * T_end,  0.0]) + np.array([dx , dy])
        
        b = - r_xy_current[1] + tool_radius + xy_workpiece[1,-1]  # radial depth of cut 
        v = -(r_xy_current[0] - xy_workpiece[0,-1]) 
        
        phi_st, phi_ex = compute_entry_exit_angles_downmilling( r_xy_current=r_xy_current,
                                                                xy_workpiece=xy_workpiece,
                                                                radius_tool=tool_radius,
                                                                diameter_end_mill= tool_diameter)
            
        # ---------------------------
        # ANALYTICAL : Milling final force at Steady State (Fixed X(T))
        # ---------------------------

        
        res = zero_order_force_analytical(
                feed_per_tooth,
                axial_cutting_depth,
                N_tool_teeth,
                phi_st,
                phi_ex,
                Ktc,
                Krc
            )

        
        Fx_vals_avg[i]= res[0]
        Fy_vals_avg[i]= res[1]
        
        
        if abs(h_iter[i]) < 1e-12:  # nominal point h=0
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
    
    
    if Fx0 is None:
    # fallback: use the closest index to 0 if floating steps miss it
        i0 = int(np.argmin(np.abs(h_iter)))
        Fx0 = Fx_vals_avg[i0]
        Fy0 = Fy_vals_avg[i0]
        # You need dF at i0 too; easiest is to recompute it once at that point:
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
            feed_per_tooth, axial_cutting_depth, b0, v0, tool_diameter, N_tool_teeth,
            phi_st0, phi_ex0, Ktc, Krc
        )
        dFx_dy0 = dF0[0, 1]
        dFy_dy0 = dF0[1, 1]

    Fx_lin = Fx0 + dFx_dy0 * h_iter
    Fy_lin = Fy0 + dFy_dy0 * h_iter


    # ---------------------------
    # Analytical expressions
    # ---------------------------
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(h_iter, Fx_vals_avg, marker="+", linestyle="None", label=f"Fx (analytical avg) at T={T_end}")
    axs[0].plot(h_iter, Fx_lin, linestyle="--", linewidth=2, label="Fx linearization at h=0")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Milling Forces vs y-offset with linearization about (dx=0, dy=0)")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(h_iter, Fy_vals_avg, marker="+", linestyle="None", label=f"Fy (analytical avg) at T={T_end}")
    axs[1].plot(h_iter, Fy_lin, linestyle="--", linewidth=2, label="Fy linearization at h=0")
    axs[1].set_xlabel("Step h [mm]  (h)")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    fig1, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axs[0].plot(h_iter, Fx_vals_avg - Fx_lin, marker="+", linestyle="None", label=f"Fx minus Fx_lin T={T_end}")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Linearization error ")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(h_iter, Fy_vals_avg - Fy_lin, marker="+", linestyle="None", label=f"Fy minus Fy_lin at T={T_end}")
    axs[1].set_xlabel("Step h [mm]  (h)")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    
    

    
    # ============================
    # Central finite differences on sampled F(h)
    # ============================
    dh = delta_h  # uniform spacing

    # central difference derivatives
    
    Fx_cdiff = (Fx_vals_avg[2:] - Fx_vals_avg[:-2]) / (h_iter[2:] - h_iter[:-2])
    Fy_cdiff = (Fy_vals_avg[2:] - Fy_vals_avg[:-2]) / (h_iter[2:] - h_iter[:-2])
    h_cdiff = h_iter[1:-1] 

    # Compare finite diff derivative to analytical gradient at h=0 (constant)
    dFx_err_vs0 = Fx_cdiff - dFx_dy0
    dFy_err_vs0 = Fy_cdiff - dFy_dy0
    
    # # 1) Derivative vs h (finite difference) + analytical dF/dy at 0
    # fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # axs[0].plot(h_cdiff, Fx_cdiff, marker=".", linestyle="None", label="dFx/dh (central diff)")
    # axs[0].axhline(dFx_dy0, linestyle="--", linewidth=2, label="Analytical dFx/dy at h=0", c="r")
    # axs[0].set_ylabel("dFx/dh [N/mm]")
    # axs[0].set_title("Derivative check: central difference vs analytical gradient at h=0")
    # axs[0].legend()
    # axs[0].grid(True, alpha=0.3)

    # axs[1].plot(h_cdiff, Fy_cdiff, marker=".", linestyle="None", label="dFy/dh (central diff)")
    # axs[1].axhline(dFy_dy0, linestyle="--", linewidth=2, label="Analytical dFy/dy at h=0" , c="r")
    # axs[1].set_xlabel("Step h [mm]")
    # axs[1].set_ylabel("dFy/dh [N/mm]")
    # axs[1].legend()
    # axs[1].grid(True, alpha=0.3)

    # plt.tight_layout()
    # plt.show()
    
    # ============================================================
# Central difference derivative at h=0 using different step sizes H
# ============================================================

# Build a step-size list from 2.0 down to 0.01.
# Using log spacing gives you coverage across scales while keeping the plot readable.
    H_list = np.unique(np.concatenate([
        np.geomspace(2.0, 0.01, 30),     # 30 log-spaced points incl endpoints
        np.array([2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01])  # ensure "nice" values
    ]))
    H_list = np.sort(H_list)[::-1]  # descending from 2.0 -> 0.01

    # Interpolate forces at +/-H from your sampled F(h) curves
    Fx_plus  = np.interp(+H_list, h_iter, Fx_vals_avg)
    Fx_minus = np.interp(-H_list, h_iter, Fx_vals_avg)
    Fy_plus  = np.interp(+H_list, h_iter, Fy_vals_avg)
    Fy_minus = np.interp(-H_list, h_iter, Fy_vals_avg)

    # Central-difference derivative estimates at h=0 for each H
    dFx_num_H = (Fx_plus - Fx_minus) / (2.0 * H_list)
    dFy_num_H = (Fy_plus - Fy_minus) / (2.0 * H_list)

    # Errors vs analytical gradient at h=0
    err_dFx = dFx_num_H - dFx_dy0
    err_dFy = dFy_num_H - dFy_dy0

    abs_err_dFx = np.abs(err_dFx)
    abs_err_dFy = np.abs(err_dFy)
    abs_err_vec = np.sqrt(err_dFx**2 + err_dFy**2)

    # ============================================================
    # Plots
    # ============================================================

    # 1) Estimated derivative vs step size H
    fig, axs = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axs[0].plot(H_list, dFx_num_H, marker="o", linestyle="None", label="Central diff estimate")
    axs[0].axhline(dFx_dy0, linestyle="--", linewidth=2, label="Analytical dFx/dy at h=0" , c = "r")
    axs[0].set_ylabel("dFx/dy [N/mm]")
    axs[0].set_title("Derivative at y0 using different central-difference step sizes H")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(H_list, dFy_num_H, marker="o", linestyle="None", label="Central diff estimate" )
    axs[1].axhline(dFy_dy0, linestyle="--", linewidth=2, label="Analytical dFy/dy at h=0" , c = "r")
    axs[1].set_xlabel("Step size H [mm]")
    axs[1].set_ylabel("dFy/dy [N/mm]")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # (optional) log x-axis since H spans decades
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")

    plt.tight_layout()
    plt.show()

    # 2) Absolute error vs step size H
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(H_list, abs_err_dFx, marker="o", linestyle="None", label="|Δ dFx/dy|")
    ax.plot(H_list, abs_err_dFy, marker="s", linestyle="None", label="|Δ dFy/dy|")
    ax.plot(H_list, abs_err_vec, marker="^", linestyle="None", label="||Δ dF/dy||")
    ax.set_xscale("log")
    ax.set_yscale("log")  # usually helpful; remove if you prefer linear y
    ax.set_xlabel("Step size H [mm]")
    ax.set_ylabel("Absolute error [N/mm]")
    ax.set_title("Absolute error vs step size H (central diff at y0)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.tight_layout()
    plt.show()

