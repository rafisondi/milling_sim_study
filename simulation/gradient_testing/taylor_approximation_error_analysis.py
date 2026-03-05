
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
    # radial_engagement = 8.0# [mm]
    axial_cutting_depth = 1.0  # [mm]
    N_tool_teeth = 1
    
    # ---------------------------
    # Cutting coefficients
    # ---------------------------
    
    Ktc = 1930.4
    Krc = 1159.6
    # Kac = 200.6
    
    
     #   ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000
    feed_per_tooth = 0.15 # 0.15 [mm/tooth]
    delta_deg = 1.0 # [deg]
    rot_angle_per_time_step = delta_deg * (2*m.pi / 360)
    T_end =  0.5 # [s]
    
    # # ---------------------------
    # # Process speeds
    # # ---------------------------
    omega = -rev_per_min * 2 * m.pi / 60  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * N_tool_teeth # mm/s
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60
    
    ae0 = 7.0  # [mm] base radial engagement

    # --- build baseline tool center for this ae ---
    offset_tool_y = tool_radius - ae0
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    else:
        offset_tool_x = -tool_radius

    xy_tool_center_0 = np.array([
        wp_p4[0] + offset_tool_x,
        wp_p4[1] + offset_tool_y
    ])

    # --- time discretization consistent with your previous approach ---
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60.0
    dt = 1.0 / sample_freq
    t = np.arange(0.0, T_end + 1e-12, dt)
    N = len(t)

    # --- dy sweep ---
    dy_list = np.arange(-0.5, 0.5 + 1e-12, 0.1)   # [-0.5..0.5] step 0.1
    M = len(dy_list)

    # Storage: true average forces and gradients for each dy, at each time
    Fx_true = np.zeros((M, N))
    Fy_true = np.zeros((M, N))

    dFxdy_true = np.zeros((M, N))
    dFydy_true = np.zeros((M, N))

    # Also store base (dy=0) force and gradient vs time for Taylor expansion
    Fx0_t = np.zeros(N)
    Fy0_t = np.zeros(N)
    dFxdy0_t = np.zeros(N)
    dFydy0_t = np.zeros(N)

    # ----------------------------
    # Time loop + dy loop
    # ----------------------------
    for it in range(N):
        x_t = feed_speed * t[it]  # tool center x shift at time t

        # First compute base dy=0 (for Taylor)
        r0 = xy_tool_center_0 + np.array([x_t, 0.0])

        b0 = -r0[1] + tool_radius + xy_workpiece[1, -1]
        v0 = -(r0[0] - xy_workpiece[0, -1])

        phi_st0, phi_ex0 = compute_entry_exit_angles_downmilling(
            r_xy_current=r0,
            xy_workpiece=xy_workpiece,
            radius_tool=tool_radius,
            diameter_end_mill=tool_diameter
        )

        F0 = zero_order_force_analytical(
            feed_per_tooth,
            axial_cutting_depth,
            N_tool_teeth,
            phi_st0,
            phi_ex0,
            Ktc,
            Krc
        )
        F0 = np.squeeze(F0)
        Fx0_t[it] = F0[0]
        Fy0_t[it] = F0[1]

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
        dFxdy0_t[it] = dF0[0, 1]
        dFydy0_t[it] = dF0[1, 1]

        # Now compute true F and gradients for each dy_i at this time
        for k, dy in enumerate(dy_list):
            r = xy_tool_center_0 + np.array([x_t, dy])

            b = -r[1] + tool_radius + xy_workpiece[1, -1]
            v = -(r[0] - xy_workpiece[0, -1])

            phi_st, phi_ex = compute_entry_exit_angles_downmilling(
                r_xy_current=r,
                xy_workpiece=xy_workpiece,
                radius_tool=tool_radius,
                diameter_end_mill=tool_diameter
            )

            Favg = zero_order_force_analytical(
                feed_per_tooth,
                axial_cutting_depth,
                N_tool_teeth,
                phi_st,
                phi_ex,
                Ktc,
                Krc
            )
            Favg = np.squeeze(Favg)
            Fx_true[k, it] = Favg[0]
            Fy_true[k, it] = Favg[1]

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
            dFxdy_true[k, it] = dF[0, 1]
            dFydy_true[k, it] = dF[1, 1]

    # ----------------------------
    # Taylor approximation about dy=0
    # F_tay(t,dy) = F(t,0) + dF/dy(t,0) * dy
    # ----------------------------
    Fx_tay = Fx0_t[None, :] + dFxdy0_t[None, :] * dy_list[:, None]
    Fy_tay = Fy0_t[None, :] + dFydy0_t[None, :] * dy_list[:, None]

    # ----------------------------
    # Plot: total plot (Fx(t), Fy(t)) with Taylor overlay for each dy
    # ----------------------------
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Plot true forces (solid) and Taylor (dashed) for each dy
    for k, dy in enumerate(dy_list):
        axs[0].plot(t, Fx_true[k, :], linewidth=1, label=f"dy={dy:+.1f} (true)")
        axs[0].plot(t, Fx_tay[k, :], linewidth=1, linestyle="--", alpha=0.9, label=f"dy={dy:+.1f} (Taylor)")

        axs[1].plot(t, Fy_true[k, :], linewidth=1, label=f"dy={dy:+.1f} (true)")
        axs[1].plot(t, Fy_tay[k, :], linewidth=1, linestyle="--", alpha=0.9, label=f"dy={dy:+.1f} (Taylor)")

    axs[0].set_ylabel("Fx [N]")
    axs[0].set_title(f"Taylor approximation about dy=0, base ae={ae0:.1f}mm: Fx(t)")

    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Fy [N]")
    axs[1].set_title(f"Taylor approximation about dy=0, base ae={ae0:.1f}mm: Fy(t)")

    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)

    # Legends will be huge; put one legend outside (or comment out)
    axs[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axs[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # Optional: a cleaner "total error" plot (recommended)
    # ----------------------------
    Fx_err = Fx_true - Fx_tay
    Fy_err = Fy_true - Fy_tay

    figE, axsE = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for k, dy in enumerate(dy_list):
        axsE[0].plot(t, Fx_err[k, :], linewidth=1, label=f"dy={dy:+.1f}")
        axsE[1].plot(t, Fy_err[k, :], linewidth=1, label=f"dy={dy:+.1f}")

    axsE[0].set_ylabel("Fx_true - Fx_Taylor [N]")
    axsE[0].set_title("Taylor error vs time (Fx)")
    axsE[0].grid(True, alpha=0.3)

    axsE[1].set_xlabel("Time [s]")
    axsE[1].set_ylabel("Fy_true - Fy_Taylor [N]")
    axsE[1].set_title("Taylor error vs time (Fy)")
    axsE[1].grid(True, alpha=0.3)

    axsE[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axsE[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()


    # ----------------------------
    # Relative error vs time (component-wise)
    # ----------------------------
    eps = 1e-1
    Fx_rel_err = np.abs(Fx_true - Fx_tay) / np.maximum(np.abs(Fx0_t)[None, :], eps)
    Fy_rel_err = np.abs(Fy_true - Fy_tay) / np.maximum(np.abs(Fy0_t)[None, :], eps)

    # eps = 1e-5
    # Fmag_true = np.sqrt(Fx_true**2 + Fy_true**2)
    # Fmag0 = np.sqrt(Fx0_t**2 + Fy0_t**2)

    # Fx_rel_err = np.abs(Fx_true - Fx_tay) / np.maximum(Fmag0[None, :], eps)
    # Fy_rel_err = np.abs(Fy_true - Fy_tay) / np.maximum(Fmag0[None, :], eps)


    # thresh = 1e-1  # N, pick something reasonable for your forces
    # Fx_rel_err = np.full_like(Fx_true, np.nan)
    # Fy_rel_err = np.full_like(Fy_true, np.nan)

    # maskFx = np.abs(Fx_true) >= thresh
    # maskFy = np.abs(Fy_true) >= thresh

    # Fx_rel_err[maskFx] = np.abs(Fx_true[maskFx] - Fx_tay[maskFx]) / np.abs(Fx_true[maskFx])
    # Fy_rel_err[maskFy] = np.abs(Fy_true[maskFy] - Fy_tay[maskFy]) / np.abs(Fy_true[maskFy])


    figR, axsR = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for k, dy in enumerate(dy_list):
        axsR[0].plot(t, Fx_rel_err[k, :], linewidth=1, label=f"dy={dy:+.1f}")
        axsR[1].plot(t, Fy_rel_err[k, :], linewidth=1, label=f"dy={dy:+.1f}")

    axsR[0].set_ylabel("Relative error Fx [-]")
    axsR[0].set_title("Taylor relative error vs time (Fx)")
    axsR[0].grid(True, alpha=0.3)

    axsR[1].set_xlabel("Time [s]")
    axsR[1].set_ylabel("Relative error Fy [-]")
    axsR[1].set_title("Taylor relative error vs time (Fy)")
    axsR[1].grid(True, alpha=0.3)

    axsR[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axsR[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    # ----------------------------
    # SMAPE-like error vs time (component-wise)
    # ----------------------------
    eps = 1e-5  # N, just prevents 0/0 when both are exactly zero

    Fx_smape = 2.0 * np.abs(Fx_true - Fx_tay) / (np.abs(Fx_true) + np.abs(Fx_tay) + eps)
    Fy_smape = 2.0 * np.abs(Fy_true - Fy_tay) / (np.abs(Fy_true) + np.abs(Fy_tay) + eps)

    # Optional: express as percent
    Fx_smape_pct = 100.0 * Fx_smape
    Fy_smape_pct = 100.0 * Fy_smape

    figS, axsS = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for k, dy in enumerate(dy_list):
        axsS[0].plot(t, Fx_smape_pct[k, :], linewidth=1, label=f"dy={dy:+.1f}")
        axsS[1].plot(t, Fy_smape_pct[k, :], linewidth=1, label=f"dy={dy:+.1f}")

    axsS[0].set_ylabel("SMAPE Fx [%]")
    axsS[0].set_title("Taylor error vs time (SMAPE, Fx)")
    axsS[0].grid(True, alpha=0.3)

    axsS[1].set_xlabel("Time [s]")
    axsS[1].set_ylabel("SMAPE Fy [%]")
    axsS[1].set_title("Taylor error vs time (SMAPE, Fy)")
    axsS[1].grid(True, alpha=0.3)

    axsS[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axsS[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    Fx_smape_mean = Fx_smape_pct.mean(axis=1)  # mean over time, per dy
    Fy_smape_mean = Fy_smape_pct.mean(axis=1)

    plt.figure(figsize=(8,4))
    plt.plot(dy_list, Fx_smape_mean, marker="o", label="Mean SMAPE Fx [%]")
    plt.plot(dy_list, Fy_smape_mean, marker="s", label="Mean SMAPE Fy [%]")
    plt.xlabel("dy [mm]")
    plt.ylabel("Mean SMAPE over time [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # ----------------------------
    # Domain-normalized second-order error (Metric B)
    # |R| / |dy|^2
    # ----------------------------

    Fx_R = Fx_true - Fx_tay
    Fy_R = Fy_true - Fy_tay

    dy_abs = np.abs(dy_list)
    mask = dy_abs > 0.0  # exclude dy = 0

    Ex = np.full_like(Fx_R, np.nan)
    Ey = np.full_like(Fy_R, np.nan)

    Ex[mask, :] = np.abs(Fx_R[mask, :]) / (dy_abs[mask, None]**2)
    Ey[mask, :] = np.abs(Fy_R[mask, :]) / (dy_abs[mask, None]**2)

    figQ, axsQ = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    for k, dy in enumerate(dy_list):
        if abs(dy) < 0.01:
            continue
        axsQ[0].plot(t, Ex[k, :], linewidth=1, label=f"dy={dy:+.1f}")
        axsQ[1].plot(t, Ey[k, :], linewidth=1, label=f"dy={dy:+.1f}")

    axsQ[0].set_ylabel(r"$|R_x| / dy^2$  [N/mm$^2$]")
    axsQ[0].set_title("Domain-normalized Taylor remainder vs time (Fx)")
    axsQ[0].grid(True, alpha=0.3)

    axsQ[1].set_xlabel("Time [s]")
    axsQ[1].set_ylabel(r"$|R_y| / dy^2$  [N/mm$^2$]")
    axsQ[1].set_title("Domain-normalized Taylor remainder vs time (Fy)")
    axsQ[1].grid(True, alpha=0.3)

    axsQ[0].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    axsQ[1].legend(ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.show()

    Ex_mean = np.nanmean(Ex, axis=1)
    Ey_mean = np.nanmean(Ey, axis=1)

    plt.figure(figsize=(8, 4))
    plt.plot(dy_list, Ex_mean, marker="o", label=r"Mean $|R_x|/dy^2$")
    plt.plot(dy_list, Ey_mean, marker="s", label=r"Mean $|R_y|/dy^2$")
    plt.xlabel("dy [mm]")
    plt.ylabel(r"Mean normalized remainder  [N/mm$^2$]")
    plt.title("Mean second-order Taylor remainder (domain-normalized)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


