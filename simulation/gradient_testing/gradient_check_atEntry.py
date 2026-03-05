
import math as m
import numpy as np
import time
import matplotlib.pyplot as plt

# Import necessary funcs / classes
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *



def simulate(        
            xy_workpiece: np.ndarray,
            xy_tool_center_0: np.ndarray,
            tool_diameter: float,
            N_tool_teeth: int,
            axial_cutting_depth: float,
            rev_per_min: float,
            feed_per_tooth: float,
            rot_angle_per_time_step: float,
            T_end: float,
            orientation_0: float = 0.0):
    
    # ---------------------------
    # Init milling geometry/model
    # ---------------------------
    milling = milling_workpiece(xy_workpiece)
    milling.diameter_end_mill = tool_diameter
    milling.number_of_teeth = N_tool_teeth
    milling.radius_tool = tool_diameter / 2.
    milling.slice_height = axial_cutting_depth
    
    Kt = milling.cutting_force_coefficient_Ktc
    Kr = milling.cutting_force_coefficient_Krc
    
    # ---------------------------
    # Simulation and Process parameters
    # ---------------------------
    omega = -rev_per_min * 2 * m.pi / 60  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * milling.number_of_teeth  # mm/s
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60
    
    dt = 1.0 / sample_freq
    t = np.arange(0.0, T_end, dt)
    N = len(t)
    
    rev_period = 1/(rev_per_min / 60.0)
    samples_per_rev = int( sample_freq * rev_period )
    
    # ---------------------------
    # Lists
    # ---------------------------
    forces_sim = np.zeros((3, N))                   # simulated milling forces (from milling model)
    forces_analytical_avg = np.zeros((2, N))        # analytical average F returns [Fx, Fy] 
    forces_zoe = np.zeros((2, N))                   # zero-order average F returns [Fx, Fy]
    forces_analytical = np.zeros((2, N))            # analytical F returns [Fx, Fy] (per your usage)
    dF_dx_analytical = np.zeros((2, N))            # analytical dF/dx 
    dF_dy_analytical = np.zeros((2, N))            # analytical dF/dy
    Hessian_vector_y = np.zeros((2, N))            # analytical d2F/dy2
    forces_sim_avg = np.zeros((2, N))
    
    for i in range(N):
        # Current tool center
        r_xy_current = xy_tool_center_0 + np.array([feed_speed * t[i], 0.0])
        
        b = - r_xy_current[1] + milling.radius_tool + xy_workpiece[1,-1]  # radial depth of cut 
        v = -(r_xy_current[0] - xy_workpiece[0,-1]) 
        
        phi_st, phi_ex = compute_entry_exit_angles_downmilling( r_xy_current=r_xy_current,
                                                                xy_workpiece=xy_workpiece,
                                                                radius_tool=milling.radius_tool,
                                                                diameter_end_mill=milling.diameter_end_mill)
            
        # ---------------------------
        # Milling simulation step
        # ---------------------------
        orientation = orientation_0 + t[i] * omega # current tool orientation [rad]
        milling.erase_step(r_xy_current, orientation, -1, t[i])
        
        if i >= 2:
            forces_sim[:, i] = milling.total_milling_force
            
            force_analytic = analytical_force(
                (rot_angle_per_time_step * i + np.pi/2 )%(2*m.pi),
                phi_st,
                phi_ex,
                milling.axial_cutting_depth,
                feed_per_tooth,
                Kt,
                Kr
            )
            forces_analytical[:, i] = force_analytic
        
        dF = zero_order_dF(
            feed_per_tooth,
            axial_cutting_depth,
            b,
            v,
            milling.diameter_end_mill,
            N_tool_teeth,
            phi_st,
            phi_ex,
            Kt,
            Kr
        )
        dF_dx_analytical[:, i] = dF[:,0]
        dF_dy_analytical[:, i] = dF[:,1]
        
        
        if i % samples_per_rev == 0 and i >= samples_per_rev:
            res = zero_order_force_analytical(
                feed_per_tooth,
                axial_cutting_depth,
                N_tool_teeth,
                phi_st,
                phi_ex,
                Kt,
                Kr
            )
            forces_zoe[:, i - samples_per_rev:i] = res[:, np.newaxis]
            
            average = forces_sim[0:2, i - samples_per_rev:i].mean(axis=1)
            forces_sim_avg[:, i - samples_per_rev:i] = average[:, np.newaxis]


    return {
        "milling": milling,
        "t": t,
        "forces_sim": forces_sim,
        "forces_analytical": forces_analytical,
        "forces_analytical_avg": forces_analytical_avg,
        "dF_dx_analytical": dF_dx_analytical,
        "dF_dy_analytical": dF_dy_analytical,
        "forces_zoe": forces_zoe,
        "Hessian_vector_y": Hessian_vector_y,
        "dt": dt,
        "sample_freq": sample_freq,
        "feed_speed": feed_speed,
        "forces_sim_average": forces_sim_avg,
    }

# if __name__ == "__main__":
    # # ---------------------------
    # # Workpiece geometry
    # # ---------------------------
    # wp_p1 = np.array([0,0])
    # wp_p2 = np.array([1000,0])
    # wp_p3 = np.array([1000,76])
    # wp_p4 = np.array([0,76])

    # xy_workpiece = np.array([wp_p1, wp_p2, wp_p3, wp_p4]).T

    # # ---------------------------
    # # Tooling parameters
    # # ---------------------------
    # tool_diameter = 20.0  # [mm]
    # tool_radius = tool_diameter / 2.0
    # radial_engagement = 5.0  # [mm]
    # axial_cutting_depth = 1.0  # [mm]
    # N_tool_teeth = 1

    # offset_tool_y = tool_radius - radial_engagement
    # if offset_tool_y > 0:
    #     offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    # else:
    #     offset_tool_x = -tool_radius

    # xy_tool_center_0 = np.array([wp_p4[0] + offset_tool_x,
    #                              wp_p4[1] + offset_tool_y])

    # # ---------------------------
    # # Spindle / feed / discretization
    # # ---------------------------
    # rev_per_min = 8000
    # feed_per_tooth = 0.15
    # delta_deg = 0.1
    # rot_angle_per_time_step = delta_deg * (2*m.pi / 360)
    # T_end = 0.125

    # h_step = 0.01 # mm
    # dx = 0.0

    # # ----------------------------------------
    # # Target evaluation times and helper
    # # ----------------------------------------
    # t_targets = np.array([0.05, 0.075, 0.1, 0.125])

    # def interp_at(t_src, y_src, t_q):
    #     return np.interp(t_q, t_src, y_src)

    # # ----------------------------------------
    # # Baseline run (dy=0): analytical gradient reference + feed speed
    # # ----------------------------------------
    # out0 = simulate(
    #     xy_workpiece,
    #     xy_tool_center_0 + np.array([dx, 0.0]),
    #     tool_diameter,
    #     N_tool_teeth,
    #     axial_cutting_depth,
    #     rev_per_min,
    #     feed_per_tooth,
    #     rot_angle_per_time_step,
    #     T_end
    # )
    
    # out_p = simulate(
    #         xy_workpiece,
    #         xy_tool_center_0 + np.array([dx, +h_step]),
    #         tool_diameter,
    #         N_tool_teeth,
    #         axial_cutting_depth,
    #         rev_per_min,
    #         feed_per_tooth,
    #         rot_angle_per_time_step,
    #         T_end
    #     )
    # out_m = simulate(
    #         xy_workpiece,
    #         xy_tool_center_0 + np.array([dx, -h_step]),
    #         tool_diameter,
    #         N_tool_teeth,
    #         axial_cutting_depth,
    #         rev_per_min,
    #         feed_per_tooth,
    #         rot_angle_per_time_step,
    #         T_end
    #     )


    # t0 = out0["t"]
    # feed_speed = out0["feed_speed"]
    # x0 = feed_speed * t0   # x-position along feed direction (relative)

    # # Analytical gradients (reference)
    # dFx_ana = out0["dF_dy_analytical"][0, :]
    # dFy_ana = out0["dF_dy_analytical"][1, :]

    # # Use averaged simulated forces (what you used before)
    # F0_avg = out0["forces_sim_average"]  
    # Fp_avg = out_p["forces_sim_average"]   # shape (2, Np)
    # Fm_avg = out_m["forces_sim_average"]   # shape (2, Nm)
    
    # F0_y = out0["forces_analytical"]
    # Fp_y = out0["forces_analytical"]
    # Fm_y = out0["forces_analytical"]
    
    # tp = out_p["t"]
    # tm = out_m["t"]

    # # If time grids match, no interpolation needed; otherwise align to baseline time grid
    # same_grid = (
    #     (len(tp) == len(t0)) and (len(tm) == len(t0)) and
    #     np.allclose(tp, t0, rtol=0, atol=1e-12) and
    #     np.allclose(tm, t0, rtol=0, atol=1e-12)
    # )

    # if same_grid:
    #     Fx_p = Fp_avg[0, :]
    #     Fy_p = Fp_avg[1, :]
    #     Fx_m = Fm_avg[0, :]
    #     Fy_m = Fm_avg[1, :]
    # else:
    #     Fx_p = np.interp(t0, tp, Fp_avg[0, :])
    #     Fy_p = np.interp(t0, tp, Fp_avg[1, :])
    #     Fx_m = np.interp(t0, tm, Fm_avg[0, :])
    #     Fy_m = np.interp(t0, tm, Fm_avg[1, :])

    # # Central finite difference in y (fixed h_step)
    # dFx_fd = (Fx_p - Fx_m) / (2.0 * h_step)
    # dFy_fd = (Fy_p - Fy_m) / (2.0 * h_step)

    # # Errors vs x
    # err_fx = np.abs(dFx_fd - dFx_ana)
    # err_fy = np.abs(dFy_fd - dFy_ana)
    # err_vec = np.sqrt((dFx_fd - dFx_ana)**2 + (dFy_fd - dFy_ana)**2)

    # # ----------------------------------------
    # # OPTIONAL: focus on your original "target times" (i.e., specific x-positions)
    # # ----------------------------------------
    # t_targets = np.array([0.10, 0.15, 0.20, 0.25])
    # mask_targets = np.isin(np.round(t0, 12), np.round(t_targets, 12))
    # # If your t0 doesn't hit those exactly, use nearest indices instead:
    # if not np.any(mask_targets):
    #     idx_targets = [int(np.argmin(np.abs(t0 - tt))) for tt in t_targets]
    #     mask_targets = np.zeros_like(t0, dtype=bool)
    #     mask_targets[idx_targets] = True

    # # ----------------------------------------
    # # Plot 1: Gradients vs x (analytical vs FD)
    # # ----------------------------------------
    # plt.figure(figsize=(9, 5))
    # plt.plot(x0, dFx_ana, label="Analytical dFx/dy")
    # plt.plot(x0, dFx_fd,  label="FD dFx/dy (central, fixed h)")
    # plt.scatter(x0[mask_targets], dFx_fd[mask_targets], s=35)  # highlight target x's
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("dFx/dy")
    # plt.title("dFx/dy vs x (Analytical vs Central FD)")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()

    # plt.figure(figsize=(9, 5))
    # plt.plot(x0, dFy_ana, label="Analytical dFy/dy")
    # plt.plot(x0, dFy_fd,  label="FD dFy/dy (central, fixed h)")
    # plt.scatter(x0[mask_targets], dFy_fd[mask_targets], s=35)  # highlight target x's
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("dFy/dy")
    # plt.title("dFy/dy vs x (Analytical vs Central FD)")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()

    # # ----------------------------------------
    # # Plot 2: Errors vs x
    # # ----------------------------------------
    # plt.figure(figsize=(9, 5))
    # plt.plot(x0, err_fx, label=r"$|\Delta(dF_x/dy)|$")
    # plt.plot(x0, err_fy, label=r"$|\Delta(dF_y/dy)|$")
    # plt.plot(x0, err_vec, label=r"$\|\Delta(d\mathbf{F}/dy)\|_2$")
    # plt.scatter(x0[mask_targets], err_vec[mask_targets], s=35)  # highlight target x's
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("absolute error")
    # plt.title("Central FD vs Analytical gradient error vs x (fixed h)")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    
    
    
    


    
    # plt.figure(figsize=(9, 5))
    # plt.plot(t0, F0_avg[1],  label="Fy0")
    # plt.plot(t0, Fp_avg[1],  label="Fyp")
    # plt.plot(t0, Fm_avg[1],  label="Fym")
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("F")
    # plt.title(f"Forces at y=y0, y+dh y-dh, {h_step}")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    

    
    # plt.figure(figsize=(9, 5))
    # plt.plot(t0, dFy_ana,  label="dFdy Analyitc")
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("F")
    # plt.title("dF dy")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()


    # plt.figure(figsize=(9, 5))
    # plt.plot(t0, F0_y[1],  label="Fy0")
    # plt.plot(t0, Fp_y[1],  label="Fyp")
    # plt.plot(t0, Fm_y[1],  label="Fym")
    # plt.xlabel("x position [mm]  (x = feed_speed * t)")
    # plt.ylabel("F")
    # plt.title("Forces at y=y0, y+dh y-dh")
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    
    # plt.show()
    
if __name__ == "__main__":
    # ---------------------------
    # Workpiece geometry
    # ---------------------------
    wp_p1 = np.array([0,0])
    wp_p2 = np.array([1000,0])
    wp_p3 = np.array([1000,76])
    wp_p4 = np.array([0,76])

    xy_workpiece = np.array([wp_p1, wp_p2, wp_p3, wp_p4]).T

    # ---------------------------
    # Tooling parameters
    # ---------------------------
    tool_diameter = 20.0  # [mm]
    tool_radius = tool_diameter / 2.0
    radial_engagement = 8.0  # [mm]
    axial_cutting_depth = 1.0  # [mm]
    N_tool_teeth = 1

    offset_tool_y = tool_radius - radial_engagement
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
    else:
        offset_tool_x = -tool_radius

    xy_tool_center_0 = np.array([wp_p4[0] + offset_tool_x,
                                 wp_p4[1] + offset_tool_y])

    # ---------------------------
    # Spindle / feed / discretization
    # ---------------------------
    rev_per_min = 8000
    feed_per_tooth = 0.15
    delta_deg = 0.25
    rot_angle_per_time_step = delta_deg * (2*m.pi / 360)
    T_end = 0.3

    h_list = sorted([0.1])
    dx = 0.0

    # ----------------------------------------
    # Target evaluation times and helper
    # ----------------------------------------
    t_targets = np.array([0.1, 0.15, 0.2, 0.25])

    def interp_at(t_src, y_src, t_q):
        return np.interp(t_q, t_src, y_src)

    # ----------------------------------------
    # Baseline run (dy=0): analytical gradient reference + feed speed
    # ----------------------------------------
    out0 = simulate(
        xy_workpiece,
        xy_tool_center_0 + np.array([dx, 0.0]),
        tool_diameter,
        N_tool_teeth,
        axial_cutting_depth,
        rev_per_min,
        feed_per_tooth,
        rot_angle_per_time_step,
        T_end
    )

    t0 = out0["t"]
    feed_speed = out0["feed_speed"]  # mm/s (note sign in your model)

    # clamp targets to simulation time range
    t_targets = t_targets[(t_targets >= t0.min()) & (t_targets <= t0.max())]
    x_targets = feed_speed * t_targets  # x-position relative to initial tool center

    dFx_dy_ana_full = out0["dF_dy_analytical"][0, :]
    dFy_dy_ana_full = out0["dF_dy_analytical"][1, :]

    dFx_ana_at = interp_at(t0, dFx_dy_ana_full, t_targets)
    dFy_ana_at = interp_at(t0, dFy_dy_ana_full, t_targets)

    # ----------------------------------------
    # Collect errors vs x for each h
    # ----------------------------------------
    # shape: (len(h_list), len(t_targets))
    err_fx = np.zeros((len(h_list), len(t_targets)))
    err_fy = np.zeros((len(h_list), len(t_targets)))
    err_vec = np.zeros((len(h_list), len(t_targets)))

    for j, h in enumerate(h_list):
        out_p = simulate(
            xy_workpiece,
            xy_tool_center_0 + np.array([dx, +h]),
            tool_diameter,
            N_tool_teeth,
            axial_cutting_depth,
            rev_per_min,
            feed_per_tooth,
            rot_angle_per_time_step,
            T_end
        )
        out_m = simulate(
            xy_workpiece,
            xy_tool_center_0 + np.array([dx, -h]),
            tool_diameter,
            N_tool_teeth,
            axial_cutting_depth,
            rev_per_min,
            feed_per_tooth,
            rot_angle_per_time_step,
            T_end
        )

        tp, tm = out_p["t"], out_m["t"]
        Fp_avg = out_p["forces_sim_average"]  # (2, Np)
        Fm_avg = out_m["forces_sim_average"]  # (2, Nm)

        Fx_p_at = interp_at(tp, Fp_avg[0, :], t_targets)
        Fy_p_at = interp_at(tp, Fp_avg[1, :], t_targets)
        Fx_m_at = interp_at(tm, Fm_avg[0, :], t_targets)
        Fy_m_at = interp_at(tm, Fm_avg[1, :], t_targets)

        dFx_num_at = (Fx_p_at - Fx_m_at) / (2.0 * h)
        dFy_num_at = (Fy_p_at - Fy_m_at) / (2.0 * h)

        err_fx[j, :] = np.abs(dFx_num_at - dFx_ana_at)
        err_fy[j, :] = np.abs(dFy_num_at - dFy_ana_at)
        err_vec[j, :] = np.sqrt((dFx_num_at - dFx_ana_at)**2 + (dFy_num_at - dFy_ana_at)**2)

    # ----------------------------------------
    # Plots: error vs x-position
    # ----------------------------------------
    # 1) |Δ dFx/dy|
    plt.figure(figsize=(8, 5))
    for j, h in enumerate(h_list):
        plt.plot(x_targets, err_fx[j, :], marker="o", linestyle="-", label=f"h={h:g} mm")
    plt.xlabel("x position [mm] (x = feed_speed * t)")
    plt.ylabel(r"$|\Delta(\partial F_x/\partial y)|$")
    plt.title(r"Finite-difference vs Analytical: Error in $\partial F_x/\partial y$ vs x")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 2) |Δ dFy/dy|
    plt.figure(figsize=(8, 5))
    for j, h in enumerate(h_list):
        plt.plot(x_targets, err_fy[j, :], marker="o", linestyle="-", label=f"h={h:g} mm")
    plt.xlabel("x position [mm] (x = feed_speed * t)")
    plt.ylabel(r"$|\Delta(\partial F_y/\partial y)|$")
    plt.title(r"Finite-difference vs Analytical: Error in $\partial F_y/\partial y$ vs x")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 3) ||Δ grad||
    plt.figure(figsize=(8, 5))
    for j, h in enumerate(h_list):
        plt.plot(x_targets, err_vec[j, :], marker="o", linestyle="-", label=f"h={h:g} mm")
    plt.xlabel("x position [mm] (x = feed_speed * t)")
    plt.ylabel(r"$\|\Delta(\partial \mathbf{F}/\partial y)\|_2$")
    plt.title(r"Finite-difference vs Analytical: Gradient error norm vs x")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.show()
