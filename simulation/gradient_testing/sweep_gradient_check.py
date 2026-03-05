
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
    T_end =  0.5 # [s]
    
    # # ---------------------------
    # # Run simulation
    # # ---------------------------

    h_list = [0.01, 0.025, 0.05, 0.075, 0.1]
    h_list = sorted(h_list)

    dx = 0.0

    # ----------------------------------------
    # Baseline run (only once): provides analytical derivative + baseline time
    # ----------------------------------------
    print(" --- Starting Baseline Simulation (dx=0, dy=0) --- ")
    t0_start = time.time()
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
    t0_end = time.time()
    print(f"Baseline simulation took {t0_end - t0_start:.4f} seconds")

    t0 = out0["t"]
    # Your analytical derivative from baseline (shape (2, N))
    dF_dy_ana_full = out0["dF_dy_analytical"]
    dFx_dy_ana_full = dF_dy_ana_full[0, :]
    dFy_dy_ana_full = dF_dy_ana_full[1, :]

    # ----------------------------------------
    # Error storage
    # ----------------------------------------
    mean_abs_err_vec = []   # mean over time of ||dF_num - dF_ana||2
    mean_abs_err_fx  = []   # mean over time of |dFx_num - dFx_ana|
    mean_abs_err_fy  = []   # mean over time of |dFy_num - dFy_ana|

    # ----------------------------------------
    # Loop over h values
    # ----------------------------------------
    for h in h_list:
        print(f"\n --- Step size h = {h:.5f}: Running offset sims (dy=±h) --- ")

        # +h simulation
        t_start = time.time()
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
        t_mid = time.time()
        print(f"  +h sim took {t_mid - t_start:.4f} seconds")

        # -h simulation
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
        t_end = time.time()
        print(f"  -h sim took {t_end - t_mid:.4f} seconds")

        # Grab averaged forces + times
        tp = out_p["t"]
        tm = out_m["t"]
        Fp_avg = out_p["forces_sim_average"]  # shape (2, Np)
        Fm_avg = out_m["forces_sim_average"]  # shape (2, Nm)

        # ----------------------------------------
        # Overlap region (avoid interpolation extrapolation)
        # ----------------------------------------
        t_min = max(t0.min(), tp.min(), tm.min())
        t_max = min(t0.max(), tp.max(), tm.max())
        valid = (t0 >= t_min) & (t0 <= t_max)

        t = t0[valid]
        dFx_ana = dFx_dy_ana_full[valid]
        dFy_ana = dFy_dy_ana_full[valid]

        # Interpolate offsets onto baseline time grid
        Fx_p = np.interp(t, tp, Fp_avg[0, :])
        Fy_p = np.interp(t, tp, Fp_avg[1, :])
        Fx_m = np.interp(t, tm, Fm_avg[0, :])
        Fy_m = np.interp(t, tm, Fm_avg[1, :])

        # ----------------------------------------
        # Central difference numerical derivative
        # ----------------------------------------
        dFx_num = (Fx_p - Fx_m) / (2.0 * h)
        dFy_num = (Fy_p - Fy_m) / (2.0 * h)

        # ----------------------------------------
        # Errors
        # ----------------------------------------
        err_fx = np.abs(dFx_num - dFx_ana)
        err_fy = np.abs(dFy_num - dFy_ana)
        err_vec = np.sqrt((dFx_num - dFx_ana)**2 + (dFy_num - dFy_ana)**2)

        mean_abs_err_fx.append(err_fx.mean())
        mean_abs_err_fy.append(err_fy.mean())
        mean_abs_err_vec.append(err_vec.mean())

        print(f"  mean(|ΔdFx/dy|) = {mean_abs_err_fx[-1]:.6g}")
        print(f"  mean(|ΔdFy/dy|) = {mean_abs_err_fy[-1]:.6g}")
        print(f"  mean(||ΔdF/dy||) = {mean_abs_err_vec[-1]:.6g}")

    # ----------------------------------------
    # Plot: Abs Error vs step size h (discrete points)
    # ----------------------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(h_list, mean_abs_err_vec, marker="o", linestyle="None", label="Mean abs vector error  ||Δ(dF/dy)||")
    plt.plot(h_list, mean_abs_err_fx,  marker="s", linestyle="None", label="Mean abs error in dFx/dy")
    plt.plot(h_list, mean_abs_err_fy,  marker="^", linestyle="None", label="Mean abs error in dFy/dy")
    plt.xlabel("Step size h [mm]")
    plt.ylabel("Mean absolute error [same units as dF/dy]")
    plt.title("Analytical dF/dy accuracy: Mean Abs. Error vs Step Size h")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()