
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
    
    # print(" --- Starting Simulation Pass 0 --- ")
    
    # start = time.time()
    # out = simulate(
    #     xy_workpiece,
    #     xy_tool_center_0,
    #     tool_diameter,
    #     N_tool_teeth,
    #     axial_cutting_depth,
    #     rev_per_min,
    #     feed_per_tooth,
    #     rot_angle_per_time_step,
    #     T_end
    # )

    # end = time.time()
    # print(f"Simulation took {end - start:.4f} seconds")

    # dx = 0.0  
    # dy = 1.0 #.0 #  positive y moves tool away from workpiece in this setup
    
    # print(" --- Starting Simulation Pass 1 --- ")
    # start = time.time()
    # out1 = simulate( # Base truth for Taylor approximation comparison
    #     xy_workpiece,
    #     xy_tool_center_0 + np.array([dx, dy]),
    #     tool_diameter,
    #     N_tool_teeth,
    #     axial_cutting_depth,
    #     rev_per_min,
    #     feed_per_tooth,
    #     rot_angle_per_time_step,
    #     T_end
    # )
    
    # end = time.time()
    # print(f"Simulation took {end - start:.4f} seconds")
    
    
    # # ----------------------------
    # # Plotting 
    # # ---------------------------
    
    # # DATA 0
    # t0 = out["t"]
    # dt = out["dt"]
    # Fx0 = out["forces_sim"][0,:]
    # Fy0 = out["forces_sim"][1,:]
    # Fx0_analytical = out["forces_analytical"][0,:]
    # Fy0_analytical = out["forces_analytical"][1,:]

    # Fx0_zoe = out["forces_zoe"][0,:]
    # Fy0_zoe = out["forces_zoe"][1,:]
    # Fx0_avg = out["forces_sim_average"][0,:]
    # Fy0_avg = out["forces_sim_average"][1,:]
    
    # dF_dx_analytical = out["dF_dx_analytical"]
    # dF_dy_analytical = out["dF_dy_analytical"]
    
    # Fx_lin = Fx0_zoe + dF_dx_analytical[0, :] * dx + dF_dy_analytical[0, :] * dy
    # Fy_lin = Fy0_zoe + dF_dx_analytical[1, :] * dx + dF_dy_analytical[1, :] * dy
    # F_lin = np.vstack((Fx_lin, Fy_lin))
    
    # # DATA 1 
    # t1 = out1["t"]
    # Fx1 = out1["forces_sim"][0,:]
    # Fy1 = out1["forces_sim"][1,:]
    # Fx1_zoe = out1["forces_zoe"][0,:]
    # Fy1_zoe = out1["forces_zoe"][1,:]
    # Fx1_avg = out1["forces_sim_average"][0,:]
    # Fy1_avg = out1["forces_sim_average"][1,:]
    
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # # Fx
    # axs[0].plot(t0, Fx0_avg, label="Case 0 Avg Fx")
    # axs[0].scatter(t1, Fx1_avg, label="Case 1 Avg Fx")
    # axs[0].set_ylabel("Force [N]")
    # axs[0].set_title("Average Milling Forces")
    # axs[0].legend()

    # # Fy
    # axs[1].plot(t0, Fy0_avg, label="Case 0 Avg Fy")
    # axs[1].scatter(t1, Fy1_avg, label="Case 1 Avg Fy", c = "r" , marker ="+" )
    # axs[1].set_xlabel("Time [s]")
    # axs[1].set_ylabel("Force [N]")
    # axs[1].legend()
    
    # fig1, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # # Fx
    # axs[0].plot(t0, (Fx0_avg - Fx1_avg), label="Case 0 Avg Fx")
    # # axs[0].scatter(t1, Fx1_avg, label="Case 1 Avg Fx")
    # axs[0].set_ylabel("Force [N]")
    # axs[0].set_title("Difference in Average Milling Forces")
    # axs[0].legend()

    # # Fy
    # axs[1].plot(t0, Fy0_avg -Fy1_avg, label="Case 0 Avg Fy")
    # # axs[1].scatter(t1, Fy1_avg, label="Case 1 Avg Fy", c = "r" , marker ="+" )
    # axs[1].set_xlabel("Time [s]")
    # axs[1].set_ylabel("Force [N]")
    # axs[1].legend()

    # plt.tight_layout()
    # plt.show()
    
    # ---------------------------
# Run simulation (Base: Pass 0)
# ---------------------------

# Baseline simulation (dx=0, dy=0)
# ---------------------------
    
    # # ---------------------------
    # # Plot 1: Average forces (F0 + all swept)
    # # ---------------------------

    # fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # # Fx
    # axs[0].plot(t0, Fx0_avg, linewidth=2, label="Baseline (dy=0) Avg Fx")
    # for dy in dy_vals:
    #     t = sweep[dy]["t"]
    #     Fx = sweep[dy]["Fx_avg"]
    #     axs[0].plot(t, Fx, alpha=0.9, label=f"dy={dy:+.1f}")
    # axs[0].set_ylabel("Force [N]")
    # axs[0].set_title("Average Milling Forces (dy sweep)")
    # axs[0].legend(ncol=2, fontsize=8)

    # # Fy
    # axs[1].plot(t0, Fy0_avg, linewidth=2, label="Baseline (dy=0) Avg Fy")
    # for dy in dy_vals:
    #     t = sweep[dy]["t"]
    #     Fy = sweep[dy]["Fy_avg"]
    #     axs[1].plot(t, Fy, alpha=0.9, label=f"dy={dy:+.1f}")
    # axs[1].set_xlabel("Time [s]")
    # axs[1].set_ylabel("Force [N]")
    # axs[1].legend(ncol=2, fontsize=8)


    # # ---------------------------
    # # Plot 2: Error difference to F0 (F0 - F(dy))
    # # ---------------------------

    # fig2, axs2 = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    # # Fx error
    # for dy in dy_vals:
    #     t = sweep[dy]["t"]
    #     Fx_err = Fx0_avg - sweep[dy]["Fx_avg"]
    #     axs2[0].plot(t, Fx_err, label=f"dy={dy:+.1f}")
    # axs2[0].axhline(0, linewidth=1)
    # axs2[0].set_ylabel("Force [N]")
    # axs2[0].set_title("Difference to Baseline: F0_avg - F(dy)_avg")
    # axs2[0].legend(ncol=2, fontsize=8)

    # # Fy error
    # for dy in dy_vals:
    #     t = sweep[dy]["t"]
    #     Fy_err = Fy0_avg - sweep[dy]["Fy_avg"]
    #     axs2[1].plot(t, Fy_err, label=f"dy={dy:+.1f}")
    # axs2[1].axhline(0, linewidth=1)
    # axs2[1].set_xlabel("Time [s]")
    # axs2[1].set_ylabel("Force [N]")
    # axs2[1].legend(ncol=2, fontsize=8)

    # plt.tight_layout()
    # plt.show()
    dx = 0.0
    h = 0.01
    
    print(" --- Starting Simulation Baseline (dx=0, dy=0) --- ")
    start = time.time()
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
    end = time.time()
    print(f"Baseline simulation took {end - start:.4f} seconds")

    t0 = out0["t"]
    Fx0_avg = out0["forces_sim_average"][0, :]
    Fy0_avg = out0["forces_sim_average"][1, :]

    # ---------------------------
    # Offset simulations (dx=0, dy=±h)
    # ---------------------------
    outs = {}
    for dy in (-h, +h):
        print(f" --- Starting Simulation Offset (dx=0, dy={dy:+.4f}) --- ")
        start = time.time()
        out = simulate(
            xy_workpiece,
            xy_tool_center_0 + np.array([dx, dy]),
            tool_diameter,
            N_tool_teeth,
            axial_cutting_depth,
            rev_per_min,
            feed_per_tooth,
            rot_angle_per_time_step,
            T_end
        )
        end = time.time()
        print(f"Offset simulation took {end - start:.4f} seconds")
        outs[dy] = out


    t0 = out0["t"]
    Fx0_avg = out0["forces_sim_average"][0, :]
    Fy0_avg = out0["forces_sim_average"][1, :]

    # analytical derivative from baseline
    dF_dy_analytical = out0["dF_dy_analytical"]     # shape (2, n0) expected
    dFx_dy_ana = dF_dy_analytical[0, :]
    dFy_dy_ana = dF_dy_analytical[1, :]

    # offset data
    tp = outs[+h]["t"]
    Fp_avg = outs[+h]["forces_sim_average"]
    tm = outs[-h]["t"]
    Fm_avg = outs[-h]["forces_sim_average"]

    # ---------------------------
    # Overlap check (start/end)
    # ---------------------------
    t_min = max(t0.min(), tp.min(), tm.min())
    t_max = min(t0.max(), tp.max(), tm.max())

    valid = (t0 >= t_min) & (t0 <= t_max)

    t = t0[valid]
    Fx0 = Fx0_avg[valid]
    Fy0 = Fy0_avg[valid]
    dFx_ana = dFx_dy_ana[valid]
    dFy_ana = dFy_dy_ana[valid]

    # interpolate offset averages onto baseline time (restricted to overlap)
    Fx_p = np.interp(t, tp, Fp_avg[0, :])
    Fy_p = np.interp(t, tp, Fp_avg[1, :])
    Fx_m = np.interp(t, tm, Fm_avg[0, :])
    Fy_m = np.interp(t, tm, Fm_avg[1, :])

    # ---------------------------
    # Numerical derivatives vs baseline
    # ---------------------------
    dFx_dy_fwd = (Fx_p - Fx0) / h
    dFy_dy_fwd = (Fy_p - Fy0) / h

    dFx_dy_bwd = (Fx0 - Fx_m) / h
    dFy_dy_bwd = (Fy0 - Fy_m) / h

    # ---------------------------
    # Plot numerical + analytical
    # ---------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axs[0].plot(t, dFx_dy_fwd, label=f"Numerical forward (F(+{h})-F0)/{h}")
    axs[0].plot(t, dFx_dy_bwd, label=f"Numerical backward (F0-F(-{h}))/{h}")
    axs[0].plot(t, dFx_ana, linestyle="--", linewidth=2, label="Analytical dFx/dy (baseline)")
    axs[0].set_ylabel("dFx/dy [N / (dy-unit)]")
    axs[0].set_title("dF/dy vs Time: Numerical (baseline-referenced) vs Analytical")
    axs[0].legend()

    axs[1].plot(t, dFy_dy_fwd, label=f"Numerical forward (F(+{h})-F0)/{h}")
    axs[1].plot(t, dFy_dy_bwd, label=f"Numerical backward (F0-F(-{h}))/{h}")
    axs[1].plot(t, dFy_ana, linestyle="--", linewidth=2, label="Analytical dFy/dy (baseline)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("dFy/dy [N / (dy-unit)]")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
    
    # ---------------------------
    # Central difference derivative (midpoint)
    # dF/dy ≈ (F(+h) - F(-h)) / (2h)
    # ---------------------------

    # baseline time + analytical derivative
    t0 = out0["t"]
    dF_dy_analytical = out0["dF_dy_analytical"]
    dFx_dy_ana = dF_dy_analytical[0, :]
    dFy_dy_ana = dF_dy_analytical[1, :]

    # offset data
    tp = outs[+h]["t"]
    Fp_avg = outs[+h]["forces_sim_average"]
    tm = outs[-h]["t"]
    Fm_avg = outs[-h]["forces_sim_average"]

    # ---------------------------
    # Overlap check (avoid interp extrapolation artifacts)
    # ---------------------------
    t_min = max(t0.min(), tp.min(), tm.min())
    t_max = min(t0.max(), tp.max(), tm.max())
    valid = (t0 >= t_min) & (t0 <= t_max)

    t = t0[valid]
    dFx_ana = dFx_dy_ana[valid]
    dFy_ana = dFy_dy_ana[valid]

    # interpolate offset averages onto baseline time (restricted to overlap)
    Fx_p = np.interp(t, tp, Fp_avg[0, :])
    Fy_p = np.interp(t, tp, Fp_avg[1, :])
    Fx_m = np.interp(t, tm, Fm_avg[0, :])
    Fy_m = np.interp(t, tm, Fm_avg[1, :])

    # ---------------------------
    # Numerical derivative (central difference)
    # ---------------------------
    dFx_dy_num = (Fx_p - Fx_m) / (2.0 * h)
    dFy_dy_num = (Fy_p - Fy_m) / (2.0 * h)

    # ---------------------------
    # Plot numerical (central) + analytical
    # ---------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axs[0].plot(t, dFx_dy_num, label=f"Numerical central (F(+{h})-F(-{h}))/(2{h})")
    axs[0].plot(t, dFx_ana, linestyle="--", linewidth=2, label="Analytical dFx/dy (baseline)")
    axs[0].set_ylabel("dFx/dy [N / (dy-unit)]")
    axs[0].set_title("dF/dy vs Time: Central Difference vs Analytical")
    axs[0].legend()

    axs[1].plot(t, dFy_dy_num, label=f"Numerical central (F(+{h})-F(-{h}))/(2{h})")
    axs[1].plot(t, dFy_ana, linestyle="--", linewidth=2, label="Analytical dFy/dy (baseline)")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("dFy/dy [N / (dy-unit)]")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
