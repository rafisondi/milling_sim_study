import math as m
import numpy as np
import time
import matplotlib.pyplot as plt

# Import necessary funcs / classes
from eraser_of_matter import milling_workpiece
from utils.analytical_mechanistic_milling import *

# ---------------------------
# Closed-form zero-order map (no integration)
# ---------------------------
def zero_order_force_with_displacement(Dx, c, a, N, phi_st, phi_ex, Ktc, Krc):
    dx, dy = float(Dx[0]), float(Dx[1])

    C = (N * a) / (8.0 * m.pi)

    dphi   = phi_ex - phi_st
    dsin2p = m.sin(2.0 * phi_ex) - m.sin(2.0 * phi_st)
    dcos2p = m.cos(2.0 * phi_ex) - m.cos(2.0 * phi_st)

    # A0 (closed form)
    A0x = C * ( Ktc * dcos2p - Krc * (2.0 * dphi - dsin2p) )
    A0y = C * ( Ktc * (2.0 * dphi - dsin2p) + Krc * dcos2p )

    # Averaged linear directional matrix (closed form)
    Axx = C * ( Ktc * (2.0 * dphi - dsin2p) + Krc * dcos2p )
    Axy = C * ( Ktc * dcos2p - Krc * (2.0 * dphi - dsin2p) )
    Ayx = -Axy
    Ayy =  Axx

    return np.array([
        A0x * c + Axx * dx + Axy * dy,
        A0y * c + Ayx * dx + Ayy * dy
    ], dtype=float)


# -----------------------------------------------------------------------------
# simulate() (your function, with phi_st/phi_ex stored)
# -----------------------------------------------------------------------------
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
    orientation_0: float = 0.0
):
    milling = milling_workpiece(xy_workpiece)
    milling.diameter_end_mill = tool_diameter
    milling.number_of_teeth = N_tool_teeth
    milling.radius_tool = tool_diameter / 2.0
    milling.slice_height = axial_cutting_depth

    Kt = milling.cutting_force_coefficient_Ktc
    Kr = milling.cutting_force_coefficient_Krc

    omega = -rev_per_min * 2 * m.pi / 60  # rad/s
    feed_speed = -omega / (2 * m.pi) * feed_per_tooth * milling.number_of_teeth  # mm/s
    time_step_p_rev = (2 * m.pi) / rot_angle_per_time_step
    sample_freq = time_step_p_rev * rev_per_min / 60

    dt = 1.0 / sample_freq
    t = np.arange(0.0, T_end, dt)
    N = len(t)

    rev_period = 1.0 / (rev_per_min / 60.0)
    samples_per_rev = int(sample_freq * rev_period)

    forces_sim = np.zeros((3, N))                   # simulated milling forces
    forces_analytical_avg = np.zeros((2, N))
    forces_zoe = np.zeros((2, N))
    forces_analytical = np.zeros((2, N))
    dF_dx_analytical = np.zeros((2, N))
    dF_dy_analytical = np.zeros((2, N))
    Hessian_vector_y = np.zeros((2, N))
    forces_sim_avg = np.zeros((2, N))

    phi_st_hist = np.zeros(N)
    phi_ex_hist = np.zeros(N)

    for i in range(N):
        # Current tool center (your feed in +x only)
        
        r_xy_current = xy_tool_center_0 + np.array([feed_speed * t[i], 0.0])

        b = -r_xy_current[1] + milling.radius_tool + xy_workpiece[1, -1]  # radial depth of cut
        v = -(r_xy_current[0] - xy_workpiece[0, -1])

        phi_st, phi_ex = compute_entry_exit_angles_downmilling(
            r_xy_current=r_xy_current,
            xy_workpiece=xy_workpiece,
            radius_tool=milling.radius_tool,
            diameter_end_mill=milling.diameter_end_mill
        )

        phi_st_hist[i] = phi_st
        phi_ex_hist[i] = phi_ex

        # Milling step
        orientation = orientation_0 + t[i] * omega
        milling.erase_step(r_xy_current, orientation, -1, t[i])

        if i >= 2:
            forces_sim[:, i] = milling.total_milling_force

            force_analytic = analytical_force(
                (rot_angle_per_time_step * i + np.pi / 2) % (2 * m.pi),
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
        dF_dx_analytical[:, i] = dF[:, 0]
        dF_dy_analytical[:, i] = dF[:, 1]

        # Rev-averaging block
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
        "phi_st": phi_st_hist,
        "phi_ex": phi_ex_hist,
    }

    

# -----------------------------------------------------------------------------
# __main__: full runnable section (workpiece + tool + spindle + plot)
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # ---------------------------
    # Workpiece geometry (rectangle)
    # 4 ------ 3
    # |        |
    # |        |
    # 1 ------ 2
    # ---------------------------
    wp_p1 = np.array([0.0,   0.0])
    wp_p2 = np.array([1000.0, 0.0])
    wp_p3 = np.array([1000.0, 76.0])
    wp_p4 = np.array([0.0,   76.0])

    xy_workpiece = np.array([wp_p1, wp_p2, wp_p3, wp_p4]).T  # shape (2,4)

    # ---------------------------
    # Tooling parameters
    # ---------------------------
    tool_diameter = 20.0          # [mm]
    tool_radius = tool_diameter / 2.0
    radial_engagement = 8.0       # [mm]
    axial_cutting_depth = 1.0     # [mm]
    N_tool_teeth = 1

    # Initial tool placement (same as your earlier logic)
    offset_tool_y = tool_radius - radial_engagement
    if offset_tool_y > 0:
        offset_tool_x = -np.sqrt(tool_radius**2 - offset_tool_y**2)
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
    feed_per_tooth = 0.15         # [mm/tooth]
    delta_deg = 1.0               # [deg]
    rot_angle_per_time_step = delta_deg * (2.0 * m.pi / 360.0)
    T_end = 0.5                   # [s]

    # ---------------------------
    # Baseline and offset
    # ---------------------------
    dx = 0.0
    dy = 0.1
    Dx = np.array([dx, dy], dtype=float)

    print(" --- Baseline simulation (dx=0, dy=0) --- ")
    t_start = time.time()
    out0 = simulate(
        xy_workpiece,
        xy_tool_center_0 + np.array([0.0, 0.0]),
        tool_diameter,
        N_tool_teeth,
        axial_cutting_depth,
        rev_per_min,
        feed_per_tooth,
        rot_angle_per_time_step,
        T_end
    )
    print(f"Baseline done in {time.time() - t_start:.3f} s")

    print(f" --- Offset simulation (dx=0, dy={dy}) --- ")
    t_start = time.time()
    out1 = simulate(
        xy_workpiece,
        xy_tool_center_0 + np.array([0.0, dy]),
        tool_diameter,
        N_tool_teeth,
        axial_cutting_depth,
        rev_per_min,
        feed_per_tooth,
        rot_angle_per_time_step,
        T_end
    )
    print(f"Offset done in {time.time() - t_start:.3f} s")

    # ---------------------------
    # Extract averaged signals
    # ---------------------------
    t = out0["t"]
    dt = out0["dt"]

    Fx0_avg = out0["forces_sim_average"][0, :]
    Fy0_avg = out0["forces_sim_average"][1, :]

    Fx1_avg = out1["forces_sim_average"][0, :]
    Fy1_avg = out1["forces_sim_average"][1, :]

    phi_st = out0["phi_st"]
    phi_ex = out0["phi_ex"]

    # Coefficients (constant here)
    Kt = out0["milling"].cutting_force_coefficient_Ktc
    Kr = out0["milling"].cutting_force_coefficient_Krc

    # ---------------------------
    # Build predicted: F_pred = F0_avg + (F_map(Dx) - F_map(0))
    # ---------------------------
    F_pred = np.zeros((2, len(t)))


    for i in range(len(t)):
        F_map_0 = zero_order_force_with_displacement(
            np.array([0.0, 0.0]),
            feed_per_tooth, axial_cutting_depth, N_tool_teeth,
            phi_st[i], phi_ex[i],
            Kt, Kr
        )
        F_map_D = zero_order_force_with_displacement(
            Dx,
            feed_per_tooth, axial_cutting_depth, N_tool_teeth,
            phi_st[i], phi_ex[i],
            Kt, Kr
        )
        

        dF = F_map_D - F_map_0
        F_pred[:, i] = np.array([Fx0_avg[i], Fy0_avg[i]]) + dF 

    # ---------------------------
    # Plot: baseline vs offset sim vs baseline + displacement-map
    # ---------------------------
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

    axs[0].plot(t, Fx0_avg, label="Baseline sim avg (dx=0,dy=0)")
    axs[0].plot(t, Fx1_avg, label=f"Offset sim avg (dx=0,dy={dy})")
    axs[0].scatter(t, F_pred[0, :], label=f"Baseline + map ΔF (dy={dy})", marker="+")
    axs[0].set_ylabel("Force [N]")
    axs[0].set_title("Avg Forces: baseline vs offset vs baseline + displacement map")
    axs[0].legend()

    axs[1].plot(t, Fy0_avg, label="Baseline sim avg (dx=0,dy=0)")
    axs[1].plot(t, Fy1_avg, label=f"Offset sim avg (dx=0,dy={dy})")
    axs[1].scatter(t, F_pred[1, :], label=f"Baseline + map ΔF (dy={dy})",marker="+")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Force [N]")
    axs[1].legend()

    plt.tight_layout()
    plt.show()
