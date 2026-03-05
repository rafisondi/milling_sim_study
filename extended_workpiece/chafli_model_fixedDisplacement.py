import os
import numpy as np
import pandas as pd
from milling_path import MillingPath
from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D, T_inv
from feed_optimizer2 import FeedOptimizer
from datetime import date

import psutil

process = psutil.Process(os.getpid())


# =========================
# Finetuning / Config
# =========================
DT = 1/4.8*1e-4
Samples_per_Period = 360
Period = DT * Samples_per_Period
frequency = 1 / Period
rpm = 60 * frequency
t_limit = 10

PRINT_EVERY_N = 100

FOLDER = os.path.join("data", "experiments")

OFFSET = None
OFFSET_SIDE = "left"

TOOL_DIAMETER_MM = 20.0
TOOL_RADIUS_MM = TOOL_DIAMETER_MM / 2.0
AXIAL_CUTTING_DEPTH_MM = 1.0

FEED_SPEED = 40.0
RPM_TARGET = rpm
Z_TEETH = 4
SPINDLE_SPIN = -1

# Single displacement in WP frame [dx_mm, dy_mm]
DH_MM = np.array([0.5, 0.0])

SIM_NAME = f"Model_{FEED_SPEED}mmps_{RPM_TARGET}rpm_{int(1/DT)}Hz_TRAJECTORY_{date.today():%Y%m%d%mm}"

# Compatibility aliases for legacy helper functions below
dt = DT
sim_name = SIM_NAME
folder = FOLDER
feed_speed = FEED_SPEED
axial_cutting_depth = AXIAL_CUTTING_DEPTH_MM
spindle_spin = SPINDLE_SPIN
offset = OFFSET
offset_side = OFFSET_SIDE
feed_per_tooth = FEED_SPEED / ((RPM_TARGET / 60.0) * Z_TEETH)
feed_speed_rev_comp = FEED_SPEED

# TODO: Adjust if milling path changes
T_base_workpiece = np.array(
                            [[ 0.,  1.,  0., -7.         ],
                             [-1.,  0.,  0., -1667.40003 ],
                             [ 0.,  0.,  1.,  0.48569812 ],
                             [ 0.,  0.,  0.,  1.         ]]
                            )

# sim_name = "Optimization_13_08_25_6_milling_force_sim_full_feedback_model3"
# milling_path_corrected_df = pd.read_csv(os.path.join("data", "experiments", "Optimization_workpiece_long", "Optimization_13_08_25_6", "milling_path_corrected_with_tracking_error.csv"))
# x_ref_corrected = milling_path_corrected_df["x tool center Base"]
# y_ref_corrected = milling_path_corrected_df["y tool center Base"]

def spindle_from_rpm_and_feed(feed_mm_s: float, rpm: float, z_teeth: int, spin_dir: int):
    rev_per_s = rpm / 60.0
    omega = spin_dir * 2.0 * np.pi * rev_per_s
    fz_mm = feed_mm_s / (rev_per_s * z_teeth)
    return omega, fz_mm


def chafli_model_preparation(dh=DH_MM):
    #milling_path_df = pd.read_csv( " /home/irz/code/robot-milling-feed-planning/data/paths/Workpiece_long_with_start_milling_path.csv " )#os.path.join("data", "paths", "Workpiece_long_with_start_milling_path.csv"), index_col=0)
    milling_path_df = pd.read_csv(os.path.join("data", "paths", "Workpiece_long_with_start_milling_path.csv"), index_col=0)

    dh = np.asarray(dh, dtype=float).reshape(-1)
    if dh.size != 2:
        raise ValueError(f"Expected dh = [dx_mm, dy_mm], got shape {dh.shape}")
    x_offset, y_offset = dh
    print(f"Running single displacement: dx={x_offset:+.3f} mm, dy={y_offset:+.3f} mm")

    milling_path_df["x"] = milling_path_df["x"] + x_offset
    milling_path_df["y"] = milling_path_df["y"] + y_offset
    milling_path = MillingPath(milling_path_df.to_numpy(), feed_curve=FEED_SPEED, offset=OFFSET, offset_side=OFFSET_SIDE)

    workpiece_Area2D = Area2D.load_from_disk(os.path.join("data", "paths", "Workpiece_long_with_start.pickle"))
    boundary_points_vec = workpiece_Area2D.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    workpiece_vertices = np.transpose(boundary_points)
    workpiece = milling_workpiece(workpiece_vertices, AXIAL_CUTTING_DEPTH_MM)

    workpiece.diameter_end_mill = TOOL_DIAMETER_MM
    workpiece.radius_tool = TOOL_RADIUS_MM
    workpiece.number_of_teeth = Z_TEETH
    workpiece.axial_cutting_depth = AXIAL_CUTTING_DEPTH_MM

    # Path generation
    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []
    while pos_on_path <= milling_path.length():
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += FEED_SPEED * DT
    trajectory_local = np.array(trajectory_local)
    pos_on_path_log = np.array(pos_on_path_log)

    omega, fz_mm = spindle_from_rpm_and_feed(
        feed_mm_s=FEED_SPEED,
        rpm=RPM_TARGET,
        z_teeth=Z_TEETH,
        spin_dir=SPINDLE_SPIN
    )
    print(f"Feed speed = {FEED_SPEED:.3f} mm/s")
    print(f"RPM target = {RPM_TARGET:.2f} rev/min")
    print(f"Omega = {omega:.6f} rad/s")
    print(f"Derived fz = {fz_mm:.6f} mm/tooth")

    def print_memory():
        # Python process memory
        mem_proc = process.memory_info().rss / (1024**2)   # MB

        # System RAM
        vm = psutil.virtual_memory()
        mem_used = vm.used / (1024**3)    # GB
        mem_total = vm.total / (1024**3)  # GB
        mem_percent = vm.percent

        print(f"[MEM] Python: {mem_proc:.1f} MB | System: {mem_used:.2f}/{mem_total:.2f} GB ({mem_percent:.1f}%)")

    # Force simulation loop (in WP-frame)
    force_log_xyz = np.zeros((3, len(trajectory_local)))
    t_eval = np.arange(0, DT * len(trajectory_local), DT)
    k = 1e3   # simplify every k steps (choose what you want)

    for i, t in enumerate(t_eval):
    
        spindle_orientation = t * omega
        workpiece.erase_step(trajectory_local[i], spindle_orientation, direction=SPINDLE_SPIN, t=t)

        milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1, 1, 1])
        force_log_xyz[:, i] = milling_wrench_WP[3:6]

        if i % k == 0:
            print_memory() 
            #workpiece.workpiece_slice[0].area.simplify(1e-3, True)
            print_memory()   # <-- memory print

        if i % PRINT_EVERY_N == 0:
            print(f"{t/t_eval[-1] * 100:.2f}%")


    # # Force simulation loop (in WP-frame)
    # force_log_xyz = np.zeros((3, len(trajectory_local)))
    # t_eval = np.arange(0, dt * len(trajectory_local), dt)
    # for i, t in enumerate(t_eval):
    #     spindle_orientation = t * omega
    #     workpiece.erase_step(trajectory_local[i], spindle_orientation, direction=spindle_spin, t=t)
    #     milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1 , 1 , 1])
    #     force_log_xyz[:, i] = milling_wrench_WP[3:6]

    #     if i % 100 == 0:
    #         print(f"{t/t_eval[-1] * 100:.2f}%")

    os.makedirs(FOLDER, exist_ok=True)

    def fmt(v: float) -> str:
        return f"{v:+.4f}".replace(".", "p")

    filename = f"{SIM_NAME}_dx{fmt(x_offset)}_dy{fmt(y_offset)}.csv"
    path = os.path.join(FOLDER, filename)

    df_dict = {}
    df_dict['Time'] = t_eval
    df_dict['Position along path'] = pos_on_path_log
    df_dict['x tool center WP'] = trajectory_local[:, 0]
    df_dict['y tool center WP'] = trajectory_local[:, 1]
    df_dict['x milling force WP'] = force_log_xyz[0, :]
    df_dict['y milling force WP'] = force_log_xyz[1, :]
    df_dict['z milling force WP'] = force_log_xyz[2, :]

    pd_header = df_dict.keys()
    df = pd.DataFrame(df_dict, columns=pd_header)
    df.to_csv(path, index=False)

def milling_model_sim_reproduction(x_base: np.ndarray, y_base: np.ndarray):
    workpiece_Area2D = Area2D.load_from_disk(os.path.join("data", "paths", "Workpiece_long.pickle"))
    boundary_points_vec = workpiece_Area2D.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    workpiece_vertices = np.transpose(boundary_points)
    workpiece = milling_workpiece(workpiece_vertices, axial_cutting_depth=axial_cutting_depth, T_base_workpiece=T_base_workpiece)

    rev_per_min = 60 / (feed_per_tooth * workpiece.number_of_teeth) * feed_speed_rev_comp  # For constant feed per tooth with feed_speed_rev_comp
    omega = spindle_spin * rev_per_min * 2 * np.pi / 60  # [rad/s]

    trajectory_base = np.vstack([1000 * x_base, 1000 * y_base, np.zeros(len(x_base)), np.ones(len(x_base))])
    trajectory_local = T_inv(workpiece.T_base_workpiece) @ trajectory_base

    # Force simulation loop (in WP-frame)
    force_log_xyz = np.zeros((3, len(x_base)))
    t_eval = np.arange(0, dt * len(x_base), dt)
    for i, t in enumerate(t_eval):
        spindle_orientation = t * omega
        workpiece.erase_step(trajectory_local[:2, i], spindle_orientation, direction=spindle_spin, t=t)
        milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1 , 1 , 1])
        force_log_xyz[:, i] = milling_wrench_WP[3:6]

        if i % 100 == 0:
            print(f"{t/t_eval[-1] * 100:.2f}%")

    path = os.path.join(folder, sim_name + ".csv")
    os.makedirs(folder, exist_ok=True)

    df_dict = {}
    df_dict['Time'] = t_eval
    # df_dict['Position along path'] = pos_on_path_log
    df_dict['x tool center WP'] = trajectory_local[0, :]
    df_dict['y tool center WP'] = trajectory_local[1, :]
    df_dict['x milling force WP'] = force_log_xyz[0, :]
    df_dict['y milling force WP'] = force_log_xyz[1, :]
    df_dict['z milling force WP'] = force_log_xyz[2, :]

    pd_header = df_dict.keys()
    df = pd.DataFrame(df_dict, columns=pd_header)
    df.to_csv(path, index=False)

def milling_model_sim_reproduction_feedback(x_base: np.ndarray, y_base: np.ndarray):
    workpiece_Area2D = Area2D.load_from_disk(os.path.join("data", "paths", "Workpiece_long.pickle"))
    boundary_points_vec = workpiece_Area2D.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    workpiece_vertices = np.transpose(boundary_points)
    workpiece = milling_workpiece(workpiece_vertices, axial_cutting_depth=axial_cutting_depth, T_base_workpiece=T_base_workpiece)

    rev_per_min = 60 / (feed_per_tooth * workpiece.number_of_teeth) * feed_speed_rev_comp  # For constant feed per tooth with feed_speed_rev_comp
    omega = spindle_spin * rev_per_min * 2 * np.pi / 60  # [rad/s]

    trajectory_base = np.vstack([1000 * x_base, 1000 * y_base, np.zeros(len(x_base)), np.ones(len(x_base))])
    trajectory_local = T_inv(workpiece.T_base_workpiece) @ trajectory_base

    # Options for feed_optimizer not relevant, since only the IIR filters/TF functions are used.
    milling_tolerance = 0.25    # in [mm]
    tracking_tolerance = 0.05   # in [mm]
    feedrate_min = 1            # in [mm/s]
    feedrate_max = 100          # in [mm/s]
    n_support_wdw = 20          # Number of support points in an optimization window
    downsampling_factor = int(10000 / 200)
    opt_options = {'disp': True, 'maxiter': 1000, 'ftol': 1e-6}     # For 'SLSQP'
    workpiece_name = "Workpiece_long"
    folder_path = os.path.join("data", "paths")
    log_folder = os.path.join("data", "experiments", "Optimization_workpiece_long")

    feed_optimizer = FeedOptimizer(milling_tolerance=milling_tolerance,
                                    tracking_tolerance=tracking_tolerance,
                                    feedrate_bounds=(feedrate_min, feedrate_max),
                                    n_support_wdw=n_support_wdw,
                                    downsampling_factor=downsampling_factor,
                                    opt_options=opt_options,
                                    folder=folder_path,
                                    log_folder=log_folder,
                                    workpiece=workpiece_name)
    
    df_opt_result = pd.read_csv(os.path.join('data', 'experiments', 'Optimization_workpiece_long',"Optimization_13_08_25_6", 'Optimal_feed_curve.csv'))
    feed_curve_i = df_opt_result['feed_curve']
    _, debug_output = feed_optimizer.objective(feed_curve_i.values, 0, extended_output=True)
    feed_optimizer.milling_error_prediction_xy_single(debug_output['fx_dt_conv_scaled'], debug_output['fy_dt_conv_scaled'], debug_output['fz_dt_conv_scaled'], initialization=True) # Initialization of IIR filter states

    # Force simulation loop (in WP-frame)
    force_log_xyz = np.zeros((3, len(x_base)))
    t_eval = np.arange(0, dt * len(x_base), dt)
    milling_error_xi = np.array([0])
    milling_error_yi = np.array([0])
    for i, t in enumerate(t_eval):
        milling_error_wp_i = T_inv(workpiece.T_base_workpiece)[:3, :3] @ np.array([milling_error_xi[0], milling_error_yi[0], 0])
        spindle_orientation = t * omega
        workpiece.erase_step(trajectory_local[:2, i] + milling_error_wp_i[:2], spindle_orientation, direction=spindle_spin, t=t)
        milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1 , 1 , 1])
        force_log_xyz[:, i] = milling_wrench_WP[3:6]

        if i % downsampling_factor == 0:
            # Smooth force simulation over window, since IIR filter model runs at lower frequency than sim
            fx_smoothed = np.average(force_log_xyz[0, np.max([0, i-4*downsampling_factor]):(i+1)])
            fy_smoothed = np.average(force_log_xyz[1, np.max([0, i-4*downsampling_factor]):(i+1)])
            fz_smoothed = np.average(force_log_xyz[2, np.max([0, i-4*downsampling_factor]):(i+1)])
            if i == 0:
                milling_error_xi, milling_error_yi = feed_optimizer.milling_error_prediction_xy_single(fx_smoothed, fy_smoothed, fz_smoothed, initialization=True) # Initialization of IIR filter states
            else:
                milling_error_xi, milling_error_yi = feed_optimizer.milling_error_prediction_xy_single(fx_smoothed, fy_smoothed, fz_smoothed) # Initialization of IIR filter states
            print("fx=", fx_smoothed, ", fy=", fy_smoothed, ", fz=", fz_smoothed)
            print("ex=", milling_error_xi, ", ey=", milling_error_yi)

        if i % 100 == 0:
            print(f"{t/t_eval[-1] * 100:.2f}%")

    path = os.path.join(folder, sim_name + ".csv")
    os.makedirs(folder, exist_ok=True)

    df_dict = {}
    df_dict['Time'] = t_eval
    # df_dict['Position along path'] = pos_on_path_log
    df_dict['x tool center WP'] = trajectory_local[0, :]
    df_dict['y tool center WP'] = trajectory_local[1, :]
    df_dict['x milling force WP'] = force_log_xyz[0, :]
    df_dict['y milling force WP'] = force_log_xyz[1, :]
    df_dict['z milling force WP'] = force_log_xyz[2, :]

    pd_header = df_dict.keys()
    df = pd.DataFrame(df_dict, columns=pd_header)
    df.to_csv(path, index=False)

if __name__ == '__main__':
    chafli_model_preparation(dh=DH_MM)
    # milling_model_sim_reproduction(x_ref_corrected, y_ref_corrected)
    # milling_model_sim_reproduction_feedback(x_ref_corrected, y_ref_corrected)
