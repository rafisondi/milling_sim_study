import os
import numpy as np
import pandas as pd
from milling_path import MillingPath
from eraser_of_matter import milling_workpiece
from GeopmetryStandalone import Area2D, T_inv
from feed_optimizer2 import FeedOptimizer
import argparse
import os
import psutil

process = psutil.Process(os.getpid())

dt = 1.0e-3
sim_name = f"Chaflimodel_100mmps_3333rpm_{int(1/dt)}Hz_with_start_TRAJECTORY"
folder = os.path.join("data", "experiments")
feed_speed = 1.0         # [mm/s]
axial_cutting_depth = 5. # [mm]

T_base_workpiece = np.array(
    [[ 0.,  1.,  0., -7.         ],
     [-1.,  0.,  0., -1667.40003 ],
     [ 0.,  0.,  1.,  0.48569812 ],
     [ 0.,  0.,  0.,  1.         ]]
)

spindle_spin = -1
feed_per_tooth = 0.09       # [mm]
feed_speed_rev_comp = 40    # [mm/s]

offset = None
offset_side = 'left'        # 'left' or 'right'


def chafli_model_preparation(x_offset=0.0, y_offset=0.0, sim_name_suffix=""):
    milling_path_df = pd.read_csv(
        os.path.join("data", "paths", "Workpiece_long_with_start_milling_path.csv"),
        index_col=0
    )

    # Apply offsets (in same units as the CSV path file columns)
    milling_path_df["x"] = milling_path_df["x"] + x_offset
    milling_path_df["y"] = milling_path_df["y"] + y_offset

    milling_path = MillingPath(
        milling_path_df.to_numpy(),
        feed_curve=feed_speed,
        offset=offset,
        offset_side=offset_side
    )

    workpiece_Area2D = Area2D.load_from_disk(os.path.join("data", "paths", "Workpiece_long_with_start.pickle"))
    boundary_points_vec = workpiece_Area2D.get_boundary_points()
    boundary_points = np.array([[b.xyz_in_array()[0], b.xyz_in_array()[2]] for b in boundary_points_vec])
    workpiece_vertices = np.transpose(boundary_points)
    workpiece = milling_workpiece(workpiece_vertices, milling_path.axial_cutting_depth)
    print("we are before path generation")
    # Path generation
    pos_on_path = 0.0
    trajectory_local = []
    pos_on_path_log = []
    while pos_on_path <= milling_path.length():
        trajectory_local.append(milling_path.points_at_distances(np.array([pos_on_path]))[:, 0])
        pos_on_path_log.append(pos_on_path)
        pos_on_path += feed_speed * dt
    trajectory_local = np.array(trajectory_local)
    pos_on_path_log = np.array(pos_on_path_log)

    rev_per_min = 60 / (feed_per_tooth * workpiece.number_of_teeth) * feed_speed_rev_comp
    omega = spindle_spin * rev_per_min * 2 * np.pi / 60
    print("omega:", omega)

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
    t_eval = np.arange(0, dt * len(trajectory_local), dt)
    k = 1e3   # simplify every k steps (choose what you want)

    for i, t in enumerate(t_eval):
        print("we are in the loop")
        spindle_orientation = t * omega
        workpiece.erase_step(trajectory_local[i], spindle_orientation, direction=spindle_spin, t=t)

        milling_wrench_WP = workpiece.total_milling_wrench_wpframe * np.array([0.001, 0.001, 0.001, 1, 1, 1])
        force_log_xyz[:, i] = milling_wrench_WP[3:6]

        if i % k == 0:
            workpiece.workpiece_slice[0].area.simplify(1e-3, True)
            print_memory()   # <-- memory print

        if i % 100 == 0:
            print(f"{t/t_eval[-1] * 100:.2f}%")



    os.makedirs(folder, exist_ok=True)

    # Build filename that encodes offsets (avoid '.' in filenames by formatting)
    def fmt(v):
        return f"{v:+.4f}".replace(".", "p")  # +0p0100 etc

    filename = f"{sim_name}{sim_name_suffix}_dx{fmt(x_offset)}_dy{fmt(y_offset)}.csv"
    path = os.path.join(folder, filename)

    df = pd.DataFrame({
        "Time": t_eval,
        "Position along path": pos_on_path_log,
        "x tool center WP": trajectory_local[:, 0],
        "y tool center WP": trajectory_local[:, 1],
        "x milling force WP": force_log_xyz[0, :],
        "y milling force WP": force_log_xyz[1, :],
        "z milling force WP": force_log_xyz[2, :],
    })
    df.to_csv(path, index=False)
    print("Saved:", path)


def run_5_offsets(h):
    # h_vec = [(0,0), (+h,0), (-h,0), (0,+h), (0,-h)]
    offsets = [
        (0.0, 0.0),
        ( h,  0.0),
        (-h,  0.0),
        (0.0,  h),
        (0.0, -h),
    ]
    for dx, dy in offsets:
        chafli_model_preparation(x_offset=dx, y_offset=dy, sim_name_suffix="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.0, help="x offset applied to path")
    parser.add_argument("--dy", type=float, default=0.0, help="y offset applied to path")
    parser.add_argument("--h", type=float, default=None, help="If provided, run 5 offsets with step h")
    args = parser.parse_args()

    if args.h is not None:
        run_5_offsets(args.h)
    else:
        chafli_model_preparation(x_offset=args.dx, y_offset=args.dy, sim_name_suffix="")