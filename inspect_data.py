import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path("data/experiments")
SETTINGS_DIR = EXPERIMENT_DIR / "settings"

def build_experiment_description(params, run_id):
    rpm = params.get("rpm", None)
    feed = params.get("feed_speed_mm_s", None)
    dt = params.get("dt", None)
    ap = params.get("axial_cutting_depth_mm", None)
    D = params.get("tool_diameter_mm", None)
    z = params.get("z_teeth", None)
    fz = params.get("feed_per_tooth_mm", None)

    fs = None
    if dt is not None and dt > 0:
        fs = 1.0 / dt

    parts = [f"run {run_id}"]

    if rpm is not None:
        parts.append(f"rpm={rpm:.1f}")
    if z is not None:
        parts.append(f"z={z}")
    if D is not None:
        parts.append(f"D={D:.1f} mm")
    if ap is not None:
        parts.append(f"ap={ap:.1f} mm")
    if feed is not None:
        parts.append(f"vf={feed:.2f} mm/s")
    if fz is not None:
        parts.append(f"fz={fz:.4f} mm/tooth")
    if fs is not None:
        parts.append(f"fs={fs:.0f} Hz")

    return " | ".join(parts)


def compute_revolution_average_from_angle(t, angle_rad, signal_2d):
    t = np.asarray(t).reshape(-1)
    angle_rad = np.asarray(angle_rad).reshape(-1)
    signal_2d = np.asarray(signal_2d)

    if signal_2d.ndim != 2:
        raise ValueError(f"signal_2d must be 2D, got shape {signal_2d.shape}")
    if len(t) != len(angle_rad) or len(t) != signal_2d.shape[0]:
        raise ValueError("t, angle_rad, and signal_2d must have matching first dimension")

    # unwrap so revolution count is monotone
    angle_unwrapped = np.unwrap(angle_rad)

    # revolution number for each sample
    rev_index = np.floor((angle_unwrapped - angle_unwrapped[0]) / (2.0 * np.pi)).astype(int)

    signal_revavg = np.full_like(signal_2d, np.nan, dtype=float)

    for rev in np.unique(rev_index):
        mask = rev_index == rev
        if np.any(mask):
            signal_revavg[mask, :] = np.mean(signal_2d[mask, :], axis=0)

    return signal_revavg, rev_index


def list_experiments():
    print("\nAvailable experiments:\n")

    for p in sorted(EXPERIMENT_DIR.iterdir()):
        if p.is_dir() and p.name != "settings":
            print(p.name)


def load_experiment(run_id: str):
    matches = [p for p in EXPERIMENT_DIR.iterdir() if p.is_dir() and run_id in p.name]

    if not matches:
        raise RuntimeError(f"No experiment found for run_id '{run_id}'")

    if len(matches) > 1:
        print("Multiple matches:")
        for m in matches:
            print(" ", m.name)
        raise RuntimeError("run_id not unique")

    exp_dir = matches[0]

    # extract hash from folder name
    parts = exp_dir.name.split("__")
    settings_hash = parts[1]

    csv_path = exp_dir / "experiment_data.csv"
    params_path = SETTINGS_DIR / settings_hash / "params.json"

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if not params_path.exists():
        raise FileNotFoundError(params_path)

    df = pd.read_csv(csv_path)

    with open(params_path) as f:
        params = json.load(f)

    return df, params, exp_dir


def main():

    parser = argparse.ArgumentParser(description="Experiment loader")
    parser.add_argument(
        "--run-id",
        help="Run id (timestamp part of experiment folder)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List experiments",
    )

    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if args.run_id is None:
        parser.error("Provide --run-id or --list")

    df, params, exp_dir = load_experiment(args.run_id)

    print("\nLoaded experiment:")
    print(exp_dir)

    print("\nParameters:")
    for k, v in params.items():
        print(f"{k:20s} : {v}")

    # ---- force plot ----
    print(f"\n{df.head()}")

    if "time_s" not in df.columns:
        raise RuntimeError("CSV missing column 'time_s'")

    t = df["time_s"].to_numpy()
    desc = build_experiment_description(params, args.run_id)

    # ---------------------------------
    # Force plot
    # ---------------------------------
    plt.figure(figsize=(11, 6))

    if "fmill_x_N" in df.columns:
        plt.plot(t, df["fmill_x_N"], label="Fx simulated", alpha=0.8)
    if "fmill_y_N" in df.columns:
        plt.plot(t, df["fmill_y_N"], label="Fy simulated", alpha=0.8)

    if "fanalyt_x_N" in df.columns:
        plt.plot(t, df["fanalyt_x_N"], "--", label="Fx analytical", linewidth=1.5)
    if "fanalyt_y_N" in df.columns:
        plt.plot(t, df["fanalyt_y_N"], "--", label="Fy analytical", linewidth=1.5)

    if "fzero_x_N" in df.columns:
        plt.plot(t, df["fzero_x_N"], ":", label="Fx zero-order", linewidth=2.0)
    if "fzero_y_N" in df.columns:
        plt.plot(t, df["fzero_y_N"], ":", label="Fy zero-order", linewidth=2.0)

    # exact per-revolution averaging using spindle angle
    if "tool_orientation" in df.columns and "fmill_x_N" in df.columns and "fmill_y_N" in df.columns:
        angle = df["tool_orientation"].to_numpy()
        fmill = np.column_stack([df["fmill_x_N"].to_numpy(), df["fmill_y_N"].to_numpy()])
        fmill_revavg, rev_idx = compute_revolution_average_from_angle(t, angle, fmill)

        plt.plot(t, fmill_revavg[:, 0], "-.", linewidth=2.0, label="Fx simulated (avg/rev)")
        plt.plot(t, fmill_revavg[:, 1], "-.", linewidth=2.0, label="Fy simulated (avg/rev)")

    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Milling force signals\n" + desc)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # # ---------------------------------
    # # Tool orientation plot
    # # ---------------------------------
    # if "tool_orientation" in df.columns:
    #     angle = df["tool_orientation"].to_numpy()
    #     angle_wrapped = np.mod(angle, 2 * np.pi)
    #     angle_unwrapped = np.unwrap(angle)
    #     revolutions = (angle_unwrapped - angle_unwrapped[0]) / (2 * np.pi)

    #     fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    #     ax[0].plot(t, angle_wrapped, label="wrapped spindle angle")
    #     ax[0].set_ylabel("Angle [rad]")
    #     ax[0].set_ylim(0, 2 * np.pi)
    #     ax[0].set_yticks(
    #         [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
    #         ["0", "π/2", "π", "3π/2", "2π"]
    #     )
    #     ax[0].set_title("Tool / spindle orientation\n" + desc)
    #     ax[0].grid(True)
    #     ax[0].legend()

    #     ax[1].plot(t, revolutions, label="revolution count")
    #     ax[1].set_xlabel("Time [s]")
    #     ax[1].set_ylabel("Revolutions [-]")
    #     ax[1].grid(True)
    #     ax[1].legend()

    #     plt.tight_layout()   

    plt.show()
if __name__ == "__main__":
    main()