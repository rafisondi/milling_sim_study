import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DIR = Path("data/experiments")
SETTINGS_DIR = EXPERIMENT_DIR / "settings"


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
    print(f"\n + {df.head()}")
    if "time_s" not in df.columns:
        raise RuntimeError("CSV missing column 'time_s'")

    t = df["time_s"]
    plt.figure()
    if "fmill_x_N" in df:
        plt.plot(t, df["fmill_x_N"], label="Fx sim")
    if "fmill_y_N" in df:
        plt.plot(t, df["fmill_y_N"], label="Fy sim")
    if "fanalyt_x_N" in df:
        plt.plot(t, df["fanalyt_x_N"], "--", label="Fx analytical")
    if "fanalyt_y_N" in df:
        plt.plot(t, df["fanalyt_y_N"], "--", label="Fy analytical")

    plt.xlabel("time [s]")
    plt.ylabel("force [N]")
    plt.title(f"Milling forces ({args.run_id})")
    plt.grid(True)
    plt.legend()

    
    
    if "tool_orientation" in df:
        plt.figure()
        plt.plot(t, df["tool_orientation"] % (2* np.pi), label="Fx sim")
        plt.xlabel("Time [s]")
        plt.ylabel("Orientation [rad]")
        plt.ylim(0, 2*np.pi)

        plt.yticks(
        [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ["0", "π/2", "π", "3π/2", "2π"]
        )
        
        plt.title(f"Spindle postition({args.run_id})")
        plt.grid(True)
        plt.legend()


    plt.show()


if __name__ == "__main__":
    main()