from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
SETTINGS_DIR = EXPERIMENTS_DIR / "settings"

RUN_FOLDERS = [
    "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000",
    "20260306_221950__c636af8ee28e__rpm833p3__feed40p000__fs10000",
    "20260306_232207__a443bc2674a4__rpm833p3__feed40p000__fs10000",
]
MAX_WAYPOINT_LABELS = 52
WORKPIECE_WIDTH_Y_MM = 150.0
RAW_PLOT_STRIDE = 1
STEADY_STATE_POINT_WINDOW = (0, 200)
RAW_TIME_WINDOW_S = (0.0, 5.0)


def load_run(run_folder: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    exp_dir = EXPERIMENTS_DIR / run_folder
    csv_path = exp_dir / "experiment_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    settings_hash = run_folder.split("__")[1]
    params_path = SETTINGS_DIR / settings_hash / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    df = pd.read_csv(csv_path)
    required = {"time_s", "tool_orientation", "fmill_x_N", "fmill_y_N"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    trace_candidates = sorted(exp_dir.glob("path_trace*.csv"))
    if not trace_candidates:
        raise FileNotFoundError(f"No path_trace*.csv found in {exp_dir}")
    trace_path = trace_candidates[0]
    df_trace = pd.read_csv(trace_path)
    if "position_along_path_mm" not in df_trace.columns:
        raise ValueError(f"{trace_path} missing column 'position_along_path_mm'")

    if len(df_trace) == len(df):
        s_mm = df_trace["position_along_path_mm"].to_numpy()
    elif "time_s" in df_trace.columns:
        s_mm = np.interp(
            df["time_s"].to_numpy(),
            df_trace["time_s"].to_numpy(),
            df_trace["position_along_path_mm"].to_numpy(),
        )
    else:
        raise ValueError(
            f"{trace_path} length ({len(df_trace)}) differs from experiment_data ({len(df)}), "
            "and no time_s column is available for interpolation."
        )

    df = df.copy()
    df["s_mm"] = s_mm

    return df, df_trace, params


def planned_waypoints_full_with_start(x_offset_mm: float = 0.0, y_offset_mm: float = 0.0) -> np.ndarray:
    path_coordinates = np.array(
        [
            [-20.0, 7.0],
            [0.0, 7.0],
            [20.0, 8.0],
            [30.0, 6.5],
            [40.0, 7.5],
            [65.0, 0.0],
            [80.0, 1.5],
            [85.0, 3.5],
            [92.0, 4.0],
            [98.0, 2.0],
            [105.0, -4.0],
            [105.0, -18.0],
            [95.0, -25.0],
            [70.0, -27.0],
            [40.0, -28.0],
            [40.0, -30.0],
            [45.0, -31.0],
            [85.0, -33.0],
            [93.0, -40.0],
            [95.0, -50.0],
            [95.0, -60.0],
        ],
        dtype=float,
    )
    path_start_coordinates = np.array(
        [
            [5.0, -190.0],
            [5.0, -150.0],
            [5.0, -100.0],
            [5.0, -50.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [3.0, 23.0],
            [-16.0, 30.0],
            [-30.0, 21.0],
            [-30.0, 12.0],
        ],
        dtype=float,
    )

    path_coordinates_mirrored_y = np.array([1.0, -1.0]) * np.flip(path_coordinates, axis=0) - np.array(
        [0.0, WORKPIECE_WIDTH_Y_MM]
    )
    path_coordinates_full_with_start = np.concatenate(
        [path_start_coordinates, path_coordinates, path_coordinates_mirrored_y], axis=0
    )
    path_coordinates_full_with_start[:, 0] += float(x_offset_mm)
    path_coordinates_full_with_start[:, 1] += float(y_offset_mm)
    return path_coordinates_full_with_start


def project_waypoints_to_origin_s(waypoint_xy: np.ndarray, df_trace_origin: pd.DataFrame) -> np.ndarray:
    required = {"tool_center_x_mm", "tool_center_y_mm", "position_along_path_mm"}
    missing = required.difference(df_trace_origin.columns)
    if missing:
        raise ValueError(f"path trace for origin missing columns: {sorted(missing)}")

    trace_xy = df_trace_origin[["tool_center_x_mm", "tool_center_y_mm"]].to_numpy(dtype=float)
    trace_s = df_trace_origin["position_along_path_mm"].to_numpy(dtype=float)
    if len(waypoint_xy) == 0 or len(trace_xy) == 0:
        return np.array([], dtype=float)

    s_i = np.zeros(len(waypoint_xy), dtype=float)
    for i, wp in enumerate(waypoint_xy):
        d2 = np.sum((trace_xy - wp) ** 2, axis=1)
        s_i[i] = trace_s[int(np.argmin(d2))]

    # Keep sequence monotone in case of small nearest-neighbor ambiguities.
    s_i = np.maximum.accumulate(s_i)
    return s_i


def full_revolution_average(df: pd.DataFrame) -> pd.DataFrame:
    angle = np.unwrap(df["tool_orientation"].to_numpy())
    revolutions = (angle - angle[0]) / (2.0 * np.pi)
    rev_idx = np.floor(revolutions).astype(int)
    rev_idx = rev_idx - rev_idx.min()

    binned = (
        df.assign(
            rev_idx=rev_idx,
        )
        .groupby("rev_idx", as_index=False)
        .agg(
            time_s=("time_s", "mean"),
            s_mm=("s_mm", "mean"),
            fx_avg_N=("fmill_x_N", "mean"),
            fy_avg_N=("fmill_y_N", "mean"),
            n_samples=("rev_idx", "size"),
        )
    )
    return binned


def interpolate_to_common_s(revavg_by_variant: dict, variants: tuple[str, ...]) -> tuple[np.ndarray, dict]:
    s_arrays = []
    for variant in variants:
        d = revavg_by_variant[variant].sort_values("s_mm")
        s_arrays.append(d["s_mm"].to_numpy())

    s_equal = True
    s_ref = s_arrays[0]
    for s in s_arrays[1:]:
        if len(s) != len(s_ref) or not np.allclose(s, s_ref, atol=1e-9, rtol=1e-9):
            s_equal = False
            break

    if s_equal:
        aligned = {
            variant: revavg_by_variant[variant].sort_values("s_mm").reset_index(drop=True)
            for variant in variants
        }
        return s_ref, aligned

    s_min = max(np.min(s) for s in s_arrays)
    s_max = min(np.max(s) for s in s_arrays)
    if s_max <= s_min:
        raise ValueError("No overlapping s-range found across variants for interpolation.")

    n_common = min(len(s) for s in s_arrays)
    s_common = np.linspace(s_min, s_max, n_common)

    aligned = {}
    for variant in variants:
        d = revavg_by_variant[variant].sort_values("s_mm")
        d = d.groupby("s_mm", as_index=False).mean(numeric_only=True)
        aligned[variant] = pd.DataFrame(
            {
                "s_mm": s_common,
                "fx_avg_N": np.interp(s_common, d["s_mm"].to_numpy(), d["fx_avg_N"].to_numpy()),
                "fy_avg_N": np.interp(s_common, d["s_mm"].to_numpy(), d["fy_avg_N"].to_numpy()),
            }
        )
    return s_common, aligned


def add_waypoint_axis(
    ax: plt.Axes,
    waypoint_s: np.ndarray,
    waypoint_ids: np.ndarray,
    max_labels: int = MAX_WAYPOINT_LABELS,
):
    for s_i in waypoint_s:
        ax.axvline(s_i, color="k", linewidth=0.7, alpha=0.12)

    if len(waypoint_s) == 0:
        return

    if len(waypoint_s) <= max_labels:
        label_idx = np.arange(len(waypoint_s), dtype=int)
    else:
        label_idx = np.linspace(0, len(waypoint_s) - 1, max_labels, dtype=int)
        label_idx = np.unique(label_idx)

    tick_pos = waypoint_s[label_idx]
    tick_lbl = [f"S{i}" for i in waypoint_ids[label_idx]]

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(tick_pos)
    secax.set_xticklabels(tick_lbl, rotation=90, fontsize=7)
    secax.set_xlabel("Path waypoints S_i")


def main():
    raw_by_variant = {}
    revavg_by_variant = {}
    trace_by_variant = {}
    params_by_variant = {}
    variants_order = ("origin", "left", "right")

    for run_folder in RUN_FOLDERS:
        df, df_trace, params = load_run(run_folder)
        variant = params.get("trajectory_variant", run_folder)
        raw_by_variant[variant] = df
        revavg_by_variant[variant] = full_revolution_average(df)
        trace_by_variant[variant] = df_trace
        params_by_variant[variant] = params
        print(f"Loaded {variant:>6s} | run={run_folder} | revolutions={len(revavg_by_variant[variant])}")

    variants_present = tuple(v for v in variants_order if v in revavg_by_variant)
    if not variants_present:
        raise RuntimeError("None of the expected variants (origin/left/right) were loaded.")

    _, revavg_by_variant_s = interpolate_to_common_s(revavg_by_variant, variants_present)

    if "origin" not in params_by_variant:
        raise RuntimeError("Origin trajectory is required for waypoint anchoring but was not loaded.")

    ref_params = params_by_variant["origin"]
    waypoint_xy = planned_waypoints_full_with_start(
        x_offset_mm=ref_params.get("path_x_offset_mm", 0.0),
        y_offset_mm=ref_params.get("path_y_offset_mm", 0.0),
    )
    waypoint_s = project_waypoints_to_origin_s(waypoint_xy, trace_by_variant["origin"])
    waypoint_ids = np.arange(1, len(waypoint_s) + 1, dtype=int)
    print(
        f"Using {len(waypoint_s)} planned support waypoints (S1..S{len(waypoint_s)}) "
        f"(showing up to {MAX_WAYPOINT_LABELS} S_i labels)."
    )

    t0, t1 = RAW_TIME_WINDOW_S

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = raw_by_variant[variant]
        d = d[(d["time_s"] >= t0) & (d["time_s"] <= t1)].iloc[::RAW_PLOT_STRIDE]
        plt.plot(d["time_s"], d["fmill_x_N"], linewidth=0.8, alpha=0.85, label=f"{variant} raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Fx raw [N]")
    plt.title(f"Raw Fx vs time in [{t0:.1f}, {t1:.1f}] s (every {RAW_PLOT_STRIDE}th sample)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = raw_by_variant[variant]
        d = d[(d["time_s"] >= t0) & (d["time_s"] <= t1)].iloc[::RAW_PLOT_STRIDE]
        plt.plot(d["time_s"], d["fmill_y_N"], linewidth=0.8, alpha=0.85, label=f"{variant} raw")
    plt.xlabel("Time [s]")
    plt.ylabel("Fy raw [N]")
    plt.title(f"Raw Fy vs time in [{t0:.1f}, {t1:.1f}] s (every {RAW_PLOT_STRIDE}th sample)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    i0, i1 = STEADY_STATE_POINT_WINDOW

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = raw_by_variant[variant].iloc[i0 : i1 + 1].iloc[::RAW_PLOT_STRIDE]
        plt.plot(d["s_mm"], d["fmill_x_N"], linewidth=0.85, alpha=0.9, label=f"{variant} raw")
    plt.xlabel("Path position s [mm]")
    plt.ylabel("Fx raw [N]")
    plt.title(f"Raw Fx in point window [{i0}, {i1}]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = raw_by_variant[variant].iloc[i0 : i1 + 1].iloc[::RAW_PLOT_STRIDE]
        plt.plot(d["s_mm"], d["fmill_y_N"], linewidth=0.85, alpha=0.9, label=f"{variant} raw")
    plt.xlabel("Path position s [mm]")
    plt.ylabel("Fy raw [N]")
    plt.title(f"Raw Fy in point window [{i0}, {i1}]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = revavg_by_variant[variant]
        plt.plot(d["time_s"], d["fx_avg_N"], label=f"{variant}")
    plt.xlabel("Time [s]")
    plt.ylabel("Fx avg over full revolution [N]")
    plt.title("Revolution-binned mean Fx vs time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = revavg_by_variant[variant]
        plt.plot(d["time_s"], d["fy_avg_N"], label=f"{variant}")
    plt.xlabel("Time [s]")
    plt.ylabel("Fy avg over full revolution [N]")
    plt.title("Revolution-binned mean Fy vs time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = revavg_by_variant_s[variant]
        plt.plot(d["s_mm"], d["fx_avg_N"], label=f"{variant}")
    add_waypoint_axis(plt.gca(), waypoint_s, waypoint_ids)
    plt.xlabel("Path position s [mm]")
    plt.ylabel("Fx avg over full revolution [N]")
    plt.title("Revolution-binned mean Fx vs path position s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(11, 5))
    for variant in variants_present:
        d = revavg_by_variant_s[variant]
        plt.plot(d["s_mm"], d["fy_avg_N"], label=f"{variant}")
    add_waypoint_axis(plt.gca(), waypoint_s, waypoint_ids)
    plt.xlabel("Path position s [mm]")
    plt.ylabel("Fy avg over full revolution [N]")
    plt.title("Revolution-binned mean Fy vs path position s")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
