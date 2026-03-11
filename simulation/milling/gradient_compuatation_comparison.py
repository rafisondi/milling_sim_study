import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"

# Keep these defaults as placeholders; pass explicit run names from CLI.
DEFAULT_ORIGIN_RUN = "20260306_211831__8a45c8c78e24__rpm833p3__feed40p000__fs10000"
DEFAULT_DX_PLUS_RUN = ""
DEFAULT_DX_MINUS_RUN = ""
DEFAULT_DY_PLUS_RUN = ""
DEFAULT_DY_MINUS_RUN = ""


def load_run_df(run_folder: str) -> pd.DataFrame:
    exp_dir = EXPERIMENTS_DIR / run_folder
    if not exp_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {exp_dir}")

    exp_csv = exp_dir / "experiment_data.csv"
    if not exp_csv.exists():
        raise FileNotFoundError(f"Missing experiment_data.csv: {exp_csv}")

    df = pd.read_csv(exp_csv)
    required = {"time_s", "tool_orientation", "fmill_x_N", "fmill_y_N"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{exp_csv} missing columns: {sorted(missing)}")

    trace_candidates = sorted(exp_dir.glob("path_trace*.csv"))
    if trace_candidates:
        tr = pd.read_csv(trace_candidates[0])
        if "position_along_path_mm" in tr.columns and len(tr) == len(df):
            df["s_mm"] = tr["position_along_path_mm"].to_numpy(dtype=float)
        elif {"time_s", "position_along_path_mm"}.issubset(tr.columns):
            df["s_mm"] = np.interp(
                df["time_s"].to_numpy(dtype=float),
                tr["time_s"].to_numpy(dtype=float),
                tr["position_along_path_mm"].to_numpy(dtype=float),
            )
        else:
            df["s_mm"] = np.arange(len(df), dtype=float)
    else:
        df["s_mm"] = np.arange(len(df), dtype=float)

    return df


def full_revolution_average(df: pd.DataFrame) -> pd.DataFrame:
    angle = np.unwrap(df["tool_orientation"].to_numpy(dtype=float))
    rev_idx = np.floor((angle - angle[0]) / (2.0 * np.pi)).astype(int)
    rev_idx -= rev_idx.min()

    return (
        df.assign(rev_idx=rev_idx)
        .groupby("rev_idx", as_index=False)
        .agg(
            s_mm=("s_mm", "mean"),
            fx_N=("fmill_x_N", "mean"),
            fy_N=("fmill_y_N", "mean"),
        )
    )


def align_on_common_s(series_by_name: dict[str, pd.DataFrame]) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    processed = {}
    for name, df in series_by_name.items():
        base = df.sort_values("s_mm").groupby("s_mm", as_index=False).mean(numeric_only=True)
        processed[name] = base

    s_min = max(float(v["s_mm"].min()) for v in processed.values())
    s_max = min(float(v["s_mm"].max()) for v in processed.values())
    if s_max <= s_min:
        raise ValueError("No overlapping s-range among selected runs.")

    n = min(len(v) for v in processed.values())
    if n < 3:
        raise ValueError("Not enough overlapping samples to align runs.")

    s_common = np.linspace(s_min, s_max, n)
    aligned = {}
    for name, df in processed.items():
        aligned[name] = pd.DataFrame(
            {
                "s_mm": s_common,
                "fx_N": np.interp(s_common, df["s_mm"].to_numpy(), df["fx_N"].to_numpy()),
                "fy_N": np.interp(s_common, df["s_mm"].to_numpy(), df["fy_N"].to_numpy()),
            }
        )

    return s_common, aligned


def require_nonempty(name: str, value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"Argument --{name.replace('_', '-')} is required for this comparison.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay force traces for old finite-difference trajectories (origin, dx+, dx-, dy+, dy-) "
            "and optional current method trajectories (left/right)."
        )
    )
    parser.add_argument("--origin-run", default=DEFAULT_ORIGIN_RUN, help="Origin run folder name.")
    parser.add_argument("--dx-plus-run", default=DEFAULT_DX_PLUS_RUN, help="Old method run for +dx shift.")
    parser.add_argument("--dx-minus-run", default=DEFAULT_DX_MINUS_RUN, help="Old method run for -dx shift.")
    parser.add_argument("--dy-plus-run", default=DEFAULT_DY_PLUS_RUN, help="Old method run for +dy shift.")
    parser.add_argument("--dy-minus-run", default=DEFAULT_DY_MINUS_RUN, help="Old method run for -dy shift.")
    parser.add_argument("--current-plus-run", default="", help="Optional current method +offset run (e.g. left).")
    parser.add_argument("--current-minus-run", default="", help="Optional current method -offset run (e.g. right).")
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Use raw samples instead of full-revolution average before alignment.",
    )
    parser.add_argument("--save", type=str, default="", help="Optional output path for figure.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive figure window.")
    args = parser.parse_args()

    run_map: dict[str, str] = {
        "origin": require_nonempty("origin_run", args.origin_run),
        "dx+": require_nonempty("dx_plus_run", args.dx_plus_run),
        "dx-": require_nonempty("dx_minus_run", args.dx_minus_run),
        "dy+": require_nonempty("dy_plus_run", args.dy_plus_run),
        "dy-": require_nonempty("dy_minus_run", args.dy_minus_run),
    }

    if args.current_plus_run.strip():
        run_map["current+"] = args.current_plus_run.strip()
    if args.current_minus_run.strip():
        run_map["current-"] = args.current_minus_run.strip()

    curves = {}
    for name, run in run_map.items():
        df = load_run_df(run)
        if args.use_raw:
            curves[name] = df.rename(columns={"fmill_x_N": "fx_N", "fmill_y_N": "fy_N"})[["s_mm", "fx_N", "fy_N"]]
        else:
            curves[name] = full_revolution_average(df)

    s_common, aligned = align_on_common_s(curves)

    styles = {
        "origin": dict(color="k", linewidth=2.0),
        "dx+": dict(color="tab:red", linewidth=1.3),
        "dx-": dict(color="tab:orange", linewidth=1.3),
        "dy+": dict(color="tab:blue", linewidth=1.3),
        "dy-": dict(color="tab:cyan", linewidth=1.3),
        "current+": dict(color="tab:green", linewidth=1.5, linestyle="--"),
        "current-": dict(color="tab:purple", linewidth=1.5, linestyle="--"),
    }

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for name in run_map:
        st = styles.get(name, {})
        axes[0].plot(s_common, aligned[name]["fx_N"].to_numpy(dtype=float), label=name, **st)
        axes[1].plot(s_common, aligned[name]["fy_N"].to_numpy(dtype=float), label=name, **st)

    axes[0].set_ylabel("Fx [N]")
    axes[1].set_ylabel("Fy [N]")
    axes[1].set_xlabel("Position along path s [mm]")

    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(ncol=4)

    title_mode = "raw" if args.use_raw else "full-revolution averaged"
    fig.suptitle(f"Force comparison: old (dx/dy) vs current trajectories ({title_mode})")
    fig.tight_layout()

    if args.save.strip():
        out = Path(args.save)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"Saved figure to: {out}")

    print("Loaded runs:")
    for name, run in run_map.items():
        print(f"  {name:9s}: {run}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
