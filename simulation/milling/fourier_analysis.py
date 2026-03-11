from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt


from milling_data import EXPERIMENTS_DIR, load_run_params, resolve_run_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


def get_oscillatory_xy_mm(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    nominal_cols = {"tool_center_nominal_x_mm", "tool_center_nominal_y_mm"}
    actual_cols = {"tool_center_actual_x_mm", "tool_center_actual_y_mm"}
    if nominal_cols.issubset(df.columns) and actual_cols.issubset(df.columns):
        dx = df["tool_center_actual_x_mm"].to_numpy(dtype=float) - df["tool_center_nominal_x_mm"].to_numpy(dtype=float)
        dy = df["tool_center_actual_y_mm"].to_numpy(dtype=float) - df["tool_center_nominal_y_mm"].to_numpy(dtype=float)
        return dx, dy, "actual - nominal"

    if {"tool_center_x_mm", "tool_center_y_mm"}.issubset(df.columns):
        x = df["tool_center_x_mm"].to_numpy(dtype=float)
        y = df["tool_center_y_mm"].to_numpy(dtype=float)
        return x - np.mean(x), y - np.mean(y), "detrended tool_center (mean removed)"

    raise ValueError("Could not find required position columns in experiment_data.csv/path_trace.csv")


def fft_single_sided(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n < 2:
        raise ValueError("Signal must contain at least 2 samples for FFT.")

    window = np.hanning(n)
    signal_w = (signal - np.mean(signal)) * window

    spec = np.fft.rfft(signal_w)
    freq_hz = np.fft.rfftfreq(n, d=1.0 / fs_hz)

    scale = np.sum(window) / 2.0
    amp = np.abs(spec) / max(scale, 1e-12)
    if len(amp) > 0:
        amp[0] *= 0.5
    return freq_hz, amp


def top_peaks(freq_hz: np.ndarray, amp: np.ndarray, n_peaks: int = 5, min_freq_hz: float = 0.0) -> list[tuple[float, float]]:
    mask = freq_hz >= min_freq_hz
    f = freq_hz[mask]
    a = amp[mask]
    if len(a) == 0:
        return []
    idx = np.argsort(a)[::-1][:n_peaks]
    return [(float(f[i]), float(a[i])) for i in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="FFT analysis of oscillatory tool-center motion.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name under data/experiments (default: latest run with experiment_data.csv).",
    )
    parser.add_argument("--max-freq-hz", type=float, default=500.0, help="Upper frequency limit for FFT plot.")
    parser.add_argument("--min-peak-freq-hz", type=float, default=1.0, help="Ignore lower frequencies when listing dominant peaks.")
    parser.add_argument("--show-time", action="store_true", help="Also show the oscillatory displacement in time domain.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive plot window.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save FFT figure.")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run)
    exp_csv = run_dir / "experiment_data.csv"
    if not exp_csv.exists():
        raise FileNotFoundError(f"Missing experiment_data.csv in: {run_dir}")

    df = pd.read_csv(exp_csv)
    if "time_s" not in df.columns:
        raise ValueError(f"{exp_csv} must include 'time_s' column.")

    t = df["time_s"].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError("Not enough time samples in experiment_data.csv")

    dt = float(np.median(np.diff(t)))
    fs_hz = 1.0 / dt

    dx_mm, dy_mm, signal_label = get_oscillatory_xy_mm(df)
    dr_mm = np.hypot(dx_mm, dy_mm)

    fx, ax = fft_single_sided(dx_mm, fs_hz)
    fy, ay = fft_single_sided(dy_mm, fs_hz)
    fr, ar = fft_single_sided(dr_mm, fs_hz)

    params = load_run_params(run_dir, missing_ok=True)
    spindle_hz = abs(float(params.get("rpm", 0.0))) / 60.0 if params else None
    tooth_pass_hz = spindle_hz * float(params.get("z_teeth", 0.0)) if params and "z_teeth" in params else None

    print(f"Run: {run_dir.name}")
    print(f"Samples: {len(t)}")
    print(f"dt = {dt:.6e} s, fs = {fs_hz:.3f} Hz")
    print(f"Signal source: {signal_label}")
    if spindle_hz:
        print(f"Spindle frequency: {spindle_hz:.3f} Hz")
    if tooth_pass_hz:
        print(f"Tooth-pass frequency: {tooth_pass_hz:.3f} Hz")

    print("\nTop peaks (x displacement):")
    for f, a in top_peaks(fx, ax, n_peaks=5, min_freq_hz=args.min_peak_freq_hz):
        print(f"  {f:9.3f} Hz | {a:10.6f} mm")

    print("\nTop peaks (y displacement):")
    for f, a in top_peaks(fy, ay, n_peaks=5, min_freq_hz=args.min_peak_freq_hz):
        print(f"  {f:9.3f} Hz | {a:10.6f} mm")

    print("\nTop peaks (radial displacement):")
    for f, a in top_peaks(fr, ar, n_peaks=5, min_freq_hz=args.min_peak_freq_hz):
        print(f"  {f:9.3f} Hz | {a:10.6f} mm")

    if plt is None:
        print("\nmatplotlib is not installed. Skipping plot generation.")
        return

    fig_cols = 2 if args.show_time else 1
    fig, axes = plt.subplots(3, fig_cols, figsize=(11 if args.show_time else 7, 8), sharex="col")
    if fig_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    row_data = [
        ("x oscillation", dx_mm, fx, ax),
        ("y oscillation", dy_mm, fy, ay),
        ("radial oscillation", dr_mm, fr, ar),
    ]

    for i, (name, sig, freq, amp) in enumerate(row_data):
        if args.show_time:
            axes[i, 0].plot(t, sig, linewidth=1.0)
            axes[i, 0].set_ylabel(f"{name}\n[mm]")
            axes[i, 0].grid(True, alpha=0.25)

        fft_ax = axes[i, 1] if args.show_time else axes[i, 0]
        mask = freq <= args.max_freq_hz
        fft_ax.plot(freq[mask], amp[mask], linewidth=1.0)
        fft_ax.set_ylabel(f"{name}\namp [mm]")
        fft_ax.grid(True, alpha=0.25)

        if spindle_hz:
            fft_ax.axvline(spindle_hz, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.8, label="spindle")
        if tooth_pass_hz:
            fft_ax.axvline(
                tooth_pass_hz,
                color="tab:green",
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label="tooth-pass",
            )

    if args.show_time:
        axes[-1, 0].set_xlabel("time [s]")
        axes[-1, 1].set_xlabel("frequency [Hz]")
    else:
        axes[-1, 0].set_xlabel("frequency [Hz]")

    handles, labels = (axes[0, 1] if args.show_time else axes[0, 0]).get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(f"Fourier analysis: {run_dir.name}")
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        print(f"\nSaved figure to: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
