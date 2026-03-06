import json
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

def hash_dict(d: dict, length: int = 12) -> str:
    """
    Create a stable hash of a dictionary.

    Handles numpy types and ensures deterministic ordering.
    """

    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        else:
            return obj

    canonical = convert(d)

    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":")
    ).encode()

    h = hashlib.sha1(encoded).hexdigest()

    return h[:length]



def save_params(params: dict, base_dir="experiments"):
    exp_id = hash_dict(params)

    exp_dir = Path(base_dir) / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    filepath = exp_dir / "params.json"

    with open(filepath, "w") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    return exp_id, exp_dir


def save_experiment_csv(
    params: dict,
    settings_hash: str,
    output_dir="data/experiments",
    t_hist=None,
    x_hist=None,
    u_hist=None,
    fmill_hist=None,
    fanalyt_hist=None,
    fzero_hist=None,
    tool_center_nominal_hist_mm=None,
    tool_center_actual_hist_mm=None,
    tool_orientation_hist = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    rpm = float(params["rpm"])
    feed = float(params["feed_speed_mm_s"])
    fs_hz = 1.0 / float(params["dt"])

    folder_name = (
        f"{run_id}"
        f"__{settings_hash}"
        f"__rpm{rpm:.1f}"
        f"__feed{feed:.3f}"
        f"__fs{fs_hz:.0f}"
    ).replace(".", "p")

    exp_dir = output_dir / folder_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    cols = {}

    if t_hist is not None:
        cols["time_s"] = np.asarray(t_hist).reshape(-1)

    def add_2d_array(arr, base_name, suffixes=None):
        if arr is None:
            return
        arr = np.asarray(arr)
        if arr.ndim == 1:
            cols[base_name] = arr
        elif arr.ndim == 2:
            if suffixes is None:
                suffixes = [f"_{i}" for i in range(arr.shape[1])]
            for i in range(arr.shape[1]):
                cols[f"{base_name}{suffixes[i]}"] = arr[:, i]
        else:
            raise ValueError(f"{base_name} must be 1D or 2D, got shape {arr.shape}")

    add_2d_array(x_hist, "x", suffixes=["_x_m", "_y_m"])
    add_2d_array(u_hist, "u", suffixes=["_x_N", "_y_N"])
    add_2d_array(fmill_hist, "fmill", suffixes=["_x_N", "_y_N"])
    add_2d_array(fanalyt_hist, "fanalyt", suffixes=["_x_N", "_y_N"])
    add_2d_array(fzero_hist, "fzero", suffixes=["_x_N", "_y_N"])
    add_2d_array(tool_center_nominal_hist_mm, "tool_center_nominal", suffixes=["_x_mm", "_y_mm"])
    add_2d_array(tool_center_actual_hist_mm, "tool_center_actual", suffixes=["_x_mm", "_y_mm"])
    
    cols["tool_orientation"] = np.asarray(tool_orientation_hist).reshape(-1)

    lengths = [len(v) for v in cols.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Not all arrays have the same length: {set(lengths)}")

    header = ",".join(cols.keys())
    data = np.column_stack([cols[k] for k in cols.keys()])

    csv_path = exp_dir / "experiment_data.csv"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")


    return exp_dir, csv_path, run_id


def _to_serializable(obj):
    """Convert numpy objects into plain Python types for logging."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def write_to_experimental_log(
    params: dict,
    settings_hash: str,
    run_id: str,
    exp_dir=None,
    csv_path=None,
    params_dir=None,
    log_path="data/experimental_log.txt",
    status="SUCCESS",
):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    params_clean = _to_serializable(params)

    lines = []
    lines.append("=" * 100)
    lines.append(f"log_timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"status          : {status}")
    lines.append(f"run_id          : {run_id}")
    lines.append(f"settings_hash   : {settings_hash}")

    if exp_dir is not None:
        lines.append(f"experiment_dir  : {Path(exp_dir)}")
    if csv_path is not None:
        lines.append(f"csv_path        : {Path(csv_path)}")
    if params_dir is not None:
        lines.append(f"params_dir      : {Path(params_dir)}")

    lines.append("params:")
    for key in sorted(params_clean.keys()):
        lines.append(f"  {key}: {params_clean[key]}")
    lines.append("")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")