import numpy as np
import pandas as pd
from pathlib import Path
import json


RUN = "20260312_105807__e01766af7b7e__rpm3333p3__feed40p000__fs10000"
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
SETTINGS_DIR = EXPERIMENTS_DIR / "settings"
WORKPIECE_WIDTH_Y_MM = 150.0


def load_params(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    return

def resolve_run_paths(run_name: str | Path, role_hint: str | None = None) -> tuple[Path, Path, Path | None, Path]:
    run_dir = resolve_run_dir(run_name)
    exp_csv = run_dir / "experiment_data.csv"
    if not exp_csv.exists():
        raise FileNotFoundError(f"Missing experiment_data.csv in run directory: {run_dir}")

    trace_candidates: list[Path] = []
    if role_hint:
        trace_candidates = sorted(run_dir.glob(f"path_trace_{role_hint}.csv"))
    if not trace_candidates:
        trace_candidates = sorted(run_dir.glob("path_trace*.csv"))
    trace_csv = trace_candidates[0] if trace_candidates else None

    settings_hash = run_dir.name.split("__")[1] if "__" in run_dir.name else ""
    params_json = SETTINGS_DIR / settings_hash / "params.json"
    return run_dir, exp_csv, trace_csv, params_json

def resolve_run_dir(run_name: str | Path | None) -> Path:
    if run_name is None:
        return find_latest_run_dir()

    run_dir = Path(run_name)
    if not run_dir.is_absolute():
        run_dir = EXPERIMENTS_DIR / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir




