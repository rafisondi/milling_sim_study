# Milling study for linearized robot with mechanistic force model

## Inspection scripts (root)

### `inspect_data.py`
Inspect force and state data from one saved experiment.

Parser options:
- `--list`: list available experiment folders in `data/experiments`.
- `--run-id <RUN_ID>`: load one run (timestamp substring from folder name, must match uniquely).

Examples:
- `python inspect_data.py --list`
- `python inspect_data.py --run-id 20260308_190829`

### `inspect_trajectory.py`
Inspect trajectory and force signals from one saved experiment.

Parser options:
- `--list`: list available experiment folders in `data/experiments`.
- `--run-id <RUN_ID>`: load one run (timestamp substring from folder name, must match uniquely).
- `--plot-realized-tool-history`: overlay realized tool XY (`tool_center_actual_x_mm`, `tool_center_actual_y_mm`) on top of nominal trajectory.

Examples:
- `python inspect_trajectory.py --list`
- `python inspect_trajectory.py --run-id 20260308_190829`
- `python inspect_trajectory.py --run-id 20260308_190829 --plot-realized-tool-history`
