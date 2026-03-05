from pathlib import Path

current_file = Path(__file__).resolve()
REPO_PATH = current_file.parent.parent
URDF_PATH_V1 = REPO_PATH / 'data' / 'urdf' / 'v1' / 'Staeubli-Huynh.urdf'
URDF_PATH_V2 = REPO_PATH / 'data' / 'urdf' / 'v2' / 'Staeubli-Huynh 1.urdf'

