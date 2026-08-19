"""Script to run tests to check the full workflow"""

# %% ===== Imports
import pandas as pd
from pathlib import Path
import sys
import importlib

# %% ==== Pathing
ROOT = Path(__file__).resolve.parents[2]
RUN_DIR = ROOT / "run"
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import createPathingJSON, addRawDataPaths

# %%
test_json_name = "test.json"

createPathingJSON(json_name=test_json_name)
addRawDataPaths(
    raw_data_paths=[str(RUN_DIR / "dummy_data.csv")],
    set_names=["test"],
    json_name=test_json_name,
)
