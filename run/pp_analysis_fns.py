"""
Functions to analyse the individual property prediction (PP) results
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json

# %% ===== Project Imports & Pathing Setup =====
from config import PATHING_JSON_PATH, SRC_DIR, PP_ANALYSIS_METRICS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "visualise"))
from visualise import Visualise

v = Visualise(save_all=False)


# %% ===== Function Definitions =====
def getPropertyPerformanceDfs(
    property_pathing: dict[str, Path],
    feature_sets: list[str],
    perf_fname: str = "rf_performance.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read internal and external mean/std performance from rf_performance.json
    files for each feature set.

    Returns
    -------
    int_property_performance_df:
        Rows for internal mean/std performance.

    ext_property_performance_df:
        Rows for external mean/std performance.
    """

    int_rows = []
    ext_rows = []

    for feat in feature_sets:
        feature_path = Path(property_pathing[feat])
        perf_json_path = feature_path / perf_fname

        with open(perf_json_path, "r") as f:
            perf_json = json.load(f)

        internal = perf_json["internal"]
        int_mean = internal["mean"]
        int_std = internal["std"]

        external = perf_json["external"]
        ext_mean = external["mean"]
        ext_std = external["std"]

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **int_mean,
            }
        )

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **int_std,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **ext_mean,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **ext_std,
            }
        )

    int_property_performance_df = pd.DataFrame(int_rows)
    ext_property_performance_df = pd.DataFrame(ext_rows)

    return int_property_performance_df, ext_property_performance_df
