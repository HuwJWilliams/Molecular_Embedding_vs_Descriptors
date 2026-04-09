# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

sys.path.insert(0, "users/yhb18174/TL_project/scripts/config/pathing")
from get_paths import getPaths

paths=getPaths()

test_dir=paths["imp_dirs"]["results_dir"] / "test_dir"
model_path = test_dir / "training_data" / "final_model.pkl"
training_features = test_dir / "training_data" / "training_features.csv.gz"
output_dir = test_dir / "shap_test"
# endregion


# region Asset loading

