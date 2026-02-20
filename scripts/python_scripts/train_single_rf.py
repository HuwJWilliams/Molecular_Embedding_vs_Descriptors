"""Script to train a single Random Forest Regressor and save outputs"""

from __future__ import annotations
import sys
import argparse
import pandas as pd
from pathlib import Path

# --- PalmerChem imports
PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))

from RFRegressor import RFRegressor

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
sys.path.insert(0, str(FILE_DIR.parents[1] / "path"))
from get_paths import getPaths

sys.path.insert(0, str(FILE_DIR.parents[1] / "models"))
from transfer_model import TL

sys.path.insert(0, str(FILE_DIR.parents[1] / "misc"))
from misc_fns import loadData

# ---- Path registry (features, targets, outputs)
print("Paths acquired")
data_paths = getPaths()

# Default target column per task (override with --target-column if needed)
DEFAULT_TARGET = {
    "bp":   "Boiling_Point",
    "logd": "LogD",
    "pka":  "pKa",
    "ld50": "LD50",
    "pic50": "pIC50"
}

# --- Parser
print("Building arguments")
p = argparse.ArgumentParser()
p.add_argument("--predict-on", required=True, choices=list(data_paths["targets"].keys()))
p.add_argument("--feature-set", required=True, choices=["rdkit", "mordred", "chemberta", "molformer"])
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
args = p.parse_args()

task = args.predict_on
feat = args.feature_set
target_col = args.target_column or DEFAULT_TARGET[task]
identifier = args.identifier or f"{task}_{feat}"

# --- Resolve paths
print("Resolving paths")
in_features = data_paths["full_features"][task][feat]
in_targets  = data_paths["targets"][task]
out_dir     = data_paths["prediction_output_dirs"]['rf'][task][feat]
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[RF] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Output  : {out_dir}")
print(f"Target column: {target_col}")

# --- Load data
X = loadData(in_features, index_col="ID", wildcard="*")

# Drop SMILES if present in descriptor tables
if "SMILES" in X.columns:
    X = X.drop(columns=["SMILES"])

y_df = pd.read_csv(in_targets, index_col="ID")

if target_col not in y_df.columns:
    raise SystemExit(f"Target column '{target_col}' not found in {in_targets}. "
                     f"Available: {list(y_df.columns)}")

y = y_df[[target_col]]

# Align indexes
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

print(f"Aligned shapes → X: {X.shape}, y: {y.shape}")

# --- Train single-target RF via TL wrapper
model = TL(log_identifier=identifier)

data = X.join(y)

# deterministic split: first 80% train, last 20% test
data = data.sort_index()
cut = int(0.8 * len(data))
train_data = data.iloc[:cut].copy()
test_data  = data.iloc[cut:].copy()

print(f"Train: {train_data.shape}  Test: {test_data.shape}")

model.trainSingleTargetRFModel(
    data=train_data,
    target_column=target_col,
    hyper_params={
        "n_estimators": [400, 500],
        "max_features": ["sqrt"],
        "max_depth": [25, 50],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [2, 4],
    },
    n_resamples=50,
    test_size=0.3,
    save_models=True,
    save_path=out_dir / identifier,
    random_seed=model.rng(),
)
