"""Script to train a Linear Regression models and save outputs"""

import sys
import argparse
import pandas as pd
from pathlib import Path

# --- Imports
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR / "path"))
from get_paths import getPaths

sys.path.insert(0, str(SCRIPTS_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from misc_fns import loadData

data_paths = getPaths()
print("Paths acquired")

# --- Constants
DEFAULT_TARGET = {
    "bp":   "Boiling_Point",
    "logd": "LogD",
    "pka":  "pKa",
    "ld50": "LD50",
    "pic50": "pIC50",
}

# --- Building Parser and Arguments
print("Building parser/arguments")
p = argparse.ArgumentParser()
p.add_argument("--predict-on", required=True, choices=data_paths["targets"].keys())
p.add_argument("--feature-set", required=True, choices=["rdkit", "mordred", "chemberta", "molformer"])
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Log file suffix (optional)")
args = p.parse_args()

# --- Creating Variables

task = args.predict_on
feat = args.feature_set
target_col = args.target_column or DEFAULT_TARGET[task]
identifier = args.identifier or f"{task}_{feat}"

# --- Resolving Paths
in_features   = data_paths["aligned_features"][task][feat]
in_targets    = data_paths["targets"][task]
out_dir       = data_paths['prediction_output_dirs']['lr'][task][feat]
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[LR] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Output  : {out_dir}")
print(f"Target column: {target_col}")

# -- Load Data
X = loadData(in_features, index_col="ID", wildcard="*")

# --- Drop SMILES is present in descriptor tables
if "SMILES" in X.columns:
    X = X.drop(columns=["SMILES"])

y_df = pd.read_csv(in_targets, index_col="ID")

if target_col not in y_df.columns:
    raise SystemExit(f"Target column '{target_col}' not found in {in_targets}. "
                     f"Available: {list(y_df.columns)}")

y = y_df[[target_col]]

# --- Align indexes
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]
print(f"Aligned shapes → X: {X.shape}, y: {y.shape}")


# --- Train Linear Refression Model
model = TL(log_identifier=identifier)

data = X.join(y)

# --- Deterministic Split: first 80% train, last 20% test
data = data.sort_index()
cut = int(0.8 * len(data))
train_data = data.iloc[:cut].copy()
test_data = data.iloc[cut:].copy()

print(f"Train: {train_data.shape}  Test: {test_data.shape}")

lr_model, params, scaler, f_test, feat_cols = model.trainSingleLRModel(
    data=train_data,
    target_column=target_col,
    scale_data=True,
    save_models=True,
    save_path=out_dir
)

model.predictSingleTargetLR(
    feature_data = test_data,
    target_column = target_col,
    test_data = test_data,
    lr_model = lr_model,
    feature_cols = feat_cols,
    save_preds = True,
    save_path = out_dir,
    filename = "all_preds",
    scaler = scaler
)
