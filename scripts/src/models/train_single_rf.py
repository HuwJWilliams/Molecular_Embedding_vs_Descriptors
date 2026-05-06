"""Script to train a single Random Forest Regressor and save outputs"""

from __future__ import annotations
import sys
import argparse
import pandas as pd
from pathlib import Path

# --- Paths
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SCRIPTS_DIR / "src"
sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData
sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import SUPPORTED_FEATURE_SETS, resolve_target_column

# ---- Path registry (features, targets, outputs)
print("Paths acquired")
data_paths = getPaths()

# --- Parser
print("Building arguments")
p = argparse.ArgumentParser()
p.add_argument("--predict-on", required=True, choices=list(data_paths["targets"].keys()))
p.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
p.add_argument("--lipinski", action="store_true", help="Flag to ensure mols fit 'relaxed' lipinski criteria")
args = p.parse_args()

task = args.predict_on
feat = args.feature_set
target_col = resolve_target_column(task, args.target_column)
identifier = args.identifier or f"{task}_{feat}"
train_split_path = data_paths["train_test_splits"][task]["train_models"]
valid_split_path = data_paths["train_test_splits"][task]["validate_models"]

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

if args.lipinski:
    lipinski_ids = set(
        pd.read_csv(
            data_paths["full_features"]["fit_lipinski"]["rdkit"],
            index_col=0,
        ).index
    )

    X = X.loc[X.index.intersection(lipinski_ids)]

# Drop metadata and any descriptor-generation error strings before RF fitting.
X = X.drop(columns=["SMILES", "SELFIES"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.dropna(axis=1)

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

if train_split_path.exists() and valid_split_path.exists():
    print(f"Loading predefined train split from: {train_split_path}")
    print(f"Loading predefined validation split from: {valid_split_path}")

    train_ids = pd.read_csv(train_split_path)["ID"].tolist()
    valid_ids = pd.read_csv(valid_split_path)["ID"].tolist()

    train_ids = [idx for idx in train_ids if idx in data.index]
    valid_ids = [idx for idx in valid_ids if idx in data.index]

    if not train_ids or not valid_ids:
        raise SystemExit(
            "Saved training/validation split IDs do not match the current dataset. "
            "Delete the saved split files to regenerate them."
        )

    train_data = data.loc[train_ids].copy()
    test_data = data.loc[valid_ids].copy()
else:
    print("No predefined split found. Creating random 80/20 split and saving it.")
    train_split_path.parent.mkdir(parents=True, exist_ok=True)

    shuffled_data = data.sample(frac=1.0, random_state=42)
    cut = int(0.8 * len(shuffled_data))
    train_data = shuffled_data.iloc[:cut].copy()
    test_data = shuffled_data.iloc[cut:].copy()

    pd.DataFrame({"ID": train_data.index}).to_csv(train_split_path, index=False)
    pd.DataFrame({"ID": test_data.index}).to_csv(valid_split_path, index=False)

    print(f"Saved training split to: {train_split_path}")
    print(f"Saved validation split to: {valid_split_path}")

print(f"Train: {train_data.shape}  Test: {test_data.shape}")

final_model, _, _, _ = model.trainSingleTargetRFModel(
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
                                save_path=out_dir,
                            )

model.predictSingleTargetRF(
    model=final_model,
    data = test_data,
    target_column = target_col,
    calc_perf=True,
    save_preds=True,
    save_path=out_dir,
    preds_filename="last_20pct_pred",
    perf_filename="last_20_pct_perf"
)
