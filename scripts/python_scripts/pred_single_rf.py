#"""Script to predict with a single Random Forest Regressor and save outputs"""
# %%
# import sys
# import pandas as pd
# from pathlib import Path
# from models.transfer_model import TL

# PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
# sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
# from RFRegressor import RFRegressor

# # --- Arguments
# # TRAIN_NAME = sys.argv[1]
# # print(TRAIN_NAME)

# # TEST_NAME = sys.argv[2]
# # print(TEST_NAME)

# IDENTIFIER = "rdkit_test"
# print(IDENTIFIER)

# TARGET_COLUMN = "LD50"
# print(TARGET_COLUMN)

# # --- Paths
# FILE_DIR = Path(__file__).resolve()
# PROJ_DIR = FILE_DIR.parents[1]

# RDKIT_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"rdkit_descriptors.csv")
# MORDRED_DESC_PATH = Path(PROJ_DIR / "datasets"/"descriptors"/"mordred_descriptors.csv")
# CHEMBERTA_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"chemberta_embeddings.csv")
# MOLFORMER_EMB_PATH = Path(PROJ_DIR / "datasets"/"embeddings"/"molformer_embeddings.csv")
# LD50_TARGETS_PATH = Path(PROJ_DIR / "datasets" / "LD50" / "LD50_targets_rat-mouse-oral_lt9500.csv")
# BP_TARGETS_PATH = Path(PROJ_DIR / "datasets" / "boiling_point" / "boiling_point.csv")


# # --- Load datasets
# datasets = {
#     "rdkit":     pd.read_csv(RDKIT_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").iloc[5000:],
#     "mordred":   pd.read_csv(MORDRED_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore").iloc[5000:],
#     "chemberta": pd.read_csv(CHEMBERTA_EMB_PATH, index_col="ID").iloc[5000:],
#     "molformer": pd.read_csv(MOLFORMER_EMB_PATH, index_col="ID").iloc[5000:],
#     "ld50":      pd.read_csv(LD50_TARGETS_PATH, index_col="ID").iloc[5000:]
# }

# data = "molformer"
# feats_pred = datasets[data]
# targs_true = datasets["ld50"]
# # %%

# common_idx = feats_pred.index.intersection(targs_true.index)

# feats_pred = feats_pred.loc[common_idx]
# targs_true = targs_true.loc[common_idx]

# path = Path(f"/users/yhb18174/TL_project/results/LD50_predictions_rf/{data}_tr_ld50_pred")


# model = TL(tokeniser=None, encoder=None, log_identifier=IDENTIFIER)
# pred, true, perf_dict = model.predictSingleTargetRF(
#     model= path / "LD50_RF_model.pkl",
#     feature_data=feats_pred,
#     calc_perf=True,
#     targets_true=targs_true,
#     save_preds=True,
#     save_path=path,
#     preds_filename="LD50_predictions.csv",
#     perf_filename="LD50_prediction_performance.json"
#     )
# # %%

# import matplotlib.pyplot as plt

# # Example data
# x = pred
# y = true

# plt.figure(figsize=(6,4))
# plt.scatter(x, y, color="blue", alpha=0.7, edgecolors="k")
# plt.xlabel("Pred values")
# plt.ylabel("True values")
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.show()


"""Script to predict with a single Random Forest Regressor and save outputs"""

from __future__ import annotations
import sys
import argparse
import pandas as pd
from pathlib import Path

# --- PalmerChem imports
PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

sys.path.insert(0, "./path")
from get_paths import getPaths

sys.path.insert(0, "./models")
from transfer_model import TL

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[1]

sys.path.insert(0, str(PROJ_DIR / "scripts" / "misc"))
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
}



# --- Parser
print("Building arguments")
p = argparse.ArgumentParser()
p.add_argument("--predict-on", required=True, choices=list(data_paths["targets"].keys()))
p.add_argument("--feature-set", required=True, choices=["rdkit", "mordred", "chemberta", "molformer"])
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")

# Where to load the trained model from
# If not provided, we look in the RF output dir for this task/feat/identifier and use <target>_RF_model.pkl
p.add_argument("--model-path", default=None, help="Path to trained RF .pkl (optional override)")
p.add_argument("--preds-filename", default=None, help="Preds CSV filename (default auto)")
p.add_argument("--perf-filename", default=None, help="Perf JSON filename (default auto)")
args = p.parse_args()

task = args.predict_on
feat = args.feature_set
target_col = args.target_column or DEFAULT_TARGET[task]
identifier = args.identifier or f"{task}_{feat}"

# --- Resolve paths (match training script)
print("Resolving paths")
in_features = data_paths["aligned_features"][task][feat]
in_targets  = data_paths["targets"][task]
out_dir     = data_paths["prediction_output_dirs"]["rf"][task][feat]
out_dir.mkdir(parents=True, exist_ok=True)

run_dir = out_dir / identifier
run_dir.mkdir(parents=True, exist_ok=True)

print(f"[RF-PRED] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Run dir : {run_dir}")
print(f"Target column: {target_col}")

# --- Load data
X = loadData(in_features, index_col="ID", wildcard="*")

# Drop SMILES if present in descriptor tables
if "SMILES" in X.columns:
    X = X.drop(columns=["SMILES"])

y_df = pd.read_csv(in_targets, index_col="ID")
if target_col not in y_df.columns:
    raise SystemExit(
        f"Target column '{target_col}' not found in {in_targets}. "
        f"Available: {list(y_df.columns)}"
    )
y_true = y_df[[target_col]]

# Align indexes
# Align indexes
common = X.index.intersection(y_df.index)
X = X.loc[common]
y_df = y_df.loc[common]

print(f"Aligned shapes → X: {X.shape}, y: {y_df.shape}")

# --- Deterministic split: last 20% only (same as training)
data = X.join(y_df[[target_col]])
data = data.sort_index()

test_data = data.tail(int(0.2 * len(data)))

X_test = test_data.drop(columns=[target_col])
y_test = test_data[[target_col]]

print(f"Predicting on test split → X_test: {X_test.shape}, y_test: {y_test.shape}")

# --- Model path
default_model_path = run_dir / f"{target_col}_RF_model.pkl"
model_path = Path(args.model_path) if args.model_path else default_model_path

if not model_path.exists():
    raise SystemExit(
        f"RF model not found: {model_path}\n"
        f"Tip: pass --model-path to override, or ensure training saved the model into {run_dir}"
    )

preds_filename = args.preds_filename or f"{target_col}_predictions"
perf_filename  = args.perf_filename  or f"{target_col}_prediction_performance"

print(f"Loading model: {model_path}")
print(f"Saving preds:  {run_dir / preds_filename}")
print(f"Saving perf:   {run_dir / perf_filename}")

# --- Predict
model = TL(log_identifier=identifier)

pred, true, perf_dict = model.predictSingleTargetRF(
    model=model_path,
    feature_data=X_test,
    calc_perf=True,
    targets_true=y_test,
    prediction_col=target_col,
    save_preds=True,
    save_path=run_dir,
    preds_filename=preds_filename,
    perf_filename=perf_filename,
)

print("Done.")
print("Performance summary:", perf_dict)
