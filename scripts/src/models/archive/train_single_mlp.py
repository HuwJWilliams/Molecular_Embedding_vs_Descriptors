"""Script to train a single MLP regressor and save outputs."""

from __future__ import annotations

import argparse
import pandas as pd
import sys
from pathlib import Path


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


print("Paths acquired")
data_paths = getPaths()

print("Building arguments")
parser = argparse.ArgumentParser()
parser.add_argument("--predict-on", required=True, choices=list(data_paths["targets"].keys()))
parser.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
parser.add_argument("--target-column", default=None, help="Override the target column name")
parser.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
args = parser.parse_args()

task = args.predict_on
feat = args.feature_set
target_col = resolve_target_column(task, args.target_column)
identifier = args.identifier or f"{feat}_pred_{task}"

print("Resolving paths")
in_features = data_paths["full_features"][task][feat]
in_targets = data_paths["targets"][task]

rf_out_dir = data_paths["prediction_output_dirs"]["rf"][task][feat]
out_dir = Path(str(rf_out_dir).replace("_predictions_rf/", "_predictions_mlp/"))
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[MLP] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Output  : {out_dir}")
print(f"Target column: {target_col}")

X = loadData(in_features, index_col="ID", wildcard="*")
if "SMILES" in X.columns:
    X = X.drop(columns=["SMILES"])

y_df = pd.read_csv(in_targets, index_col="ID")
if target_col not in y_df.columns:
    raise SystemExit(
        f"Target column '{target_col}' not found in {in_targets}. "
        f"Available: {list(y_df.columns)}"
    )

y = y_df[[target_col]]

common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

print(f"Aligned shapes -> X: {X.shape}, y: {y.shape}")

model = TL(log_identifier=identifier)
data = X.join(y)

data = data.sort_index()
cut = int(0.8 * len(data))
train_data = data.iloc[:cut].copy()
test_data = data.iloc[cut:].copy()

print(f"Train: {train_data.shape}  Test: {test_data.shape}")

mlp_model, _, scaler, _, feature_cols = model.trainMLPModel(
    data=train_data,
    target_column=target_col,
    hidden_sizes=(128, 64),
    random_seed=model.rng(),
    save_models=True,
    save_path=out_dir,
    epochs=300,
    learning_rate=1e-3,
    batch_size=256,
    scale_data=True,
    dropout=0.0,
)

model.predictSingleTargetMLP(
    feature_data=test_data,
    target_column=target_col,
    mlp_model=mlp_model,
    feature_cols=feature_cols,
    scaler=scaler,
    test_data=test_data,
    save_preds=True,
    save_path=out_dir,
    filename="last_20pct_pred",
)
