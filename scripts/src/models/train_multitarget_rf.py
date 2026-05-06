"""Script to train a Multitarget Random Forest Regressors i.e., training to predict each column in target_df"""

import sys
import pandas as pd
from pathlib import Path

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
SCRIPTS_DIR = FILE_DIR.parents[2]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL
sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths
sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import SUPPORTED_FEATURE_SETS

paths = getPaths()
feature_paths = paths["full_features"]
full_feats = feature_paths["all"]

# --- Arguments
TRAIN_NAME = sys.argv[1]
TEST_NAME = sys.argv[2]
IDENTIFIER = sys.argv[3]
CROSS_EMBEDDING_DIR = sys.argv[4]

print("\nInput Parameters\n====================")
print(TRAIN_NAME)
print(TEST_NAME)
print(IDENTIFIER)
DATASET_NAMES = SUPPORTED_FEATURE_SETS


def load_feature_dataset(name: str) -> pd.DataFrame:
    df = pd.read_csv(full_feats[name], index_col="ID")
    df = df.drop(columns=["SMILES", "SELFIES"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1)
    return df


datasets = {name: load_feature_dataset(name) for name in DATASET_NAMES}

train_df = datasets[TRAIN_NAME]
target_df = datasets[TEST_NAME]

common_idx = train_df.index.intersection(target_df.index)
print(len(common_idx))

train_sample = train_df.loc[common_idx]
target_sample = target_df.loc[common_idx]

# --- Train model
print(f"[Multi-Target RF] train={TRAIN_NAME}, test={TEST_NAME}, id={IDENTIFIER}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")

model=TL(log_identifier=IDENTIFIER)
model.trainMultiTargetRFModels(
    features_df=train_sample,
    targets_df=target_sample,
    output_csv=f"{IDENTIFIER}.csv",
    existing_performance_csv= (
        paths["prediction_output_dirs"][CROSS_EMBEDDING_DIR][IDENTIFIER] / f"{IDENTIFIER}.csv"
        ),
    hyper_params={
    "n_estimators": [200],
    "max_features": ["sqrt"],
    "max_depth": [50],
    "min_samples_split": [2],
    "min_samples_leaf": [2],
    },
    n_resamples=1,
    test_size=0.3,
    save_path=paths["prediction_output_dirs"][CROSS_EMBEDDING_DIR][IDENTIFIER],
    skip_existing=True,
    save_models=False,
)
