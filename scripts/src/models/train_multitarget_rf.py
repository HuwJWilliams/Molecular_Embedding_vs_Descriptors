"""Script to train a Multitarget Random Forest Regressors i.e., training to predict each column in target_df"""

import sys
import pandas as pd
from pathlib import Path
import numpy as np

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

PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

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
    df = df.drop(columns=["SMILES"], errors="ignore")
    return df


datasets = {name: load_feature_dataset(name) for name in DATASET_NAMES}

train_df = datasets[TRAIN_NAME]
target_df = datasets[TEST_NAME]

common_idx = train_df.index.intersection(target_df.index)
print(len(common_idx))
sample_size = len(common_idx)
rng = np.random.default_rng(42)
random_idx = rng.choice(common_idx, size=sample_size, replace=False)
random_idx_df = pd.DataFrame({"ID": random_idx})
# random_idx_df.to_csv(PROJ_DIR / "results" / "embeddings_and_descriptor_predictions" / "trained_ids.csv")

train_sample = train_df.loc[random_idx]
target_sample = target_df.loc[random_idx]

# --- Train model
print(f"[Multi-Target RF] train={TRAIN_NAME}, test={TEST_NAME}, id={IDENTIFIER}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")

model=TL(log_identifier=IDENTIFIER)
model.trainMultiTargetRFModels(
    features_df=train_sample,
    targets_df=target_sample,
    rf_regressor_class=RFRegressor,
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
    random_seed=42
)
