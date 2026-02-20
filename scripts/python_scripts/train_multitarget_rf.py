"""Script to train a Multitarget Random Forest Regressors i.e., training to predict each column in target_df"""

import sys
import pandas as pd
from pathlib import Path
import numpy as np

# --- Paths
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = FILE_DIR.parents[1]

sys.path.insert(0, str(SCRIPTS_DIR))
from models.transfer_model import TL
from path.get_paths import getPaths

PALMERCHEM_SOFTWARE = Path.home() / "PalmerChem_Software" / "src" / "models"
sys.path.insert(0, str(PALMERCHEM_SOFTWARE))
from RFRegressor import RFRegressor

paths = getPaths()
feature_paths = paths["full_features"]
aligned_feats = feature_paths["all"]

# --- Arguments
TRAIN_NAME = sys.argv[1]
TEST_NAME = sys.argv[2]
IDENTIFIER = sys.argv[3]

print("\nInput Parameters\n====================")
print(TRAIN_NAME)
print(TEST_NAME)
print(IDENTIFIER)




RDKIT_DESC_PATH = aligned_feats['rdkit']
MORDRED_DESC_PATH = aligned_feats['mordred']
CHEMBERTA_EMB_PATH = aligned_feats['chemberta']
MOLFORMER_EMB_PATH = aligned_feats['molformer']

# --- Load datasets
datasets = {
    "rdkit":     pd.read_csv(RDKIT_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore"),
    "mordred":   pd.read_csv(MORDRED_DESC_PATH, index_col="ID").drop(columns=["SMILES"], errors="ignore"),
    "chemberta": pd.read_csv(CHEMBERTA_EMB_PATH, index_col="ID"),
    "molformer": pd.read_csv(MOLFORMER_EMB_PATH, index_col="ID"),
}

train_df = datasets[TRAIN_NAME]
target_df = datasets[TEST_NAME]

common_idx = train_df.index.intersection(target_df.index)
sample_size = 10000
random_idx = np.random.choice(common_idx, size=sample_size, replace=False)
random_idx_df = pd.DataFrame({"ID": random_idx})
random_idx_df.to_csv(PROJ_DIR / "results" / "embeddings_and_descriptor_predictions" / "trained_ids.csv")

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
    existing_performance_csv=f"{IDENTIFIER}.csv",
    hyper_params={
    "n_estimators": [200],
    "max_features": ["sqrt"],
    "max_depth": [50],
    "min_samples_split": [2],
    "min_samples_leaf": [2],
    },
    n_resamples=10,
    test_size=0.3,
    save_path=PROJ_DIR / "results" / "embeddings_and_descriptor_predictions",
    skip_existing=True,
    save_models=False,
    random_seed=model.rng()
)