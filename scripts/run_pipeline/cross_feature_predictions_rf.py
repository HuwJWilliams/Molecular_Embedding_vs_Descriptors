"""
Script to train Random Forest Regressors on one set of features to predict another 
(e.g., train on ChemBERTa embeddings to predict RDKit descriptors)
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import argparse

# --- Paths
FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent
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

# !!! NEED TO COPY PALMERCHEM RFR IN THIS REPO !!! ""

paths = getPaths()
feature_paths = paths["full_features"]
full_feats = feature_paths["all"]
# endregion

# region Function Definitions
def _load_feature_datasets(name: str) -> pd.DataFrame:
    df = pd.read_csv(full_feats[name], index_col="ID")
    df = df.drop(columns=["SMILES"], errors="ignore")
    return df


# endregion

# region Argument Parsing
parser = argparse.ArgumentParser(
    description="Generating cross-feature predictions"
    )

parser.add_argument(
    "--train",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to train RF models on. Choices are:\n" \
        f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--test",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to test RF models on. Choices are:\n" \
        f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--save-dir",
    default="cross_feature_predictions",
    help="Directory to save the results to"
)

parser.add_argument(
    "--n-estimators",
    nargs="+",
    type=int,
    default=[200],
)

parser.add_argument(
    "--max-features",
    nargs="+",
    default=["sqrt"],
    help="Number of features to process at each node"
)

parser.add_argument(
    "--max-depth",
    nargs="+",
    type=int,
    default=[50],
    help="Maximum depth of the Random Forest"
)

parser.add_argument(
    "--min-samples-split",
    nargs="+",
    type=int,
    default=[2],
    help="Minimum samples to split a node"
)

parser.add_argument(
    "--min-samples-leaf",
    nargs="+",
    type=int,
    default=[2],
    help="Minimum samples to make a leaf node"
)

parser.add_argument(
    "--n-resamples",
    type=int,
    default=1,
    help="Number of resamples in the outer loop"
)

parser.add_argument(
    "--test-size",
    type=float,
    default=0.3,
    help="Fraction of the training set (0.3 = 30 %)"
)

parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Flag to skip already processed features"
)

parser.add_argument(
    "--save-models",
    action="store_true",
    help="Flag to save trained models"
)


args = parser.parse_args()
test = args.test
train = args.train
save_dir = args.save_dir

identifier = f"pred_{test}_tr_{train}"
# endregion

# region Data Handling
train_df = _load_feature_datasets(train)
target_df = _load_feature_datasets(test)

common_idx = train_df.index.intersection(target_df.index)
print(f"Length of common IDs: {len(common_idx)}")

train_df, target_df = train_df.loc[common_idx], target_df.loc[common_idx]
# endregion

# region Model Training
print(f"[Multi-Target RF] train={train}, test={test}, id={identifier}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")

model = TL(log_to_file=False, log_identifier=identifier)
model.trainMultiTargetRFModels(
    features_df=train_df,
    targets_df=target_df,
    rf_regressor_class=RFRegressor,
    output_csv=f"{identifier}.csv",
    existing_performance_csv= (
        paths["prediction_output_dirs"][save_dir][identifier] / f"{identifier}.csv"
        ),
    hyper_params={
    "n_estimators": args.n_estimators,
    "max_features": args.max_features,
    "max_depth": args.max_depth,
    "min_samples_split": args.min_samples_split,
    "min_samples_leaf": args.min_samples_leaf,
    },
    n_resamples=args.n_resamples,
    test_size=args.test_size,
    save_path=paths["prediction_output_dirs"][save_dir][identifier],
    skip_existing=args.skip_existing,
    save_models=args.save_models,
    random_seed=42
)
