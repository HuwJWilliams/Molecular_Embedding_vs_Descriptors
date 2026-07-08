"""
Script to train Random Forest Regressors on one set of features to predict another
(e.g., train on ChemBERTa embeddings to predict RDKit descriptors)
"""

# %% ===== Python Imports =====
import argparse
import pandas as pd
from pathlib import Path
import sys

# %% ===== Project Imports & Pathing Setup =====
from config import SRC_DIR, SUPPORTED_FEATURE_SETS, PATHING_JSON_PATH

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_cleaning import cleanFeatureDF

# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(description="Generating cross-feature predictions")

parser.add_argument(
    "--train",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to train RF models on. Choices are:\n" f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--target",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to test RF models on. Choices are:\n" f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--save-dir",
    default="cross_feature_predictions",
    help="Directory to save the results to",
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
    help="Number of features to process at each node",
)

parser.add_argument(
    "--max-depth",
    nargs="+",
    type=int,
    default=[50],
    help="Maximum depth of the Random Forest",
)

parser.add_argument(
    "--min-samples-split",
    nargs="+",
    type=int,
    default=[2],
    help="Minimum samples to split a node",
)

parser.add_argument(
    "--min-samples-leaf",
    nargs="+",
    type=int,
    default=[2],
    help="Minimum samples to make a leaf node",
)

parser.add_argument(
    "--n-resamples", type=int, default=1, help="Number of resamples in the outer loop"
)

parser.add_argument(
    "--test-size",
    type=float,
    default=0.3,
    help="Fraction of the training set (0.3 = 30 %)",
)

parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Flag to skip already processed features",
)

parser.add_argument(
    "--save-models", action="store_true", help="Flag to save trained models"
)

parser.add_argument(
    "--shuffle-data",
    action="store_true",
    help="Flag to shuffle the data prior to training",
)

parser.add_argument(
    "--save-feat-imp", action="store_true", help="Flag to save feature importance data"
)

parser.add_argument(
    "--lipinski-mols",
    action="store_true",
    help="Flag to do cross-feature predictions on Lipinksi-fitting molecules only \
        (if created using join_all_target_molecule_datasets.py, otherwise it will crash)",
)

parser.add_argument(
    "--minimum-targs",
    type=int,
    default=2500,
    help="Minimum number of samples to train RF models",
)

args = parser.parse_args()
mols = "fit_lipinski" if args.lipinski_mols else "all"
train = args.train.lower()
target = args.target.lower()
identifier = f"pred_{target}_tr_{train}"
save_dir = args.save_dir
FULL_PATHING = getPaths(PATHING_JSON_PATH)
full_feats = FULL_PATHING["full_features"][mols]


# %% ===== Helper Functions =====
def loadAndCleanData(
    name: str,
    full_feats: dict,
    correlation_threshold: float | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(full_feats[name], index_col="ID")

    df, clean_report = cleanFeatureDF(
        df,
        max_nan_fraction=0.10,
        drop_constant_cols=True,
        median_impute=True,
        correlation_threshold=correlation_threshold,
    )

    print(f"\nFeature cleaning report for {name}:")
    print(clean_report)
    print(f"Final shape: {df.shape}")

    return df


# %% ===== Running Cross-Feature Prediction Experiments =====
train_df = loadAndCleanData(train, full_feats=full_feats, correlation_threshold=None)

target_df = loadAndCleanData(target, full_feats=full_feats, correlation_threshold=None)

common_idx = train_df.index.intersection(target_df.index)
train_df, target_df = train_df.loc[common_idx], target_df.loc[common_idx]

print(f"[Multi-Target RF] train={train}, test={target}, id={identifier}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")

model = TL(log_to_file=False, log_identifier=identifier)
model.trainMultiTargetRFModels(
    features_df=train_df,
    targets_df=target_df,
    output_csv=f"{identifier}.csv",
    existing_performance_csv=(
        FULL_PATHING["prediction_output_dirs"][save_dir][identifier]
        / f"{identifier}.csv"
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
    save_path=FULL_PATHING["prediction_output_dirs"][save_dir][identifier],
    skip_existing=args.skip_existing,
    save_models=args.save_models,
    save_feat_imp=args.save_feat_imp,
    min_training_samples=args.minimum_targs,
)
