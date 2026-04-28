"""
Script to train Random Forest Regressors on all features apart from one, and predict the remaining
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
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

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups


# !!! NEED TO COPY PALMERCHEM RFR IN THIS REPO !!! ""

paths = getPaths()
# endregion

# region Function Definitions
def _load_feature_datasets(name: str, mols:str="all", use_nan_dfs:bool=False) -> pd.DataFrame:
    p = paths["full_features"][mols][name]
    p = Path(str(p).replace(".csv", "_with_nans.csv")) if use_nan_dfs else p

    df = pd.read_csv(p, index_col="ID")
    df = df.drop(columns=["SMILES"], errors="ignore")
    return df


# endregion

# region Argument Parsing
parser = argparse.ArgumentParser(
    description="Generating witin-feature predictions"
    )

parser.add_argument(
    "--feat",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to train RF models on. Choices are:\n" \
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
    "--excl-group-desc",
    action="store_true",
    help="Flag to exclude group descriptors in training features"
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

parser.add_argument(
    "--shuffle-data",
    action="store_true",
    help="Flag to shuffle the data prior to training"
)

parser.add_argument(
    "--save-feat-imp",
    action="store_true",
    help="Flag to save feature importance data"
)

parser.add_argument(
    "--lipinski-mols",
    action="store_true",
    help="Flag to do cross-feature predictions on Lipinksi-fitting molecules only \
        (if created using join_all_target_molecule_datasets.py, otherwise it will crash)"
)

parser.add_argument(
    "--use-nan-dfs",
    action="store_true",
    help="Flag to use full target datasets containint NaN values.\
        Set --minimum-targs to change how many observations required \
        to train RF models."
)

parser.add_argument(
    "--minimum-targs",
    type=int,
    default=2500,
    help="Minimum number of samples to train RF models"
)

args = parser.parse_args()
feat = args.feat
save_dir = args.save_dir

identifier = f"pred_{feat}_tr_{feat}"
# endregion

# region Data Handling
mols="fit_lipinski" if args.lipinski_mols else "all"

data_df = _load_feature_datasets(name=feat, mols=mols, use_nan_dfs=args.use_nan_dfs)

if args.shuffle_data:
    shuffled_idx = data_df.sample(
        frac=1, replace=False, random_state=42
        ).index
    data_df = data_df.loc[shuffled_idx]
# endregion

# region Model Training
output_csv = f"{identifier}.csv" if not args.excl_group_desc else f"{identifier}_excl_group_desc.csv"
group_map = None
if args.excl_group_desc:
    if feat.lower() not in {"rdkit", "mordred", "maccs"}:
        raise ValueError(
            "--excl-group-desc only supports feat='rdkit', feat='mordred', or feat='maccs'"
            f"(got '{feat}')"
        )
    group_map = getGroups(feat)

model = TL(log_to_file=False, log_identifier=identifier)
model.trainWithinFeatureSetRFModels(
    data_df=data_df,
    rf_regressor_class=RFRegressor,
    output_csv=output_csv,
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
    random_seed=42,
    save_feat_imp=args.save_feat_imp,
    min_training_samples=args.minimum_targs,
    group_map=group_map,
    exclude_same_group=args.excl_group_desc
)
