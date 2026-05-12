"""
Script to generate features for datasets and saving them to the pathing
"""
# region Script Functionality
# region Imports
import argparse
import pandas as pd
from pathlib import Path
import sys
# endregion

# region Path Setup
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "datasets"))
sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SCRIPTS_DIR / "config"))

from feature_generator import FeatureGenerator
from pipeline_config import SUPPORTED_FEATURE_SETS
from get_paths import getPaths

# endregion

# region Argument Setup
parser = argparse.ArgumentParser(
    description="Generate descriptors, fingerprints, or embeddings for a target dataset."
)

parser.add_argument(
    "--task",
    required=True,
    help="Dataset key to generate features for, e.g. bp, pka, ld50.",
)

parser.add_argument(
    "--feature-set",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature set to generate.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=100_000,
    help="Number of molecules to process per batch.",
)

parser.add_argument(
    "--paths-json",
    default=None,
    help="Optional pathing JSON. Defaults to the canonical paths.json.",
)
# endregion


# region Resolve Feature Paths Helper
def resolve_feature_paths(paths: dict, task: str, feature_set: str):
    if task not in paths["targets"]:
        available = list(paths["targets"].keys())
        raise KeyError(
            f"Task '{task}' is not available in paths['targets']. "
            f"Available tasks: {available}"
        )

    if task not in paths["full_features"]:
        available = list(paths["full_features"].keys())
        raise KeyError(
            f"Task '{task}' is not available in paths['full_features']. "
            f"Available tasks: {available}"
        )

    if feature_set not in paths["full_features"][task]:
        available = list(paths["full_features"][task].keys())
        raise KeyError(
            f"Feature set '{feature_set}' missing for task '{task}'. "
            f"Available feature sets: {available}"
        )
    return paths["targets"][task], paths["full_features"][task][feature_set]
# endregion
# endregion

# region Running Script
if __name__ == "__main__":
    args = parser.parse_args()

    paths = getPaths(args.paths_json) if args.paths_json else getPaths()

    task = args.task.lower()
    feature_set = args.feature_set.lower()

    in_path, out_path = resolve_feature_paths(
        paths=paths,
        task=task,
        feature_set=feature_set,
    )

    in_df = pd.read_csv(in_path, index_col="ID")

    if "SMILES" not in in_df.columns:
        raise KeyError(f"Input file must contain a 'SMILES' column: {in_path}")

    generator = FeatureGenerator(
        feature_set=feature_set,
        log_name=f"FG_{task}",
    )

    generator.calcBatchFeatures(
        smiles_ls=in_df["SMILES"].to_list(),
        id_ls=in_df.index.to_list(),
        fpath=out_path,
        batch_size=args.batch_size,
    )
# endregion
