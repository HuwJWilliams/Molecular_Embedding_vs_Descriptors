"""
Scrip to generate features for datasets and saving them to the pathing
"""

# region Imports and Pathing
from pathlib import Path
import sys
import argparse
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

# Pathing
sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_generator import FeatureGenerator


sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import SUPPORTED_FEATURE_SETS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths
# endregion

# region Argument Parsing
p = argparse.ArgumentParser(description="Generate descriptors/embeddings for a target dataset.")

p.add_argument(
    "--paths-json",
    default=None,
    help=(
        "Optional path to a pathing JSON file. If omitted, uses the canonical "
        "paths.json via getPaths()."
    ),
)

p.add_argument(
    "--task", 
    required=True, 
    help="Target dataset to generate features for."
    )

p.add_argument(
    "--feature-set", 
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature set to generate. Currently supported options are:\n" \
        f"{SUPPORTED_FEATURE_SETS}")

# Batch size for processing 
p.add_argument(
    "--batch-size", 
    type=int, 
    default=100000,
    help="Batch size for processing (in development)"
    )

args = p.parse_args()

paths = getPaths(args.paths_json) if args.paths_json else getPaths()

task = args.task
feature_set = args.feature_set.lower()
batch_size = args.batch_size

# --- Paths
if task not in paths["targets"]:
    raise KeyError(
        f"Task '{task}' is not available in paths['targets']. "
        f"Available tasks: {list(paths['targets'].keys())}"
    )

if task not in paths["full_features"]:
    raise KeyError(
        f"Task '{task}' is not available in paths['full_features']. "
        f"Available tasks: {list(paths['full_features'].keys())}"
    )

if feature_set not in paths["full_features"][task]:
    raise KeyError(
        f"Feature set '{feature_set}' missing for task '{task}' in paths['full_features']. "
        f"Available feature sets for '{task}': {list(paths['full_features'][task].keys())}"
    )

in_path = paths["targets"][task]
out_path = paths["full_features"][task][feature_set]

in_df = pd.read_csv(in_path, index_col="ID")

# --- Run Generator
FG = FeatureGenerator(feature_set=feature_set, log_name=f"FG_{task}")
out_df_paths = FG.calcBatchFeatures(
    smiles_ls=in_df["SMILES"].to_list(),
    id_ls = in_df.index.to_list(),
    fpath=out_path,
    drop_cols=False,
    batch_size=batch_size
)
