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

PATHS_JSON = str(SRC_DIR / "pathing" / "test_paths.json")
paths = getPaths(PATHS_JSON)
# endregion

# region Argument Parsing
p = argparse.ArgumentParser(description="Generate descriptors/embeddings for a target dataset.")

p.add_argument(
    "--task", 
    required=True, 
    choices=list(paths["targets"].keys()),
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

task = args.task
feature_set = args.feature_set.lower()
batch_size = args.batch_size

# --- Paths
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
