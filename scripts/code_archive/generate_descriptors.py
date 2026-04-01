# %%
import sys
import argparse
import pandas as pd
from pathlib import Path

FILE_DIR = Path(__file__).resolve()
SCRIPTS_DIR = FILE_DIR.parents[2]
SRC_DIR = SCRIPTS_DIR / "src"
print(f"Project Reference Dir for file {FILE_DIR.name}:\n{SCRIPTS_DIR}")
# %%
sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_generator import FeatureGenerator

sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import SUPPORTED_FEATURE_SETS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

data_paths = getPaths()

# --- Adding script arguments
p = argparse.ArgumentParser(description="Generate descriptors/embeddings for a target dataset.")

# Target dataset to generate features for
p.add_argument("--task", required=True, choices=list(data_paths["targets"].keys()))

# Feature set to generate
p.add_argument("--feature-set", required=True, dest="feature_set",
               choices=SUPPORTED_FEATURE_SETS)

# Batch size for processing 
p.add_argument("--batch-size", type=int, default=100000)

args = p.parse_args()

task = args.task
feature_set = args.feature_set.lower()
batch_size = args.batch_size

# --- Paths
in_path = data_paths["targets"][task]
out_path = data_paths["full_features"][task][feature_set]

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
