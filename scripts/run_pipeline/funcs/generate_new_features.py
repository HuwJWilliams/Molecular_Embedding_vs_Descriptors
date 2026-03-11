import sys
import argparse
import pandas as pd
from pathlib import Path

# Paths
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))

from path.get_paths import getPaths
all_paths = getPaths()

# Importing Feature Generator
sys.path.insert(0, str(SCRIPTS_DIR / "datasets"))
from feature_generator import FeatureGenerator

FEATURE_SETS = ["rdkit", "mordred", "chemberta", "molformer", "morgan"]


def main():
    parser = argparse.ArgumentParser(description="Generate features for one or more datasets")

    # Multiple Datasets
    parser.add_argument(
        "--tasks", 
        nargs="+",
        required=True, 
        choices=[k for k in all_paths["targets"].keys() if k != "all"],
        help="Datasets to generate features for"
    )

    # Multiple Feature Sets
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        required=True,
        dest="feature_sets",
        choices=FEATURE_SETS,
        help="Feature sets to generate"
    )

    parser.add_argument("--batch-size", type=int, default=10000, help="Batch size for feature generation")

    args = parser.parse_args()
    tasks = args.tasks
    feature_sets = [f.lower() for f in args.feature_sets]
    batch_size = args.batch_size

    # Feature Generation Loop
    for task in tasks:
        in_path = all_paths["targets"][task]
        in_df = pd.read_csv(in_path, index_col="ID")

        smiles_ls = in_df["SMILES"].to_list()
        id_ls = in_df.index.tolist()

        for feature_set in feature_sets:
            print(f"\n=== Generating {feature_set} features for {task} ===")

            out_path = all_paths["full_features"][task][feature_set]

            FG = FeatureGenerator(
                feature_set=feature_set,
                log_name=f"FG_{task}_{feature_set}"
            )

            try:
                FG.calcBatchFeatures(
                    smiles_ls=smiles_ls,
                    id_ls=id_ls,
                    fpath=out_path,
                    drop_cols=True,
                    batch_size=batch_size
                    )
            
            except Exception as e:
                print(f"[FAILED] {task} / {feature_set}:\n{e}")

# === Running script === #
if __name__ == "__main__":
    main()