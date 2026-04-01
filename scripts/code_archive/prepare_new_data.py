from pathlib import Path
import sys
import argparse
import json

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

# Pathing
sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import addNewDatasetPaths

# Standardisation Functions
sys.path.insert(0, str(SRC_DIR / "datasets"))
from standardise_dataset import cleanAndSaveDataset


# === Setting Up System Arguments === #
def main():
    parser = argparse.ArgumentParser(description="Add and clean a new property dataset")

    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--dataset-prefix", required=True)
    parser.add_argument("--dataset-folder", required=True)

    parser.add_argument("--raw-data-path", required=True)
    parser.add_argument("--cleaned-data-path", required=True)

    parser.add_argument("--usecols", required=True,
                        help='List, e.g., ["SMILES", "pIC50"]')
    parser.add_argument("--rename", default="{}",
                        help='Dict, e.g., {"old_column_name": "new_column_name"}')
    
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--unit-col", default="Unit")
    parser.add_argument("--id-prefix", required=True)

    parser.add_argument("--plot-analysis", default=True)

    args = parser.parse_args()

    usecols = json.loads(args.usecols)
    rename = json.loads(args.rename)

    raw_data_path = Path(args.raw_data_path)
    cleaned_data_path = Path(args.cleaned_data_path)

    # Clean Dataset
    cleanAndSaveDataset(
        in_path=raw_data_path,
        out_path=cleaned_data_path,
        usecols=usecols,
        rename=rename,
        target_col=args.target_col,
        unit_col=args.unit_col,
        id_prefix=args.id_prefix,
        average_ranges=True
    )

    # Save New Paths
    addNewDatasetPaths(
        dataset_key=args.dataset_key,
        target_file=cleaned_data_path.name,
        dataset_prefix=args.dataset_prefix,
        dataset_folder_name=args.dataset_folder
    )

    print(f"Finished preparing dataset: {args.dataset_key}")


# === Running script === #
if __name__ == "__main__":
    main()
