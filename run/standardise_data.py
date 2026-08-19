"""
Run this script to standardise data by:
    1- Averaging ranges of values and uncertainties
    2- Removing fragments in SMILES strings
    3- Removing metals
    4- Dropping NaN values
    5- Shuffling rows
    6- Creating sequential IDS
    7- Updating the pathing.json
    8- Optionally plots the feature distribution of selected descriptors. Default is:
        MolWt, MolLogP, NumAromaticRings, NumRotatableBonds, NumHDonors, NumHAcceptors
"""

# %% ===== Python Imports =====
from pathlib import Path
import sys
import argparse
import json

# %% ===== Project Imports & Pathing Setup =====
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(RUN_DIR / "config"))
from config import PATHING_PATH, PATHING_JSON_PATH, SRC_DIR

sys.path.insert(0, str(PATHING_PATH))
sys.path.insert(0, str(SRC_DIR / "datasets"))

from get_paths import addNewDatasetPaths, getPaths
from standardise_dataset import cleanAndSaveDataset

FULL_PATHING = getPaths(PATHING_JSON_PATH)

# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(
    description="Standardising datasets for the workflow and adding cleaned paths to global pathing"
)
supported_names = list(FULL_PATHING["raw_data"].keys())

parser.add_argument(
    "--dataset-name",
    required=True,
    choices=supported_names,
    help="Name of raw data paths in pathing json. Current supported:\n"
    f"{supported_names}",
)

parser.add_argument(
    "--target-col", required=True, help="Name of target column in the raw dataset file"
)

parser.add_argument(
    "--smiles-col", required=True, help="Name of smiles column in raw dataset file"
)

parser.add_argument(
    "--id-col", default=None, help="Name of the id column in raw dataset file"
)

parser.add_argument(
    "--id-prefix",
    default="id",
    help="If no existing ID column, set this to create unique ID column (e.g., bp-1, ...)",
)

parser.add_argument("--shuffle-data", action="store_true", help="Flag to shuffle data")

parser.add_argument(
    "--plot-distribution",
    action="store_true",
    help="Plots distribution of Lipinski features in dataset",
)

parser.add_argument(
    "--rename-cols", default=None, help="Dictionary to rename columns (e.g., \n \
        {'Boiling Point {measured, converted}: 'Boiling _Point, 'Unit, K' : 'Unit'})"
)

args = parser.parse_args()
dataset_name = args.dataset_name

raw_data_path = Path(FULL_PATHING["raw_data"][dataset_name])
data_name = raw_data_path.stem
cleaned_data_path = (
    FULL_PATHING["imp_dirs"]["datasets_dir"] / dataset_name / f"{data_name}_cleaned.csv"
)

usecols = [args.smiles_col, args.target_col]
if bool(args.id_col):
    usecols.append(args.id_col)

rename_cols = json.loads(args.rename_cols) if args.rename_cols else None

# %% ===== Standardise Dataset =====
cleanAndSaveDataset(
    in_path=raw_data_path,
    out_path=cleaned_data_path,
    usecols=usecols,
    target_col=args.target_col,
    smiles_col=args.smiles_col,
    id_col=args.id_col,
    rename=rename_cols,
    shuffle_rows=args.shuffle_data,
    plot_feature_distribution=args.plot_distribution,
    id_prefix=args.id_prefix,
)

addNewDatasetPaths(
    dataset_key=dataset_name,
    target_file=cleaned_data_path.name,
    dataset_prefix=dataset_name,
    dataset_folder_name=dataset_name,
    json_path=PATHING_JSON_PATH,
)
