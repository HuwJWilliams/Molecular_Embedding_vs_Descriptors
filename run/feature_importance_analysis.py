"""
Run feature-importance averaging for individual property RF predictions.
"""

# %% ===== Python Imports =====
import argparse
import sys

# %% ===== Project Imports & Pathing Setup =====
from config import (
    SRC_DIR,
    PATHING_JSON_PATH,
    SUPPORTED_FEATURE_SETS,
)
from feature_importance_analysis_fns import runFeatureImportanceAnalysis

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)
RESULTS_DIR = FULL_PATHING["imp_dirs"]["results_dir"]
PREDS_DIR = FULL_PATHING["prediction_output_dirs"]["rf"]


# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(
    description="Average RF feature importances across repeats."
)

parser.add_argument(
    "--properties",
    nargs="+",
    default=list(PREDS_DIR.keys()),
    choices=list(PREDS_DIR.keys()),
    help="Properties to analyse.",
)

parser.add_argument(
    "--feature-sets",
    nargs="+",
    default=SUPPORTED_FEATURE_SETS,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature sets to analyse.",
)

parser.add_argument(
    "--save-dir",
    default=str(RESULTS_DIR / "feature_importance"),
    help="Directory to save averaged feature-importance outputs.",
)

parser.add_argument(
    "--top-n-feats",
    type=int,
    default=50,
    help="Number of top features to plot.",
)

args = parser.parse_args()


# %% ===== Running Analysis =====
runFeatureImportanceAnalysis(
    properties=args.properties,
    feature_sets=args.feature_sets,
    preds_dir=PREDS_DIR,
    save_dir=args.save_dir,
    top_n_feats=args.top_n_feats,
)
