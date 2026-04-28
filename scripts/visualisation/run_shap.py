"""
Separated script to run any of the SHAP plotting functions defined in src/visualisation/vis.py
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import random
from glob import glob

# --- Paths
FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

PATHS = getPaths()

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import SUPPORTED_FEATURE_SETS

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups
# endregion

# region Class and Parser Setup
v = Visualise()

parser = argparse.ArgumentParser(
    description="Generating SHAP analysis"
)

# endregion

# region Argument Parsing
parser.add_argument(
    "--pred-set",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature set predicted on. Used to obtain the results directory \n \
        (e.g., pred_{pred_set}_tr_{train_set} for cross-feature predictions) \n"
)

parser.add_argument(
    "--train-set",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature set trained on. Used to obtain the results directory \n \
        (e.g., pred_{pred_set}_tr_{train_set} for cross-feature predictions) \n"
)

parser.add_argument(
    "--pred-feat",
    default="MolWt",
    help="Feature to run SHAP analysis on, defaulted to MolWt for testing"
)

parser.add_argument(
    "--results-dir",
    default="cross_feature_predictions",
    help="Pathing key in paths['prediction_output_dirs'], e.g., 'cross_feature_predictions"
)

parser.add_argument(
    "--max-bg",
    type=int,
    default=200,
    help="Maximum number of rows to use as the SHAP background baseline"
)

parser.add_argument(
    "--max-exp",
    type=int,
    default=500,
    help="Maximum rows to use for SHAP explanations"
)

parser.add_argument(
    "--n-display",
    type=int,
    default=20,
    help="Number of features to display on SHAP beeswarm plot"
)

parser.add_argument(
    "--plot-shap",
    action="store_true",
    help="Flag to plot the SHAP beeswarm analysis"
)

parser.add_argument(
    "--plot-dep",
    action="store_true",
    help="Flag to plot dependence plots"
)

parser.add_argument(
    "--group-shap",
    action="store_true",
    help="Flag to run group SHAP analysis"
)

parser.add_argument(
    "--top-n",
    type=int,
    default=12,
    help="Showing the number of top impacting features"
)

parser.add_argument(
    "--save-full-shap",
    action="store_true",
    help="Flag to store the full shap analysis csv so calculations dont need to be \
        run over and over."
)

parser.add_argument(
    "--check-full-shap",
    action="store_true",
    help="Flag to look for full shap CSV and use results in there first,\
    if not present, calculate again"
)

parser.add_argument(
    "--full-feat-path",
    default="fit_lipinski",
    help="Key in paths.json in full_features"
)

args = parser.parse_args()
# endregion

# region Argument Preparation
pred_set = args.pred_set.lower()
train_set = args.train_set.lower()
pred_feat = f"{args.pred_feat}_{pred_set}"
bg = args.max_bg
exp = args.max_exp
# SHAP must run on the model input space (train_set), not the predicted target space (pred_set).
full_train_feats_ = PATHS["full_features"][args.full_feat_path][train_set]

results_dir = \
    PATHS["prediction_output_dirs"][args.results_dir][f"pred_{pred_set}_tr_{train_set}"] / "training_data"
shap_dir = results_dir.parent / "shap"
shap_dir.mkdir(parents=True, exist_ok=True)

available_models = sorted(glob(str(results_dir / "*.pkl*")))

model_path = next((p for p in available_models if pred_feat in Path(p).name), None)
if model_path is None and not args.check_full_shap:
    raise FileNotFoundError(f"No model found for '{pred_feat}' in {results_dir}")

tr_feat_path = results_dir / "training_features.csv.gz"
full_shap_path = shap_dir / "full_shap_analysis.joblib.gz"

if args.save_full_shap:
    print(f"Saving full SHAP bundle to: {full_shap_path}")
    v.shapAnalysisAll(
        models_dir=results_dir,
        features=full_train_feats_,
        output_dir=shap_dir,
        max_bg=bg,
        max_explain=exp,
        save_full=args.save_full_shap,
        full_shap_name=full_shap_path.name,
        check_full=args.check_full_shap
    )

shap_v, feat_explain, explainer = v.shapAnalysis(
    model=model_path,
    features=full_train_feats_,
    pred_feature=pred_feat,
    output_dir=shap_dir,
    max_bg=bg,
    max_explain=exp,
    plot=args.plot_shap,
    max_display=args.n_display,
    check_full=args.check_full_shap,
    full_shap_path=full_shap_path,
)

if args.plot_dep:
    v.shapDependencePlot(
        shap_values=shap_v,
        pred_feature=pred_feat,
        feat_explain=feat_explain, 
        explainer=explainer,
        output_dir=shap_dir
    )

if args.group_shap:
    if pred_set in ["rdkit", "mordred", "maccs"]:
        v.shapAnalysisForGroups(
            models_dir=results_dir,
            features=tr_feat_path,
            output_dir=shap_dir,
            max_bg=bg,
            max_explain=exp,
            descriptor_groups=getGroups(pred_set),
            top_n=args.top_n,
            save_full=args.save_full_shap
        )

    else:
        raise ValueError(f"Cannot get groups for {pred_set}.\n \
                          Groupings are only valid for 'rdkit' and 'mordred'.")
