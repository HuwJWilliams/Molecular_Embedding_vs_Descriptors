"""
Script to run the analysis following the cross-feature predictions.
"""

# region Imports and Pathing
import argparse
import sys
from pathlib import Path

import pandas as pd

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import getLowVarianceColumns
from group_descriptors import getGroups

paths = getPaths()
# endregion

# region Setup
v = Visualise(save_all=False)
print("Visualise class loaded")

cp = list(paths["prediction_output_dirs"]["cross_feature_predictions"].keys())
lcp = list(paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"].keys())
unique_exp_names = sorted(set(cp + lcp))
# endregion

# region Arguments
parser = argparse.ArgumentParser(description="Generating cross-feature analysis")

parser.add_argument(
    "--result-dir",
    default="cross_feature_predictions",
    help=(
        "Name of the result dir as it appears in the pathing json "
        "(e.g. 'cross_feature_predictions')"
    ),
)
parser.add_argument(
    "--run-all",
    action="store_true",
    help="Run all available cross-prediction experiments",
)
parser.add_argument(
    "--run-experiment",
    nargs="+",
    choices=unique_exp_names,
    help=(
        f"Experiments to run. Full cross-feature:\n{cp}\n"
        f"Lipinski cross-feature:\n{lcp}\n"
        "Note: only works with the original default pathing."
    ),
)
parser.add_argument(
    "--exclude-low-var",
    action="store_true",
    help="Exclude low-variance columns from regression bar plots and group averages",
)
parser.add_argument(
    "--show-var",
    action="store_true",
    help="Save a low-variance overview plot for the predicted feature set",
)
parser.add_argument(
    "--var-threshold",
    type=float,
    default=0.8,
    help="Low-variance threshold: fraction of rows sharing the most common value",
)
parser.add_argument(
    "--no-groups",
    action="store_true",
    help="Skip group-level radar and member plots",
)
parser.add_argument(
    "--skip-cols",
    nargs="+",
    help="Descriptor names to skip from all bar plots",
)
parser.add_argument(
    "--radar-skip-cols",
    nargs="+",
    default=["Ipc"],
    help="Descriptor names to skip from radar / group-average plots only",
)
parser.add_argument("--radar-task", default="regression")
parser.add_argument(
    "--run-avg",
    action="store_true",
    help="Average performance across all specified experiments before plotting",
)
parser.add_argument(
    "--plot-poor-distributions",
    action="store_true",
    help="Plot raw descriptor distributions for members below a performance threshold",
)
parser.add_argument(
    "--poor-distribution-threshold",
    type=float,
    default=0.6,
    help="Performance threshold for --plot-poor-distributions",
)

args = parser.parse_args()
# endregion

# region Constants
cp_dir = paths["prediction_output_dirs"][args.result_dir]
var_threshold = args.var_threshold

TASK_METRICS = {
    "regression": {
        "metric": "Pearson_r",
        "bar_metrics": ["Pearson_r", "r2"],
        "group_metrics": ["Pearson_r", "r2", "RMSE", "Bias"],
        "member_suffix": "reg",
        "radar_metrics": ["avg_Pearson_r", "avg_r2"],
    },
    "classification": {
        "metric": "AUC",
        "bar_metrics": ["AUC", "MCC", "Balanced_Accuracy"],
        "group_metrics": [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "AUC",
            "MCC",
            "Balanced_Accuracy",
        ],
        "member_suffix": "cla",
        "radar_metrics": ["avg_AUC"],
    },
    "multiclass": {
        "metric": "Balanced_Accuracy",
        "bar_metrics": ["AUC_OVR", "MCC", "Balanced_Accuracy"],
        "group_metrics": ["Accuracy", "Balanced_Accuracy", "F1_macro", "AUC_OVR", "MCC"],
        "member_suffix": "mcla",
        "radar_metrics": ["avg_AUC_OVR"],
    },
}

TASK_TYPE_MAP = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}
# endregion

# region Data helpers (no plotting — stay in run script)
def avg_results(exp_list: list[str]) -> pd.DataFrame:
    """Average per-descriptor performance CSVs across multiple experiments."""
    dfs = [pd.read_csv(cp_dir[e] / f"{e}.csv", index_col=0) for e in exp_list]
    return pd.concat(dfs, axis=0, keys=exp_list).groupby(level=1).mean(numeric_only=True)
# endregion

# region Main loop
if args.run_all:
    exp_list = list(cp_dir.keys())
elif args.run_experiment:
    exp_list = args.run_experiment
else:
    raise ValueError(
        "Specify '--run-all' or '--run-experiment'. "
        f"Available experiments:\n{unique_exp_names}"
    )

if args.run_avg:
    pred = exp_list[0].split("_")[1]
    list_to_avg = exp_list
    exp_list = [f"pred_{pred}_tr_avg"]

for exp in exp_list:
    wrds = exp.split("_")
    pred = wrds[1]
    tr = wrds[3]
    exp_dir = cp_dir[exp]

    # --- Load performance table
    exp_perf_df = (
        avg_results(list_to_avg)
        if args.run_avg
        else pd.read_csv(exp_dir / f"{exp}.csv", index_col=0)
    )

    pred_ft_path = Path(paths["full_features"]["all"][pred])
    l_var_col = getLowVarianceColumns(pred_ft_path, threshold=var_threshold)
    group_map = None if args.no_groups else getGroups(pred)

    v.plotCrossFeatureAnalysis(
        exp_perf_df=exp_perf_df,
        l_var_col=l_var_col,
        pred_ft_path=pred_ft_path,
        exp_dir=exp_dir,
        exp=exp,
        pred=pred,
        tr=tr,
        task_metrics=TASK_METRICS,
        task_type_map=TASK_TYPE_MAP,
        group_map=group_map,
        exclude_low_var=args.exclude_low_var,
        skip_cols=args.skip_cols,
        radar_skip_cols=args.radar_skip_cols,
        radar_task=args.radar_task,
        show_var=args.show_var,
        var_threshold=var_threshold,
        no_groups=args.no_groups,
        plot_poor_distributions=args.plot_poor_distributions,
        poor_distribution_threshold=args.poor_distribution_threshold,
    )
# endregion
