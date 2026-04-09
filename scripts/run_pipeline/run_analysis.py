"""
Script to run the analysis following the cross-feature predictions
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = PROJ_DIR / "results"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "pathing")) 
from get_paths import getPaths, addNewDatasetPaths, addFeatureSetPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups
from analyse_datasets import getLowVarianceColumns, plotLowVarianceColumns, getOutlierSummary

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import getFeatures

paths = getPaths()
# endregion

# region Setting up Analysis Class and Variables
v = Visualise(save_all=False)
print("Visualise class loaded")

# --- Getting default run names
cp = list(paths["prediction_output_dirs"]["cross_feature_predictions"].keys())
lcp = list(paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"].keys())
unique_exp_names = list(set(cp + lcp))
# endregion

# region Passing Arguments

parser = argparse.ArgumentParser(
    description="Generating cross-feature analysis"
    )

parser.add_argument(
    "--result-dir",
    default="cross_feature_predictions",
    help="Directory to look for cross-feature predictions.\
          Put here the name of dir as it appears in the pathing json (e,g,. 'cross_feature_predictions')"
)

parser.add_argument(
    "--run-all",
    action="store_true",
    help="Flag to run all available cross-prediciton analysis"
)

parser.add_argument(
    "--run-experiment",
    nargs="+",
    choices=unique_exp_names,
    help=f"Name of the experiments ran. For full cross-feature predictions:\n{cp}\n" \
    f"For Lipinski cross-feature predicitons:\n{lcp}\n"\
    "Note: This only works with the original default pathing"
)

parser.add_argument(
    "--exclude-low-var",
    action="store_true",
    help="Flag to exclude low variance columns"
)

parser.add_argument(
    "--show-var",
    action="store_true",
    help="Flag to show the variance of target features"
)

parser.add_argument(
    "--var-threshold",
    type=float,
    default=0.8,
    help="Fraction of entries which have the same common value \n" \
    "(i.e., 0.8 = 20 % of values are different from the most common)"
)

# endregion

# region Parsing Arguments
args = parser.parse_args()
cp_dir = paths["prediction_output_dirs"][args.result_dir]
var_threshold = args.var_threshold

# endregion

# region Running script functionality
if args.run_all:
    exp_list = list(cp_dir.keys())

elif not args.run_all and bool(args.run_experiment):
    exp_list = args.run_experiment

else:
    raise ValueError("You must set either '--run-all' or specify results with '--run-experiment'"\
                     f"Experiments to choose from:\n{unique_exp_names}")

for exp in exp_list:
    wrds = exp.split("_")
    pred = wrds[1]
    tr = wrds[3]

    exp_dir = cp_dir[exp]
    exp_perf_df_path = exp_dir / f"{exp}.csv"
    exp_perf_df = pd.read_csv(Path(exp_perf_df_path), index_col=0)

    pred_ft_df = Path(paths["full_features"]["all"][pred])
    
    l_var_col = getLowVarianceColumns(
        pred_ft_df, threshold=var_threshold
        )
    
    excl_cols = l_var_col if args.exclude_low_var else []
    
    if args.show_low_var:
        desc_an_dir = pred_ft_df.parent / "descriptor_analysis"
        save_name = f"low_variance_features_{pred}"

        if not Path(desc_an_dir / save_name).exists():
            print(f"Plotting low variance columns in following path:\n{desc_an_dir / save_name}")
            
            plotLowVarianceColumns(
                input_df=pred_ft_df,
                threshold=var_threshold,
                output_path=desc_an_dir,
                save_name=save_name)
        else:
            print(f"Low variance column plot exists in following path:\n{desc_an_dir / save_name}")
        
    group_map = getGroups(pred)

    group_performance_df = v.computeGroupPerf(
        data=exp_perf_df,
        descriptor_groups=group_map,
        metrics=["Pearson_r", "r2", "RMSE", "Bias"],
        exclude=excl_cols
    )

# --- Plotting the overall cross-prediction performance
    gr_title=f"{pred.capitalize()} Prediction ({tr.capitalize()} trained)"
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
    gr_description=f"Performance when training RFR models on {tr} to predict {pred} features.\n \
                Plot shows the Pearson R of predictions grouped by similar features \n \
                Created: {timestamp}"
    gr_fname_suffix = "excl_low_var" if args.exclude_low_var else ""

    v.plotGroupRadar(
        group_performance_df,
        title=gr_title,
        save_plot=True,
        save_path=exp_dir,
        save_fname=f"{exp}_{gr_fname_suffix}_group_radar",
        metadata={
            "Title": gr_title,
            "Description": gr_description
        }
    )
    
    for group_name, group_members in group_map.items():
# --- Plotting performance of individual members of a group
        mb_title=f"Performance for {group_name} (trained {tr.capitalize()}, predicted {pred.capitalize()})"
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        mb_description = f"Performance on {pred.capitalize()} features in the group '{group_name}.\n \
            This group consists of:\n {group_members}"
    
        v.plotMemberBar(
            perf_df=exp_perf_df,
            group_map=group_map,
            group_name=group_name,
            value_col="Pearson_r",
            save_plot=True,
            save_path=exp_dir,
            save_fname=f"{exp}_{group_name}_{pred}_bar",
            metadata={
                "Title": mb_title,
                "Description": mb_description
            }
            )

# --- Plotting the feature distribution of poorly predicted features
        v.plotPoorPredictionFeatureDistribution(
                perf_df=exp_perf_df,
                full_features=paths["full_features"]["all"][pred],
                group_map=group_map,
                group_name=group_name,
                value_col="Pearson_r",
                save_plot=True,
                save_path=exp_dir,            
        )

# endregion