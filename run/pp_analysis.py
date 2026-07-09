"""
Running individual property prediction (PP) analysis
"""

# %% ===== Python Imports =====
import argparse
import json

# %% ===== Project Imports & Pathing Setup=====
from config import (
    SRC_DIR,
    PATHING_JSON_PATH,
    SUPPORTED_FEATURE_SETS,
    PP_ANALYSIS_METRICS,
)
from pp_analysis_fns import *

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

v = Visualise(save_all=False)

FULL_PATHING = getPaths(PATHING_JSON_PATH)
RESULTS_DIR = FULL_PATHING["imp_dirs"]["results_dir"]
PREDS_DIR = RESULTS_DIR["prediction_output_dirs"]["rf"]

# %% ===== Argument Parsing =====

parser = argparse.ArgumentParser(description="Generating cross-feature analysis")

parser.add_argument(
    "--properties",
    nargs="+",
    choices=FULL_PATHING["prediction_output_dirs"]["rf"].keys(),
    help=f"Name of the experiments ran.",
)

parser.add_argument(
    "--feature-sets",
    nargs="+",
    default=SUPPORTED_FEATURE_SETS,
    choices=SUPPORTED_FEATURE_SETS,
    help=f"Name of the experiments ran.",
)

parser.add_argument(
    "--save-dir",
    default=str(RESULTS_DIR / "pp_analysis"),
    help="Where to plot the results",
)

args = parser.parse_args()

# %% ===== Loading the Data =====
full_performance_dict = {}
save_path = args.save_dir
save_path.mkdir(parents=True, exist_ok=True)

for prop in args.properties:
    property_pathing = PREDS_DIR[prop]

    int_property_performance_df, ext_property_performance_df = (
        getPropertyPerformanceDfs(
            property_pathing=property_pathing,
            feature_sets=args.feature_sets,
        )
    )

    full_performance_dict[prop] = {
        "internal": int_property_performance_df,
        "external": ext_property_performance_df,
    }

# %% ===== Running Analysis =====
# Creating a bar plots for each property & performance metric
for prop, int_ext_perfs in full_performance_dict.items():
    int_perf = int_ext_perfs["internal"]
    ext_perf = int_ext_perfs["external"]

    for split_name, perf_df in {
        "internal": int_perf,
        "external": ext_perf,
    }.items():

        for metric in PP_ANALYSIS_METRICS["regression"]["bar_metrics"]:

            if metric not in perf_df.columns:
                print(f"{metric} not in {split_name} df columns")
                continue

            mean_df = perf_df.loc[
                perf_df["stat"] == "mean",
                ["feature_set", metric],
            ].copy()

            std_df = perf_df.loc[
                perf_df["stat"] == "std",
                ["feature_set", metric],
            ].copy()

            plot_df = mean_df.merge(
                std_df,
                on="feature_set",
                suffixes=("", "_std"),
            )

            std_metric = f"{metric}_std"

            plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
            plot_df[std_metric] = pd.to_numeric(plot_df[std_metric], errors="coerce")

            plot_df = plot_df.dropna(subset=[metric, std_metric])

            if plot_df.empty:
                print(f"No valid data for {prop} / {split_name} / {metric}")
                continue

            ascending = metric.lower() in ["rmse", "mse", "mae", "bias", "sdep"]
            plot_df = plot_df.sort_values(metric, ascending=ascending)

            v.plotBar(
                data=plot_df,
                x_label="feature_set",
                y_label=metric,
                y_err_label=std_metric,
                title=f"{prop}: {split_name} {metric}",
                save_plot=True,
                save_path=save_path / prop / split_name,
                save_fname=f"{prop}_{split_name}_{metric}_bar",
            )
