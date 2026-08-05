"""
Running individual property prediction (PP) analysis
"""

# %% ===== Python Imports =====
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# %% ===== Project Imports & Pathing Setup=====
from config import (
    SRC_DIR,
    PATHING_JSON_PATH,
    SUPPORTED_FEATURE_SETS,
    PP_ANALYSIS_METRICS,
    TARGET_COLUMNS,
)
from pp_analysis_fns import *

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

v = Visualise(save_all=False)

FULL_PATHING = getPaths(PATHING_JSON_PATH)
RESULTS_DIR = FULL_PATHING["imp_dirs"]["results_dir"]
PREDS_DIR = FULL_PATHING["prediction_output_dirs"]["rf"]

# %% ===== Argument Parsing =====

parser = argparse.ArgumentParser(description="Generating cross-feature analysis")

parser.add_argument(
    "--properties",
    nargs="+",
    default=FULL_PATHING["prediction_output_dirs"]["rf"].keys(),
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
save_path = Path(args.save_dir)
save_path.mkdir(parents=True, exist_ok=True)
feature_sets = args.feature_sets

colour_map = getFeatureColourConfig(feature_ls=feature_sets, colour_map=v.colour_map)


for prop in args.properties:
    try:
        property_pathing = PREDS_DIR[prop]

        int_property_performance_df, ext_property_performance_df = (
            getPropertyPerformanceDfs(
                property_pathing=property_pathing,
                feature_sets=feature_sets,
            )
        )

        full_performance_dict[prop] = {
            "internal": int_property_performance_df,
            "external": ext_property_performance_df,
        }
    except Exception as e:
        print(f"Could not find performance performance for {prop}:\n{e}")

# %% ===== Running Analysis =====
# Creating a bar plots for each property & performance metric
internal_summary_rows = []
external_summary_rows = []
ft_difference_rows = []
summary_metrics = ["r2", "pearson_r", "bias", "sdep"]

# %% ===== Running Analysis =====
# Creating a bar plots for each property & performance metric
for prop, int_ext_perfs in full_performance_dict.items():
    print(f"Processing {prop} performance...")

    for split_name, perf_df in {
        "internal": int_ext_perfs["internal"],
        "external": int_ext_perfs["external"],
    }.items():
        print(f"Processing {split_name}...")

        summary_metric_cols = {
            metric: getMetricColumn(metric=metric, data=perf_df)
            for metric in summary_metrics
        }
        summary_metric_cols = {
            metric: metric_col
            for metric, metric_col in summary_metric_cols.items()
            if metric_col is not None
        }

        if summary_metric_cols:
            summary_df = perf_df.loc[
                perf_df["stat"] == "mean",
                ["feature_set"] + list(summary_metric_cols.values()),
            ].copy()
            summary_df = summary_df.rename(
                columns={
                    metric_col: metric
                    for metric, metric_col in summary_metric_cols.items()
                    if metric_col != metric
                }
            )

            summary_df["property"] = prop
            summary_df["split"] = split_name

            summary_df = summary_df[
                ["property", "split", "feature_set"] + list(summary_metric_cols.keys())
            ]

            if split_name == "internal":
                internal_summary_rows.append(summary_df)
            elif split_name == "external":
                external_summary_rows.append(summary_df)

        for metric in PP_ANALYSIS_METRICS["regression"]["bar_metrics"]:

            metric_col = getMetricColumn(metric=metric, data=perf_df)

            if metric_col is None:
                print(f"{metric} not in {split_name} df columns")
                continue

            mean_df = perf_df.loc[
                perf_df["stat"] == "mean",
                ["feature_set", metric_col],
            ].copy()

            std_df = perf_df.loc[
                perf_df["stat"] == "std",
                ["feature_set", metric_col],
            ].copy()

            plot_df = mean_df.merge(
                std_df,
                on="feature_set",
                suffixes=("", "_std"),
            )

            std_metric_col = f"{metric_col}_std"

            plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
            plot_df[std_metric_col] = pd.to_numeric(
                plot_df[std_metric_col],
                errors="coerce",
            )

            plot_df = plot_df.dropna(subset=[metric_col, std_metric_col])

            if plot_df.empty:
                print(f"No valid data for {prop} / {split_name} / {metric_col}")
                continue

            ascending = metric_col.lower() in ["rmse", "mse", "mae", "bias", "sdep"]
            plot_df = plot_df.sort_values(metric_col, ascending=ascending)

            if metric_col.lower() == "bias" or metric_col.lower() == "rmse":
                y_lim = None
            else:
                y_lim = (0, 1)

            v.plotBar(
                data=plot_df,
                x_label="feature_set",
                y_label=metric_col,
                y_err_label=std_metric_col,
                title=f"{prop}: {split_name} {metric_col}",
                save_plot=True,
                save_path=save_path / prop / split_name,
                save_fname=f"{prop}_{split_name}_{metric_col}_bar",
                colour_map=colour_map,
                y_lims=y_lim,
            )

            ft_diff_df = plotFTDifferenceBar(
                data=perf_df,
                prop=prop,
                split_name=split_name,
                metric_col=metric_col,
                save_path=save_path,
                colour_map=colour_map,
            )
            if not ft_diff_df.empty:
                ft_difference_rows.append(ft_diff_df)

if internal_summary_rows:
    internal_summary_df = pd.concat(internal_summary_rows, ignore_index=True)
    internal_summary_df.to_csv(
        save_path / "internal_average_performances.csv",
        index=False,
    )
    plotSummaryHeatmaps(
        summary_df=internal_summary_df,
        split_name="internal",
        save_path=save_path,
        metrics=summary_metrics,
        feature_order=feature_sets,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=internal_summary_df,
        split_name="internal",
        save_path=save_path,
        metric="r2",
        feature_order=feature_sets,
        colour_map=colour_map,
    )

if external_summary_rows:
    external_summary_df = pd.concat(external_summary_rows, ignore_index=True)
    external_summary_df.to_csv(
        save_path / "external_average_performances.csv",
        index=False,
    )
    plotSummaryHeatmaps(
        summary_df=external_summary_df,
        split_name="external",
        save_path=save_path,
        metrics=summary_metrics,
        feature_order=feature_sets,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=external_summary_df,
        split_name="external",
        save_path=save_path,
        metric="r2",
        feature_order=feature_sets,
        colour_map=colour_map,
    )

if ft_difference_rows:
    ft_difference_df = pd.concat(ft_difference_rows, ignore_index=True)
    ft_difference_df.to_csv(
        save_path / "ft_differences.csv",
        index=False,
    )
    plotFTDifferenceSummaryBars(
        ft_difference_df=ft_difference_df,
        save_path=save_path,
        metrics=["r2", "pearson_r"],
    )

lipinski_external_df = getLipinskiFilteredExternalPerformanceDf(
    properties=args.properties,
    feature_sets=feature_sets,
    full_pathing=FULL_PATHING,
    preds_dir=PREDS_DIR,
    target_columns=TARGET_COLUMNS,
)

if not lipinski_external_df.empty:
    lipinski_external_df.to_csv(
        save_path / "external_lipinski_average_performances.csv",
        index=False,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=lipinski_external_df,
        split_name="external_lipinski",
        save_path=save_path,
        metric="r2",
        feature_order=feature_sets,
        colour_map=colour_map,
    )

iqr_external_df = get3xIQRFilteredExternalPerformanceDf(
    properties=args.properties,
    feature_sets=feature_sets,
    full_pathing=FULL_PATHING,
    preds_dir=PREDS_DIR,
    target_columns=TARGET_COLUMNS,
)

if not iqr_external_df.empty:
    iqr_external_df.to_csv(
        save_path / "external_3xIQR_average_performances.csv",
        index=False,
    )
    for metric in ["r2", "pearson_r", "RMSE", "Bias", "SDEP"]:
        plotGroupedPropertyFeatureBar(
            summary_df=iqr_external_df,
            split_name="external_3xIQR",
            save_path=save_path,
            metric=metric,
            feature_order=feature_sets,
            colour_map=colour_map,
        )
