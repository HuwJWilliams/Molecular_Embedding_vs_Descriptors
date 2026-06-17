"""
Script to run the analysis following the cross-feature predictions
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns   

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = PROJ_DIR / "results"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "pathing")) 
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups
from analyse_datasets import getLowVarianceColumns, plotLowVarianceColumns


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
    help="Flag to run all available cross-prediction analysis"
)

parser.add_argument(
    "--run-experiment",
    nargs="+",
    choices=unique_exp_names,
    help=f"Name of the experiments ran. For full cross-feature predictions:\n{cp}\n" \
    f"For Lipinski cross-feature predictions:\n{lcp}\n"\
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

parser.add_argument(
    "--no-groups",
    action="store_true",
    help="Flag to plot individual features (mostly useful for feature sets without groups)"
)

parser.add_argument(
    "--skip-cols",
    nargs="+",
    help="Feature columns to skip from feature/member bar plots"
)

parser.add_argument(
    "--radar-skip-cols",
    nargs="+",
    help="Feature columns to skip from radar/group averages only",
    default=["Ipc"]
)

parser.add_argument(
    "--radar-task",
    default="regression"
)

parser.add_argument(
    "--run-avg",
    action="store_true",
    help="Flag to average across all specified experiments"
)

parser.add_argument(
    "--plot-poor-distributions",
    action="store_true",
    help="Plot raw descriptor distributions for group members below a performance threshold"
)

parser.add_argument(
    "--poor-distribution-threshold",
    type=float,
    default=0.6,
    help="Performance threshold used with --plot-poor-distributions"
)

# endregion

# region Parsing Arguments
args = parser.parse_args()
cp_dir = paths["prediction_output_dirs"][args.result_dir]
var_threshold = args.var_threshold
TASK_METRICS = {
    "regression": {
        "metric": "Pearson_r",
        "bar_metrics": ["Pearson_r", "r2"],
        "group_metrics": ["Pearson_r", "r2", "RMSE", "Bias"],
        "member_suffix": "reg",
        "radar_metrics": ["avg_Pearson_r", "avg_r2"]
    },
    "classification": {
        "metric": "AUC",
        "bar_metrics": ["AUC", "MCC", "Balanced_Accuracy"],
        "group_metrics": ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "MCC", "Balanced_Accuracy"],
        "member_suffix": "cla",
        "radar_metrics": ["avg_AUC"]
    },
    "multiclass": {
        "metric": "Balanced_Accuracy",
        "bar_metrics": ["AUC_OVR", "MCC", "Balanced_Accuracy"],
        "group_metrics": ["Accuracy", "Balanced_Accuracy", "F1_macro", "AUC_OVR", "MCC"],
        "member_suffix": "mcla",
        "radar_metrics": ["avg_AUC_OVR"]
    },
}
task_type_map = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}


def _available_metrics(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [m for m in wanted if m in df.columns]


def resolve_excluded_cols(index, cols):
    if not cols:
        return []

    labels = [str(label) for label in index]
    resolved = set()
    for col in cols:
        col = str(col)
        resolved.update(label for label in labels if label == col)
        resolved.update(label for label in labels if label.startswith(f"{col}_"))

    return sorted(resolved)


def clean_plot_label(label: str) -> str:
    label = str(label)
    for suffix in ("_rdkit", "_mordred"):
        label = label.replace(suffix, "")
    return label

# endregion

# region Function Definitions prior to running
def avg_results(exp_list):
    dfs = []
    for exp in exp_list:
        exp_df = pd.read_csv(cp_dir[exp] / f"{exp}.csv", index_col=0)
        dfs.append(exp_df)
    
    combined = pd.concat(dfs, axis=0, keys=exp_list)
    avg_df = combined.groupby(level=1).mean(numeric_only=True)
    return avg_df
    
def get_group_perf_per_task(group_map, exp_perf_df, excl_cols):
    group_perf_by_task = {}

    for task_name, task_cfg in TASK_METRICS.items():
        metrics_present = _available_metrics(exp_perf_df, task_cfg["group_metrics"])
        if not metrics_present:
            print(f"No available metrics for {task_name}. Skipping.")
            continue
        gp_df = v.computeGroupPerf(
            data=exp_perf_df,
            descriptor_groups=group_map,
            metrics=metrics_present,
            exclude=excl_cols
        )
        gp_df.to_csv(exp_dir / f"{task_name}_group_perf.csv")
        group_perf_by_task[task_name] = gp_df
    return group_perf_by_task

def plot_group_radar_for_val(group_perf_by_task, task):
    palette = sns.color_palette("tab10")
    c1, c2 = (palette[0], palette[3]) if task == "regression" else (palette[1], palette[4])
    group_perf_df = group_perf_by_task.get(task)
    if group_perf_df is None:
        print(f"Skipping {task} radar for {exp}: no grouped performance table.")
        return

    for val in TASK_METRICS[task]["radar_metrics"]:
        if val not in group_perf_df.columns:
            print(f"Skipping {task} radar for {exp}: missing {val}.")
            continue

        radar_df = group_perf_df[[val]].copy()
        radar_df[val] = pd.to_numeric(radar_df[val], errors="coerce")
        radar_df = radar_df.replace([float("inf"), float("-inf")], pd.NA).dropna(
            subset=[val]
        )
        if radar_df.empty:
            print(f"Skipping {task} radar for {exp}: no finite {val} groups.")
            continue

        gr_title = f"{pred.capitalize()} Prediction ({tr.capitalize()} trained): {val}"
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        gr_description = (
            f"Performance when training RF models on {tr} to predict {pred} features.\n"
            f"Plot shows grouped {val} values.\nCreated: {timestamp}"
        )
        gr_fname_suffix = "excl_low_var" if args.exclude_low_var else "all_features"

        v.plotGroupRadar(
            radar_df,
            title=gr_title,
            save_plot=True,
            save_path=exp_dir,
            save_fname=f"{exp}_{gr_fname_suffix}_group_radar_{task}_{val}",
            metadata={
                "Title": gr_title,
                "Description": gr_description
            },
            c1=c1,
            c2=c2
        )

def plot_members_of_groups(exp_perf_df, group_name, group_members):
    print(f"Plotting performance of individual members for the group {group_name}:")
    present_members = [m for m in group_members if m in exp_perf_df.index]
    if not present_members:
        print(
            f"Skipping group '{group_name}': no members found in performance index."
        )
        return

    title = f"{group_name} (Trained: {tr}, Predicted: {pred})"
    description = f"Performance on {pred.capitalize()} features in the group '{group_name}.\n \
                This group consists of:\n {group_members}"
    
    for task_name, task_cfg in TASK_METRICS.items():
        print(f"Task name: {task_name}")
        task_df = exp_perf_df.copy()
        metric_col = task_cfg["metric"]

        if metric_col not in exp_perf_df.columns:
            continue

        # Prefer task_type filtering, but gracefully fall back to metric-based rows.
        if "task_type" in task_df.columns:
            filtered = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()
            if not filtered.empty and metric_col in filtered.columns and filtered[metric_col].notna().any():
                task_df = filtered
            else:
                task_df = task_df.loc[task_df[metric_col].notna()].copy()
        else:
            task_df = task_df.loc[task_df[metric_col].notna()].copy()

        if task_df.empty:
            continue

        present_members = [m for m in group_members if m in task_df.index]
        print(f"Present members:\n{present_members}")
        finite_vals = pd.to_numeric(
            task_df.loc[present_members, metric_col], errors="coerce"
        ).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if finite_vals.empty:
            print(
                f"Skipping group '{group_name}' for metric '{metric_col}': "
                "no finite values."
            )
            continue
        
        try:
            v.plotMemberBar(
                perf_df=task_df,
                group_map=group_map,
                group_name=group_name,
                value_col=metric_col,
                save_plot=True,
                save_path=exp_dir,
                save_fname=f"{exp}_{group_name}_{pred}_bar_{task_cfg['member_suffix']}",
                metadata={
                    "Title": f"{title} ({metric_col})",
                    "Description": description
                }
            )
        except ValueError as e:
            print(
                f"Skipping group '{group_name}' for metric '{metric_col}' "
                f"due to plotting error: {e}"
            )


def plot_poor_distributions_for_group(exp_perf_df, group_name):
    for task_name, task_cfg in TASK_METRICS.items():
        metric_col = task_cfg["metric"]
        if metric_col not in exp_perf_df.columns:
            continue

        task_df = exp_perf_df.copy()
        if "task_type" in task_df.columns:
            task_df = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()

        if task_df.empty or metric_col not in task_df.columns:
            continue

        task_save_path = exp_dir / f"poor_distributions_{task_name}"
        try:
            v.plotPoorPredictionFeatureDistribution(
                perf_df=task_df,
                full_features=paths["full_features"]["all"][pred],
                group_map=group_map,
                group_name=group_name,
                value_col=metric_col,
                save_plot=True,
                threshold=args.poor_distribution_threshold,
                save_path=task_save_path,
            )
        except Exception as e:
            print(
                f"Skipping poor distributions for group '{group_name}' "
                f"({task_name}, {metric_col}): {e}"
            )

def plot_no_group_bars(exp_perf_df: pd.DataFrame, l_var_col: list[str]) -> None:
    """
    Plot feature-level grouped bars (no groups) for all available task types.

    - Regression: Pearson_r, r2
    - Classification: AUC, MCC, Accuracy
    - Multiclass: F1_macro, Balanced_Accuracy, AUC_OVR
    """
    gr_fname_suffix = "excl_low_var" if args.exclude_low_var else ""

    for task_name, task_cfg in TASK_METRICS.items():
        metric = task_cfg["metric"]
        bar_metrics = [m for m in task_cfg.get("bar_metrics", [metric]) if m in exp_perf_df.columns]
        if not bar_metrics:
            print(f"Skipping {task_name}: none of the requested bar metrics are present")
            continue

        task_df = exp_perf_df.copy()
        original_task_df = task_df.copy()

        # Prefer task_type filtering, but gracefully fall back to metric-based rows.
        if "task_type" in task_df.columns:
            filtered = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()
            if not filtered.empty and filtered[bar_metrics].notna().any().any():
                task_df = filtered
            else:
                task_df = task_df.loc[task_df[bar_metrics].notna().any(axis=1)].copy()

        # Keep low-variance exclusion for regression only.
        # Always apply explicitly skipped columns for every task.
        drop_cols = set(args.skip_cols or [])
        if args.exclude_low_var and task_name == "regression":
            drop_cols.update(l_var_col)
        if drop_cols:
            task_df = task_df.drop(index=[c for c in drop_cols if c in task_df.index], errors="ignore")

        plot_df = task_df.dropna(subset=bar_metrics, how="all").copy()
        # If exclusions removed everything, fall back to non-excluded metric rows.
        if plot_df.empty:
            fallback_df = original_task_df.dropna(subset=bar_metrics, how="all").copy()
            if not fallback_df.empty:
                print(
                    f"{task_name}: exclusions removed all rows; "
                    "falling back to full metric-based plot."
                )
                plot_df = fallback_df
        if plot_df.empty:
            print(f"Skipping {task_name}: no finite values for {bar_metrics} to plot")
            continue

        sort_metric = metric if metric in plot_df.columns else bar_metrics[0]
        plot_df = plot_df.sort_values(by=sort_metric, ascending=False)
        plot_df = plot_df[bar_metrics].apply(pd.to_numeric, errors="coerce")
        plot_df["feature"] = plot_df.index.astype(str)
        long_df = plot_df.melt(
            id_vars="feature",
            value_vars=bar_metrics,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if long_df.empty:
            print(f"Skipping {task_name}: no plottable values after reshaping")
            continue
        long_df["feature_label"] = long_df["feature"].map(clean_plot_label)

        gr_title = (
            f"{pred.capitalize()} Prediction ({tr.capitalize()} trained): "
            f"{', '.join(bar_metrics)}"
        )
        plt.figure(figsize=(16, 6))
        sns.barplot(data=long_df, x="feature_label", y="value", hue="metric")

        # Show descriptor names only when small enough to remain readable.
        if plot_df["feature"].nunique() < 100:
            plt.xticks(
                rotation=90,
                ha="right",
                fontsize=12
            )
        else:
            plt.xticks([])

        plt.ylabel("Performance score")
        plt.title(gr_title)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend(
            title="Performance metric",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()
        plt.savefig(
            exp_dir / f"{exp}_{gr_fname_suffix}_feature_bar_{task_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# endregion

# region Running script functionality
if args.run_all:
    exp_list = list(cp_dir.keys())

elif not args.run_all and bool(args.run_experiment):
    exp_list = args.run_experiment

else:
    raise ValueError("You must set either '--run-all' or specify results with '--run-experiment'"\
                     f"Experiments to choose from:\n{unique_exp_names}")

if args.run_avg:
    wrds = exp_list[0].split("_")
    pred = wrds[1]
    list_to_avg = exp_list
    exp_list=[f"pred_{pred}_tr_avg"]

for exp in exp_list:
    wrds = exp.split("_")
    pred = wrds[1]
    tr = wrds[3]
    exp_dir = cp_dir[exp]

    if args.run_avg:
        exp_perf_df = avg_results(list_to_avg)
    else:
        exp_perf_df_path = exp_dir / f"{exp}.csv"
        exp_perf_df = pd.read_csv(Path(exp_perf_df_path), index_col=0)

    pred_ft_df = Path(paths["full_features"]["all"][pred])
    
    l_var_col = getLowVarianceColumns(
        pred_ft_df, threshold=var_threshold
        )
    
    excl_cols = l_var_col if args.exclude_low_var else []
    if args.skip_cols:
        excl_cols.extend(resolve_excluded_cols(exp_perf_df.index, args.skip_cols))
    radar_excl_cols = list(excl_cols)
    radar_excl_cols.extend(resolve_excluded_cols(exp_perf_df.index, args.radar_skip_cols))
    
    if args.show_var:
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


    if not args.no_groups:
        group_map=getGroups(pred)
        group_perf_by_task=get_group_perf_per_task(group_map=group_map, exp_perf_df=exp_perf_df, excl_cols=radar_excl_cols)
        plot_group_radar_for_val(group_perf_by_task=group_perf_by_task, task=args.radar_task)
        

        for group_name, group_members in group_map.items():
            plot_members_of_groups(
                group_name=group_name, group_members=group_members, exp_perf_df=exp_perf_df
                )
            if args.plot_poor_distributions:
                plot_poor_distributions_for_group(
                    exp_perf_df=exp_perf_df,
                    group_name=group_name,
                )

    # --- Plotting without groups (always run for all available task metrics)
    plot_no_group_bars(exp_perf_df=exp_perf_df, l_var_col=l_var_col)


# endregion


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
    help="Flag to run all available cross-prediction analysis"
)

parser.add_argument(
    "--run-experiment",
    nargs="+",
    choices=unique_exp_names,
    help=f"Name of the experiments ran. For full cross-feature predictions:\n{cp}\n" \
    f"For Lipinski cross-feature predictions:\n{lcp}\n"\
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

parser.add_argument(
    "--no-groups",
    action="store_true",
    help="Flag to plot individual features (mostly useful for feature sets without groups)"
)

parser.add_argument(
    "--skip-cols",
    nargs="+",
    help="Feature columns to skip from feature/member bar plots"
)

parser.add_argument(
    "--radar-skip-cols",
    nargs="+",
    help="Feature columns to skip from radar/group averages only",
    default=["Ipc"]
)

parser.add_argument(
    "--radar-task",
    default="regression"
)

parser.add_argument(
    "--run-avg",
    action="store_true",
    help="Flag to average across all specified experiments"
)

parser.add_argument(
    "--plot-poor-distributions",
    action="store_true",
    help="Plot raw descriptor distributions for group members below a performance threshold"
)

parser.add_argument(
    "--poor-distribution-threshold",
    type=float,
    default=0.6,
    help="Performance threshold used with --plot-poor-distributions"
)

# endregion

# region Parsing Arguments
args = parser.parse_args()
cp_dir = paths["prediction_output_dirs"][args.result_dir]
var_threshold = args.var_threshold
TASK_METRICS = {
    "regression": {
        "metric": "Pearson_r",
        "bar_metrics": ["Pearson_r", "r2"],
        "group_metrics": ["Pearson_r", "r2", "RMSE", "Bias"],
        "member_suffix": "reg",
        "radar_metrics": ["avg_Pearson_r", "avg_r2"]
    },
    "classification": {
        "metric": "AUC",
        "bar_metrics": ["AUC", "MCC", "Balanced_Accuracy"],
        "group_metrics": ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "AUC", "MCC", "Balanced_Accuracy"],
        "member_suffix": "cla",
        "radar_metrics": ["avg_AUC"]
    },
    "multiclass": {
        "metric": "Balanced_Accuracy",
        "bar_metrics": ["AUC_OVR", "MCC", "Balanced_Accuracy"],
        "group_metrics": ["Accuracy", "Balanced_Accuracy", "F1_macro", "AUC_OVR", "MCC"],
        "member_suffix": "mcla",
        "radar_metrics": ["avg_AUC_OVR"]
    },
}
task_type_map = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}


def _available_metrics(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [m for m in wanted if m in df.columns]


def resolve_excluded_cols(index, cols):
    if not cols:
        return []

    labels = [str(label) for label in index]
    resolved = set()
    for col in cols:
        col = str(col)
        resolved.update(label for label in labels if label == col)
        resolved.update(label for label in labels if label.startswith(f"{col}_"))

    return sorted(resolved)


def clean_plot_label(label: str) -> str:
    label = str(label)
    for suffix in ("_rdkit", "_mordred"):
        label = label.replace(suffix, "")
    return label

# endregion

# region Function Definitions prior to running
def avg_results(exp_list):
    dfs = []
    for exp in exp_list:
        exp_df = pd.read_csv(cp_dir[exp] / f"{exp}.csv", index_col=0)
        dfs.append(exp_df)
    
    combined = pd.concat(dfs, axis=0, keys=exp_list)
    avg_df = combined.groupby(level=1).mean(numeric_only=True)
    return avg_df
    
def get_group_perf_per_task(group_map, exp_perf_df, excl_cols):
    group_perf_by_task = {}

    for task_name, task_cfg in TASK_METRICS.items():
        metrics_present = _available_metrics(exp_perf_df, task_cfg["group_metrics"])
        if not metrics_present:
            print(f"No available metrics for {task_name}. Skipping.")
            continue
        gp_df = v.computeGroupPerf(
            data=exp_perf_df,
            descriptor_groups=group_map,
            metrics=metrics_present,
            exclude=excl_cols
        )
        gp_df.to_csv(exp_dir / f"{task_name}_group_perf.csv")
        group_perf_by_task[task_name] = gp_df
    return group_perf_by_task

def plot_group_radar_for_val(group_perf_by_task, task):
    palette = sns.color_palette("tab10")
    c1, c2 = (palette[0], palette[3]) if task == "regression" else (palette[1], palette[4])
    group_perf_df = group_perf_by_task.get(task)
    if group_perf_df is None:
        print(f"Skipping {task} radar for {exp}: no grouped performance table.")
        return

    for val in TASK_METRICS[task]["radar_metrics"]:
        if val not in group_perf_df.columns:
            print(f"Skipping {task} radar for {exp}: missing {val}.")
            continue

        radar_df = group_perf_df[[val]].copy()
        radar_df[val] = pd.to_numeric(radar_df[val], errors="coerce")
        radar_df = radar_df.replace([float("inf"), float("-inf")], pd.NA).dropna(
            subset=[val]
        )
        if radar_df.empty:
            print(f"Skipping {task} radar for {exp}: no finite {val} groups.")
            continue

        gr_title = f"{pred.capitalize()} Prediction ({tr.capitalize()} trained): {val}"
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        gr_description = (
            f"Performance when training RF models on {tr} to predict {pred} features.\n"
            f"Plot shows grouped {val} values.\nCreated: {timestamp}"
        )
        gr_fname_suffix = "excl_low_var" if args.exclude_low_var else "all_features"

        v.plotGroupRadar(
            radar_df,
            title=gr_title,
            save_plot=True,
            save_path=exp_dir,
            save_fname=f"{exp}_{gr_fname_suffix}_group_radar_{task}_{val}",
            metadata={
                "Title": gr_title,
                "Description": gr_description
            },
            c1=c1,
            c2=c2
        )

def plot_members_of_groups(exp_perf_df, group_name, group_members):
    print(f"Plotting performance of individual members for the group {group_name}:")
    present_members = [m for m in group_members if m in exp_perf_df.index]
    if not present_members:
        print(
            f"Skipping group '{group_name}': no members found in performance index."
        )
        return

    title = f"{group_name} (Trained: {tr}, Predicted: {pred})"
    description = f"Performance on {pred.capitalize()} features in the group '{group_name}.\n \
                This group consists of:\n {group_members}"
    
    for task_name, task_cfg in TASK_METRICS.items():
        print(f"Task name: {task_name}")
        task_df = exp_perf_df.copy()
        metric_col = task_cfg["metric"]

        if metric_col not in exp_perf_df.columns:
            continue

        # Prefer task_type filtering, but gracefully fall back to metric-based rows.
        if "task_type" in task_df.columns:
            filtered = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()
            if not filtered.empty and metric_col in filtered.columns and filtered[metric_col].notna().any():
                task_df = filtered
            else:
                task_df = task_df.loc[task_df[metric_col].notna()].copy()
        else:
            task_df = task_df.loc[task_df[metric_col].notna()].copy()

        if task_df.empty:
            continue

        present_members = [m for m in group_members if m in task_df.index]
        print(f"Present members:\n{present_members}")
        finite_vals = pd.to_numeric(
            task_df.loc[present_members, metric_col], errors="coerce"
        ).replace([float("inf"), float("-inf")], pd.NA).dropna()
        if finite_vals.empty:
            print(
                f"Skipping group '{group_name}' for metric '{metric_col}': "
                "no finite values."
            )
            continue
        
        try:
            v.plotMemberBar(
                perf_df=task_df,
                group_map=group_map,
                group_name=group_name,
                value_col=metric_col,
                save_plot=True,
                save_path=exp_dir,
                save_fname=f"{exp}_{group_name}_{pred}_bar_{task_cfg['member_suffix']}",
                metadata={
                    "Title": f"{title} ({metric_col})",
                    "Description": description
                }
            )
        except ValueError as e:
            print(
                f"Skipping group '{group_name}' for metric '{metric_col}' "
                f"due to plotting error: {e}"
            )


def plot_poor_distributions_for_group(exp_perf_df, group_name):
    for task_name, task_cfg in TASK_METRICS.items():
        metric_col = task_cfg["metric"]
        if metric_col not in exp_perf_df.columns:
            continue

        task_df = exp_perf_df.copy()
        if "task_type" in task_df.columns:
            task_df = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()

        if task_df.empty or metric_col not in task_df.columns:
            continue

        task_save_path = exp_dir / f"poor_distributions_{task_name}"
        try:
            v.plotPoorPredictionFeatureDistribution(
                perf_df=task_df,
                full_features=paths["full_features"]["all"][pred],
                group_map=group_map,
                group_name=group_name,
                value_col=metric_col,
                save_plot=True,
                threshold=args.poor_distribution_threshold,
                save_path=task_save_path,
            )
        except Exception as e:
            print(
                f"Skipping poor distributions for group '{group_name}' "
                f"({task_name}, {metric_col}): {e}"
            )

def plot_no_group_bars(exp_perf_df: pd.DataFrame, l_var_col: list[str]) -> None:
    """
    Plot feature-level grouped bars (no groups) for all available task types.

    - Regression: Pearson_r, r2
    - Classification: AUC, MCC, Accuracy
    - Multiclass: F1_macro, Balanced_Accuracy, AUC_OVR
    """
    gr_fname_suffix = "excl_low_var" if args.exclude_low_var else ""

    for task_name, task_cfg in TASK_METRICS.items():
        metric = task_cfg["metric"]
        bar_metrics = [m for m in task_cfg.get("bar_metrics", [metric]) if m in exp_perf_df.columns]
        if not bar_metrics:
            print(f"Skipping {task_name}: none of the requested bar metrics are present")
            continue

        task_df = exp_perf_df.copy()
        original_task_df = task_df.copy()

        # Prefer task_type filtering, but gracefully fall back to metric-based rows.
        if "task_type" in task_df.columns:
            filtered = task_df.loc[task_df["task_type"] == task_type_map[task_name]].copy()
            if not filtered.empty and filtered[bar_metrics].notna().any().any():
                task_df = filtered
            else:
                task_df = task_df.loc[task_df[bar_metrics].notna().any(axis=1)].copy()

        # Keep low-variance exclusion for regression only.
        # Always apply explicitly skipped columns for every task.
        drop_cols = set(args.skip_cols or [])
        if args.exclude_low_var and task_name == "regression":
            drop_cols.update(l_var_col)
        if drop_cols:
            task_df = task_df.drop(index=[c for c in drop_cols if c in task_df.index], errors="ignore")

        plot_df = task_df.dropna(subset=bar_metrics, how="all").copy()
        # If exclusions removed everything, fall back to non-excluded metric rows.
        if plot_df.empty:
            fallback_df = original_task_df.dropna(subset=bar_metrics, how="all").copy()
            if not fallback_df.empty:
                print(
                    f"{task_name}: exclusions removed all rows; "
                    "falling back to full metric-based plot."
                )
                plot_df = fallback_df
        if plot_df.empty:
            print(f"Skipping {task_name}: no finite values for {bar_metrics} to plot")
            continue

        sort_metric = metric if metric in plot_df.columns else bar_metrics[0]
        plot_df = plot_df.sort_values(by=sort_metric, ascending=False)
        plot_df = plot_df[bar_metrics].apply(pd.to_numeric, errors="coerce")
        plot_df["feature"] = plot_df.index.astype(str)
        long_df = plot_df.melt(
            id_vars="feature",
            value_vars=bar_metrics,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if long_df.empty:
            print(f"Skipping {task_name}: no plottable values after reshaping")
            continue
        long_df["feature_label"] = long_df["feature"].map(clean_plot_label)

        gr_title = (
            f"{pred.capitalize()} Prediction ({tr.capitalize()} trained): "
            f"{', '.join(bar_metrics)}"
        )
        plt.figure(figsize=(16, 6))
        sns.barplot(data=long_df, x="feature_label", y="value", hue="metric")

        # Show descriptor names only when small enough to remain readable.
        if plot_df["feature"].nunique() < 100:
            plt.xticks(
                rotation=90,
                ha="right",
                fontsize=12
            )
        else:
            plt.xticks([])

        plt.ylabel("Performance score")
        plt.title(gr_title)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend(
            title="Performance metric",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
        plt.tight_layout()
        plt.savefig(
            exp_dir / f"{exp}_{gr_fname_suffix}_feature_bar_{task_name}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


# endregion

# region Running script functionality
if args.run_all:
    exp_list = list(cp_dir.keys())

elif not args.run_all and bool(args.run_experiment):
    exp_list = args.run_experiment

else:
    raise ValueError("You must set either '--run-all' or specify results with '--run-experiment'"\
                     f"Experiments to choose from:\n{unique_exp_names}")

if args.run_avg:
    wrds = exp_list[0].split("_")
    pred = wrds[1]
    list_to_avg = exp_list
    exp_list=[f"pred_{pred}_tr_avg"]

for exp in exp_list:
    wrds = exp.split("_")
    pred = wrds[1]
    tr = wrds[3]
    exp_dir = cp_dir[exp]

    if args.run_avg:
        exp_perf_df = avg_results(list_to_avg)
    else:
        exp_perf_df_path = exp_dir / f"{exp}.csv"
        exp_perf_df = pd.read_csv(Path(exp_perf_df_path), index_col=0)

    pred_ft_df = Path(paths["full_features"]["all"][pred])
    
    l_var_col = getLowVarianceColumns(
        pred_ft_df, threshold=var_threshold
        )
    
    excl_cols = l_var_col if args.exclude_low_var else []
    if args.skip_cols:
        excl_cols.extend(resolve_excluded_cols(exp_perf_df.index, args.skip_cols))
    radar_excl_cols = list(excl_cols)
    radar_excl_cols.extend(resolve_excluded_cols(exp_perf_df.index, args.radar_skip_cols))
    
    if args.show_var:
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


    if not args.no_groups:
        group_map=getGroups(pred)
        group_perf_by_task=get_group_perf_per_task(group_map=group_map, exp_perf_df=exp_perf_df, excl_cols=radar_excl_cols)
        plot_group_radar_for_val(group_perf_by_task=group_perf_by_task, task=args.radar_task)
        

        for group_name, group_members in group_map.items():
            plot_members_of_groups(
                group_name=group_name, group_members=group_members, exp_perf_df=exp_perf_df
                )
            if args.plot_poor_distributions:
                plot_poor_distributions_for_group(
                    exp_perf_df=exp_perf_df,
                    group_name=group_name,
                )

    # --- Plotting without groups (always run for all available task metrics)
    plot_no_group_bars(exp_perf_df=exp_perf_df, l_var_col=l_var_col)


# endregion
