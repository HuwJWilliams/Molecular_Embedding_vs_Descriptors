"""
Functions for Cross-Feature Prediction (CFP) Analysis
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# %% ===== Project Imports & Pathing Setup =====
from config import PATHING_JSON_PATH, SRC_DIR, CFP_ANALYSIS_METRICS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups
from analyse_datasets import getLowVarianceColumns

v = Visualise(save_all=False)


# %% ===== Function Definitions =====
def getCFPSavePath(
    full_pathing: dict,
    save_dir: str,
    identifier: str,
    property_name: str | None = None,
) -> Path:
    if property_name is None:
        return Path(full_pathing["prediction_output_dirs"][save_dir][identifier])

    property_rf_paths = full_pathing["prediction_output_dirs"]["rf"][property_name]
    first_feature_path = Path(next(iter(property_rf_paths.values())))

    # Walk up to the *_predictions_rf directory.
    prediction_root = next(
        parent for parent in [first_feature_path, *first_feature_path.parents]
        if parent.name.endswith("_predictions_rf")
    )

    return prediction_root / save_dir / identifier

def getCFPAnalysisDir(
    full_pathing: dict,
    result_dir: str,
    exp: str,
    property_name: str | None = None,
) -> Path:
    results_dir = Path(full_pathing["imp_dirs"]["results_dir"])

    if property_name is None:
        return Path(full_pathing["prediction_output_dirs"][result_dir][exp])

    return results_dir / result_dir / property_name / exp

def plotGroupRadar(
    group_perf_by_task: dict[str, pd.DataFrame],
    plot_dir: str | Path,
    exp_name: str,
    task_type: str = "regression",
):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_parts = exp_name.split("_")
    pred = exp_parts[1] if len(exp_parts) > 1 else "unknown"
    tr = exp_parts[3] if len(exp_parts) > 3 else "unknown"

    palette = sns.color_palette("tab10")
    c1, c2 = (
        (palette[0], palette[3])
        if task_type == "regression"
        else (palette[1], palette[4])
    )

    group_perf_df = group_perf_by_task.get(task_type)

    if group_perf_df is None:
        print(f"Skipping {task_type} radar: no grouped performance table.")
        return

    for val in CFP_ANALYSIS_METRICS[task_type]["radar_metrics"]:
        if val not in group_perf_df.columns:
            print(f"Skipping radar metric '{val}': column not found.")
            continue

        radar_df = group_perf_df[[val]].copy()
        radar_df[val] = pd.to_numeric(radar_df[val], errors="coerce")
        radar_df = radar_df.replace([float("inf"), float("-inf")], pd.NA)
        radar_df = radar_df.dropna(subset=[val])

        if radar_df.empty:
            print(f"Skipping radar metric '{val}': no finite values.")
            continue

        gr_title = (
            f"{pred.capitalize()} Prediction " f"({tr.capitalize()} trained): {val}"
        )

        v.plotGroupRadar(
            radar_df,
            title=gr_title,
            save_plot=True,
            save_path=plot_dir,
            save_fname=f"{exp_name}_group_radar_{task_type}_{val}",
            c1=c1,
            c2=c2,
        )

    return


def getTaskGroupPerf(
    df: pd.DataFrame,
    group_map: dict[str, list[str]],
    excl_cols: list[str],
    save_dir: str | Path,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    group_perf_by_task = {}

    for task_name, task_cfg in CFP_ANALYSIS_METRICS.items():
        metrics = task_cfg["group_metrics"]

        gp_df = v.computeGroupPerf(
            data=df,
            descriptor_groups=group_map,
            metrics=metrics,
            exclude=excl_cols,
        )

        gp_df.to_csv(save_dir / f"{task_name}_group_perf.csv")
        group_perf_by_task[task_name] = gp_df

    return group_perf_by_task


def plotGroupMemberBars(
    df: pd.DataFrame,
    group_name: str,
    group_members: list[str],
    exp_name: str,
    group_map: dict[str, list[str]],
    plot_dir: str | Path,
):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_parts = exp_name.split("_")
    pred = exp_parts[1] if len(exp_parts) > 1 else "unknown"
    tr = exp_parts[3] if len(exp_parts) > 3 else "unknown"

    title = (
        f"{group_name} " f"(Trained: {tr.capitalize()}, Predicted: {pred.capitalize()})"
    )

    for task_name, task_cfg in CFP_ANALYSIS_METRICS.items():
        print(f"Task: {task_name}")

        if "task_type" not in df.columns:
            print("Skipping: dataframe does not contain a 'task_type' column.")
            continue

        task_df = df.loc[df["task_type"] == task_name].copy()

        if task_df.empty:
            print(f"Skipping task '{task_name}': no rows found.")
            continue

        metric_col = task_cfg["metric"]

        if metric_col not in task_df.columns:
            print(f"Skipping task '{task_name}': metric '{metric_col}' not found.")
            continue

        present_members = [m for m in group_members if m in task_df.index]
        print(f"Present members:\n{present_members}")

        if not present_members:
            print(
                f"Skipping group '{group_name}': no members found in dataframe index."
            )
            continue

        finite_vals = (
            pd.to_numeric(task_df.loc[present_members, metric_col], errors="coerce")
            .replace([float("inf"), float("-inf")], pd.NA)
            .dropna()
        )

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
                title=title,
                save_plot=True,
                save_path=plot_dir,
                save_fname=(
                    f"{exp_name}_{group_name}_{pred}_bar_"
                    f"{task_cfg['member_suffix']}"
                ),
            )

        except ValueError as e:
            print(
                f"Skipping group '{group_name}' for metric '{metric_col}' "
                f"due to plotting error: {e}"
            )


def plotFullTaskBar(
    df: pd.DataFrame,
    plot_dir: str | Path,
    exp_name: str,
):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    exp_parts = exp_name.split("_")
    pred = exp_parts[1] if len(exp_parts) > 1 else "unknown"
    tr = exp_parts[3] if len(exp_parts) > 3 else "unknown"

    for task_name, task_cfg in CFP_ANALYSIS_METRICS.items():
        print(f"Task: {task_name}")

        if "task_type" not in df.columns:
            print("Skipping: dataframe does not contain a 'task_type' column.")
            continue

        task_df = df.loc[df["task_type"] == task_name].copy()

        if task_df.empty:
            print(f"Skipping task '{task_name}': no rows found.")
            continue

        metric_col = task_cfg["metric"]

        if metric_col not in task_df.columns:
            print(f"Skipping task '{task_name}': metric '{metric_col}' not found.")
            continue

        task_df[metric_col] = pd.to_numeric(task_df[metric_col], errors="coerce")
        task_df = task_df.replace([float("inf"), float("-inf")], pd.NA)
        task_df = task_df.dropna(subset=[metric_col])

        if task_df.empty:
            print(f"Skipping task '{task_name}': no finite values for {metric_col}.")
            continue

        task_df = task_df.sort_values(metric_col, ascending=False)

        title = (
            f"Full {task_name.capitalize()} Task Performance "
            f"(Trained: {tr.capitalize()}, Predicted: {pred.capitalize()})"
        )

        plt.figure(figsize=(12, 6))
        plt.bar(task_df.index.astype(str), task_df[metric_col])
        plt.xlabel("Feature")
        plt.ylabel(metric_col)
        plt.title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        save_path = plot_dir / f"{exp_name}_full_task_bar_{task_name}_{metric_col}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved full task bar to: {save_path}")


def plotGroupTaskFractionSummary(
    df: pd.DataFrame,
    group_map: dict[str, list[str]],
    exp_name: str,
    plot_dir: str | Path,
    threshold: float = 0.7,
    excl_cols: list[str] | None = None,
):
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    excl_cols = set(excl_cols or [])
    rows = []

    for task_name, task_cfg in CFP_ANALYSIS_METRICS.items():
        metric = task_cfg["metric"]

        if "task_type" not in df.columns:
            print("Skipping group task fraction summary: no task_type column.")
            continue

        task_df = df.loc[df["task_type"] == task_name].copy()

        if task_df.empty:
            print(f"Skipping group task fraction summary for {task_name}: no rows.")
            continue

        if metric not in task_df.columns:
            print(f"Skipping group task fraction summary for {task_name}: no {metric}.")
            continue

        values = pd.to_numeric(task_df[metric], errors="coerce")
        values = values.replace([float("inf"), float("-inf")], pd.NA)

        valid_descriptors = [
            idx
            for idx in task_df.index
            if str(idx) not in excl_cols and pd.notna(values.loc[idx])
        ]

        total_task_descriptors = len(valid_descriptors)

        if total_task_descriptors == 0:
            print(
                f"Skipping group task fraction summary for {task_name}: "
                "no valid descriptors."
            )
            continue

        valid_descriptor_index = pd.Index(valid_descriptors)

        for group_name, group_members in group_map.items():
            present_members = [
                desc for desc in group_members if desc in valid_descriptor_index
            ]

            if not present_members:
                continue

            member_values = values.reindex(present_members)

            n_group_task = len(present_members)
            n_low = int((member_values <= threshold).sum())
            n_high = n_group_task - n_low

            rows.append(
                {
                    "task": task_name,
                    "task_type": task_name,
                    "metric": metric,
                    "descriptor_group": group_name,
                    "n_total_task_descriptors": total_task_descriptors,
                    "n_group_task_descriptors": n_group_task,
                    "n_group_low": n_low,
                    "n_group_high": n_high,
                    "fraction_group_total": n_group_task / total_task_descriptors,
                    "fraction_group_low": n_low / total_task_descriptors,
                    "fraction_group_high": n_high / total_task_descriptors,
                    "low_threshold": threshold,
                }
            )

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        print("No group-task fraction summary data generated.")
        return summary_df

    summary_df = summary_df.sort_values(
        ["task", "fraction_group_total"],
        ascending=[True, False],
    )

    threshold_label = str(threshold).replace(".", "p")
    summary_path = (
        plot_dir / f"{exp_name}_group_fraction_lt{threshold_label}_summary.csv"
    )
    summary_df.to_csv(summary_path, index=False)

    print(f"Saved group-task fraction summary to: {summary_path}")

    v.plotGroupTaskFractionBars(
        summary_df=summary_df,
        save_path=plot_dir,
        threshold=threshold,
        save_plot=True,
        show_plot=False,
        fname_prefix=f"{exp_name}_group_task_fraction",
    )

    return summary_df


def averageExperimentPerformance(
    exp_list: list[str],
    cfp_dir: dict[str, str | Path],
):
    """
    Average performance tables across multiple CFP experiments.

    Assumes experiment names follow:
        ..._{pred}_..._{train}_...

    Saves:
        pred_{pred}_tr_avg/pred_{pred}_tr_avg.csv
        pred_{pred}_tr_avg/averaged_feature_sets.txt
    """

    if not exp_list:
        raise ValueError("No experiments provided for averaging.")

    exp_parts = [exp.split("_") for exp in exp_list]

    preds = [parts[1] if len(parts) > 1 else "unknown" for parts in exp_parts]
    trains = [parts[3] if len(parts) > 3 else "unknown" for parts in exp_parts]

    unique_preds = sorted(set(preds))

    if len(unique_preds) != 1:
        raise ValueError(
            "Cannot average experiments with different prediction targets. "
            f"Found predicted feature sets: {unique_preds}"
        )

    pred = unique_preds[0]
    avg_exp_name = f"pred_{pred}_tr_avg"

    first_exp_dir = Path(cfp_dir[exp_list[0]])
    avg_dir = first_exp_dir.parent / avg_exp_name
    avg_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for exp in exp_list:
        exp_dir = Path(cfp_dir[exp])
        exp_path = exp_dir / f"{exp}.csv"

        if not exp_path.exists():
            raise FileNotFoundError(f"Could not find experiment file: {exp_path}")

        exp_df = pd.read_csv(exp_path, index_col=0)
        dfs.append(exp_df)

    combined_df = pd.concat(
        dfs,
        keys=exp_list,
        names=["experiment", "descriptor"],
    )

    numeric_cols = combined_df.select_dtypes(include="number").columns.tolist()
    non_numeric_cols = [col for col in combined_df.columns if col not in numeric_cols]

    avg_numeric_df = combined_df.groupby(level="descriptor")[numeric_cols].mean()

    def _first_non_null(series: pd.Series):
        series = series.dropna()
        return series.iloc[0] if not series.empty else pd.NA

    if non_numeric_cols:
        avg_non_numeric_df = combined_df.groupby(level="descriptor")[
            non_numeric_cols
        ].agg(_first_non_null)
        avg_df = avg_numeric_df.join(avg_non_numeric_df)
    else:
        avg_df = avg_numeric_df

    # Restore column order as much as possible
    original_cols = dfs[0].columns.tolist()
    ordered_cols = [col for col in original_cols if col in avg_df.columns]
    avg_df = avg_df[ordered_cols]

    avg_csv_path = avg_dir / f"{avg_exp_name}.csv"
    avg_df.to_csv(avg_csv_path)

    txt_path = avg_dir / "averaged_feature_sets.txt"

    with open(txt_path, "w") as f:
        f.write(f"Averaged experiment name: {avg_exp_name}\n")
        f.write(f"Predicted feature set: {pred}\n\n")

        f.write("Training feature sets averaged:\n")
        for train in trains:
            f.write(f"- {train}\n")

        f.write("\nExperiments averaged:\n")
        for exp in exp_list:
            f.write(f"- {exp}\n")

    print(f"Saved averaged performance table to: {avg_csv_path}")
    print(f"Saved averaged feature-set list to: {txt_path}")

    return avg_df, avg_dir, avg_exp_name, pred, trains


def runCFPAnalysisForPerformanceDF(
    exp_perf_df: pd.DataFrame,
    pred_ft_df: pd.DataFrame,
    pred: str,
    exp_name: str,
    exp_dir: str | Path,
    args,
):
    exp_dir = Path(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Getting target columns with low variance or non-numeric values
    low_var_cols = getLowVarianceColumns(
        pred_ft_df,
        threshold=args.var_threshold,
    )

    non_numeric_cols = pred_ft_df.select_dtypes(exclude="number").columns.tolist()

    excl_cols = low_var_cols.copy() if args.exclude_low_var else []

    excl_cols.extend(col for col in non_numeric_cols if col not in excl_cols)

    if args.skip_cols:
        excl_cols.extend(col for col in args.skip_cols if col not in excl_cols)

    with open(exp_dir / "low_variance_columns.txt", "w") as f:
        for col in low_var_cols:
            f.write(f"{col}\n")

    with open(exp_dir / "excluded_columns.txt", "w") as f:
        for col in excl_cols:
            f.write(f"{col}\n")

    group_map = getGroups(pred)

    group_perf_by_task = getTaskGroupPerf(
        df=exp_perf_df,
        group_map=group_map,
        excl_cols=excl_cols,
        save_dir=exp_dir,
    )

    plotGroupRadar(
        group_perf_by_task=group_perf_by_task,
        plot_dir=exp_dir,
        exp_name=exp_name,
        task_type=args.radar_task,
    )

    for group_name, group_members in group_map.items():
        plotGroupMemberBars(
            df=exp_perf_df,
            group_members=group_members,
            group_name=group_name,
            group_map=group_map,
            plot_dir=exp_dir,
            exp_name=exp_name,
        )

    plotFullTaskBar(
        df=exp_perf_df,
        plot_dir=exp_dir,
        exp_name=exp_name,
    )

    plotGroupTaskFractionSummary(
        avg_df=exp_perf_df,
        group_map=group_map,
        plot_dir=exp_dir,
        threshold=0.7,
        exclude_cols=excl_cols,
        fname_prefix=f"{exp_name}_group_task_fraction",
    )

    return excl_cols
