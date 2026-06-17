# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import joblib
from glob import glob
from datetime import datetime

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
from group_descriptors import getGroups
from analyse_datasets import getLowVarianceColumns

# from analyse_datasets import plotDescriptorAnalysis

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
from vis import Visualise

paths = getPaths()
v = Visualise()

pred = "mordred"
cap_pred = "RDKit" if pred == "rdkit" else pred.capitalize()
exclude_low_var = True
var_threshold = 0.8

tr2avg_ls = ["chemberta", "molformer", "molformer-c3-1b", "selformer", "chembertasey"]

tr_ls = ["rdkit", "mordred", "morgan"]

full_tr_ls = tr2avg_ls + tr_ls

# endregion
results_dir = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
avg_results_df_paths = [
    results_dir[f"pred_{pred}_tr_{tr_avg}"] / f"pred_{pred}_tr_{tr_avg}.csv"
    for tr_avg in tr2avg_ls
]

other_results_df_paths = [
    results_dir[f"pred_{pred}_tr_{tr}"] / f"pred_{pred}_tr_{tr}.csv" for tr in tr_ls
]

exp_keys = [f"pred_{pred}_tr_{tr}" for tr in full_tr_ls]

all_paths = avg_results_df_paths + other_results_df_paths

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
        "group_metrics": [
            "Accuracy",
            "Balanced_Accuracy",
            "F1_macro",
            "AUC_OVR",
            "MCC",
        ],
        "member_suffix": "mcla",
        "radar_metrics": ["avg_AUC_OVR"],
    },
}
TASK_TYPE_MAP = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}


def clean_plot_label(label: str) -> str:
    label = str(label)
    for suffix in ("_rdkit", "_mordred"):
        label = label.replace(suffix, "")
    return label


def _available_metrics(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    return [m for m in wanted if m in df.columns]


group_map = getGroups(pred)
pred_ft_df = Path(paths["full_features"]["all"][pred])
l_var_col = getLowVarianceColumns(pred_ft_df, threshold=var_threshold)
excl_cols = l_var_col if exclude_low_var else []

gr_fname_suffix = "excl_low_var" if exclude_low_var else ""

save_dir = (
    paths["imp_dirs"]["results_dir"]
    / "lipinski_embeddings_and_descriptor_predictions"
    / f"pred_{pred}_tr_avg_emb"
)
save_dir.mkdir(parents=True, exist_ok=True)


def _build_task_merged(metric: str, task_name: str) -> pd.DataFrame | None:
    loaded = []
    for p, tr_desc in zip(all_paths, full_tr_ls):
        if not Path(p).exists():
            continue
        df = pd.read_csv(p, index_col=0)
        if metric not in df.columns:
            continue

        task_df = df.copy()
        if "task_type" in task_df.columns:
            filtered = task_df.loc[
                task_df["task_type"] == TASK_TYPE_MAP[task_name]
            ].copy()
            if not filtered.empty and filtered[metric].notna().any():
                task_df = filtered
            else:
                task_df = task_df.loc[task_df[metric].notna()].copy()
        else:
            task_df = task_df.loc[task_df[metric].notna()].copy()

        if task_df.empty:
            continue

        loaded.append(task_df[[metric]].rename(columns={metric: tr_desc}))

    if not loaded:
        return None

    merged = pd.concat(loaded, axis=1)
    avg_col = f"Avg_{metric}_Embeddings"
    present_embed_cols = [c for c in tr2avg_ls if c in merged.columns]
    if not present_embed_cols:
        return None
    merged[avg_col] = merged[present_embed_cols].mean(axis=1, skipna=True)
    return merged


def plot_task_bars(
    merged: pd.DataFrame,
    task_name: str,
    task_cfg: dict,
) -> None:
    metric = task_cfg["metric"]
    bar_metrics = task_cfg.get("bar_metrics", [metric])
    avg_bar_cols = [
        f"Avg_{m}_Embeddings"
        for m in bar_metrics
        if f"Avg_{m}_Embeddings" in merged.columns
    ]

    if not avg_bar_cols:
        print(f"Skipping {task_name} bar plot: no averaged bar metrics present.")
        return

    sort_metric = avg_bar_cols[0]
    plot_df = merged.dropna(subset=avg_bar_cols, how="all").copy()
    if plot_df.empty:
        print(f"Skipping {task_name} bar plot: no finite values.")
        return

    plot_df = plot_df.sort_values(by=sort_metric, ascending=False)
    plot_df = plot_df[avg_bar_cols].apply(pd.to_numeric, errors="coerce")
    plot_df["feature"] = plot_df.index.astype(str)
    long_df = plot_df.melt(
        id_vars="feature",
        value_vars=avg_bar_cols,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if long_df.empty:
        print(f"Skipping {task_name} bar plot: no plottable values after reshape.")
        return

    long_df["metric_label"] = long_df["metric"].map(
        lambda c: c.replace("Avg_", "").replace("_Embeddings", "")
    )
    long_df["feature_label"] = long_df["feature"].map(clean_plot_label)

    gr_title = (
        f"{cap_pred} Prediction (Average Embedding Performance): "
        f"{', '.join(bar_metrics)}"
    )
    plt.figure(figsize=(16, 6))
    sns.barplot(data=long_df, x="feature_label", y="value", hue="metric_label")

    if plot_df["feature"].nunique() < 100:
        plt.xticks(rotation=90, ha="right", fontsize=12)
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
        save_dir / f"avg_embedding_feature_bar_{task_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_regression_radar(merged: pd.DataFrame, task_cfg: dict) -> None:
    metric = task_cfg["metric"]
    avg_col = f"Avg_{metric}_Embeddings"
    group_col = f"avg_{avg_col}"

    group_performance_df = v.computeGroupPerf(
        data=merged[[avg_col]],
        descriptor_groups=group_map,
        metrics=[avg_col],
        exclude=excl_cols,
    )
    if group_col not in group_performance_df.columns:
        numeric_cols = group_performance_df.select_dtypes(include="number").columns
        if numeric_cols.empty:
            print("Skipping regression radar plot: no numeric grouped values.")
            return
        group_col = numeric_cols[0]

    radar_df = group_performance_df[[group_col]].copy()
    radar_df[group_col] = pd.to_numeric(radar_df[group_col], errors="coerce")
    radar_df = radar_df.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=[group_col]
    )
    if radar_df.empty:
        print(
            f"Skipping regression radar plot: no finite grouped values for '{group_col}'."
        )
        return

    pretty_metric = " ".join(part.capitalize() for part in metric.split("_"))
    gr_title = f"{cap_pred} Prediction (Average Embedding Performance): {pretty_metric}"
    v.plotGroupRadar(
        radar_df,
        title=gr_title,
        save_plot=True,
        save_path=save_dir,
        save_fname=f"avg_emb_group_radar_regression_{gr_fname_suffix}".rstrip("_"),
        metadata={"Title": gr_title},
    )


for task_name, task_cfg in TASK_METRICS.items():
    if task_name == "regression":
        metric = task_cfg["metric"]
        merged = _build_task_merged(metric=metric, task_name=task_name)
        if merged is None:
            print(f"Skipping {task_name}: no available data for metric '{metric}'.")
            continue
        plot_regression_radar(merged=merged, task_cfg=task_cfg)
        merged.to_csv(save_dir / f"avg_embedding_merged_{task_name}.csv")
        continue

    metrics_to_build = task_cfg.get("bar_metrics", [task_cfg["metric"]])
    merged_frames = []
    for metric in metrics_to_build:
        metric_merged = _build_task_merged(metric=metric, task_name=task_name)
        if metric_merged is None:
            continue
        avg_col = f"Avg_{metric}_Embeddings"
        if avg_col in metric_merged.columns:
            merged_frames.append(
                metric_merged[[avg_col]].rename(
                    columns={avg_col: f"Avg_{metric}_Embeddings"}
                )
            )

    if not merged_frames:
        print(
            f"Skipping {task_name}: no available data for any requested metrics {metrics_to_build}."
        )
        continue

    merged = pd.concat(merged_frames, axis=1)
    plot_task_bars(merged=merged, task_name=task_name, task_cfg=task_cfg)
    merged.to_csv(save_dir / f"avg_embedding_merged_{task_name}.csv")
