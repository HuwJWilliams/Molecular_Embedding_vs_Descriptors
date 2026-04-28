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


paths=getPaths()
v = Visualise()

pred = "maccs"
cap_pred = "RDKit" if pred == "rdkit" else pred.capitalize()
exclude_low_var = True
var_threshold = 0.8

tr2avg_ls = [
    "chemberta",
    "molformer",
    "molformer-c3-1b",
    "selformer",
    "chembertasey"
]

tr_ls=[
    "rdkit",
    "mordred",
    "morgan"
]

full_tr_ls = tr2avg_ls + tr_ls

#endregion
results_dir = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
avg_results_df_paths = [
    results_dir[f"pred_{pred}_tr_{tr_avg}"] / f"pred_{pred}_tr_{tr_avg}.csv" 
    for tr_avg in tr2avg_ls
]

other_results_df_paths = [
    results_dir[f"pred_{pred}_tr_{tr}"] /  f"pred_{pred}_tr_{tr}.csv" 
    for tr in tr_ls
    ]

exp_keys = [
    f"pred_{pred}_tr_{tr}" for tr in full_tr_ls
]

all_paths = avg_results_df_paths + other_results_df_paths

TASK_METRICS = {
    "regression": "Pearson_r",
    "classification": "AUC",
    "multiclass": "AUC_OVR",
}
TASK_TYPE_MAP = {
    "regression": "regression",
    "classification": "binary_classification",
    "multiclass": "multiclass_classification",
}


group_map = getGroups(pred)
pred_ft_df = Path(paths["full_features"]["all"][pred])
l_var_col = getLowVarianceColumns(pred_ft_df, threshold=var_threshold)
excl_cols = l_var_col if exclude_low_var else []

gr_fname_suffix = "excl_low_var" if exclude_low_var else "" 

save_dir = paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions" / f"pred_{pred}_tr_avg_emb" 
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
            filtered = task_df.loc[task_df["task_type"] == TASK_TYPE_MAP[task_name]].copy()
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


for task_name, metric in TASK_METRICS.items():
    merged = _build_task_merged(metric=metric, task_name=task_name)
    if merged is None:
        print(f"Skipping {task_name}: no available data for metric '{metric}'.")
        continue

    avg_col = f"Avg_{metric}_Embeddings"
    group_performance_df = v.computeGroupPerf(
        data=merged[[avg_col]],
        descriptor_groups=group_map,
        metrics=[avg_col],
        exclude=excl_cols
    )
    # Drop NaN/Inf groups before radar plotting
    group_performance_df[avg_col] = pd.to_numeric(group_performance_df[avg_col], errors="coerce")
    group_performance_df = group_performance_df.replace([float("inf"), float("-inf")], pd.NA)
    group_performance_df = group_performance_df.dropna(subset=[avg_col])
    if group_performance_df.empty:
        print(f"Skipping {task_name}: no finite grouped values for '{avg_col}'.")
        continue

    save_path = save_dir / f"avg_embedding_group_radar_{task_name}.csv"
    merged.to_csv(save_path)

    pretty_metric = " ".join(part.capitalize() for part in metric.split("_"))
    gr_title = f"{cap_pred} Prediction (Average Embedding Performance): {pretty_metric}"

    v.plotGroupRadar(
        group_performance_df[[avg_col]],
        title=gr_title,
        save_plot=True,
        save_path=save_path.parent,
        save_fname=f"avg_emb_group_radar_{task_name}_{gr_fname_suffix}".rstrip("_"),
        metadata={
            "Title": gr_title,
        }
    )
