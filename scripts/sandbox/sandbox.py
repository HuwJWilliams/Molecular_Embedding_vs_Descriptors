import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# data = pd.read_csv(
#     "/users/yhb18174/TL_project/datasets/embeddings/BP_embeddings/"
#     "bp_ft-random-molformer-c3-1b_fine_tuned_model/fine_tuning_log_history.csv"
# )

# train_loss = data.dropna(subset=["loss"]).copy()

# x_col = "epoch" if "epoch" in train_loss.columns else "step"

# plt.figure(figsize=(7, 4))
# plt.plot(train_loss[x_col], train_loss["loss"], marker="o")
# plt.xlabel(x_col.capitalize())
# plt.ylabel("Training loss")
# plt.title("Fine-tuning Training Loss")
# plt.tight_layout()
# plt.show()

# %% Checking averages

data = pd.read_csv(
    "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_avg_transformers/descriptor_r2_by_model_with_average.csv",
    index_col=0,
)


def averageExperimentPerformanceTotalDescriptors(
    exp: str,
    exp_dir: str | Path,
    save: bool = True,
) -> pd.DataFrame:
    """
    Average one CFP experiment across all descriptor rows.

    This is similar to averageExperimentPerformance(), but instead of returning
    one averaged row per descriptor, it returns one total averaged row for the
    whole experiment.
    """

    exp_dir = Path(exp_dir)
    exp_path = exp_dir / f"{exp}.csv"

    if not exp_path.exists():
        raise FileNotFoundError(f"Could not find experiment file: {exp_path}")

    exp_df = pd.read_csv(exp_path, index_col=0)

    task_metric_cols = {
        "regression": [
            "Bias",
            "SDEP",
            "MSE",
            "RMSE",
            "r2",
            "Pearson_r",
            "Pearson_p",
        ],
        "binary_classification": [
            "Accuracy",
            "Balanced_Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "AUC",
            "MCC",
        ],
        "multiclass_classification": [
            "Accuracy",
            "Balanced_Accuracy",
            "F1_macro",
            "AUC_OVR",
            "MCC",
        ],
    }

    descriptor_counts = {}
    mean_metrics = {}

    if "task_type" in exp_df.columns:
        for task_name, metric_cols in task_metric_cols.items():
            task_df = exp_df.loc[exp_df["task_type"] == task_name].copy()
            descriptor_counts[f"n_{task_name}_descriptors"] = len(task_df)

            for metric_col in metric_cols:
                if metric_col not in task_df.columns:
                    continue

                metric_values = pd.to_numeric(task_df[metric_col], errors="coerce")
                if metric_values.notna().any():
                    mean_metrics[f"{task_name}_{metric_col}"] = metric_values.mean()
    else:
        numeric_cols = exp_df.select_dtypes(include="number").columns.tolist()
        mean_metrics = exp_df[numeric_cols].mean(numeric_only=True).to_dict()

    exp_parts = exp.split("_")
    total_average_df = pd.DataFrame(
        [
            {
                "experiment": exp,
                "pred": exp_parts[1] if len(exp_parts) > 1 else pd.NA,
                "train": exp_parts[3] if len(exp_parts) > 3 else pd.NA,
                "n_descriptors": len(exp_df),
                **descriptor_counts,
                **mean_metrics,
            }
        ]
    )

    if save:
        total_average_df.to_csv(exp_dir / "total_descriptor_average.csv", index=False)

    return total_average_df


import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
sys.path.insert(0, str(CONFIG_DIR))
from config import FULL_PATHING, SRC_DIR

total_average_dfs = []

cfp_block = FULL_PATHING["prediction_output_dirs"][
    "lipinski_cross_feature_predictions"
]["all"]

# for exp, exp_dir in cfp_block.items():
#     exp_parts = exp.split("_")

#     if len(exp_parts) < 4:
#         continue

#     pred_feature = exp_parts[1]

#     if pred_feature != "mordred" and exp != "pred_rdkit_tr_mordred":
#         continue

#     try:
#         total_average_df = averageExperimentPerformanceTotalDescriptors(
#             exp=exp,
#             exp_dir=exp_dir,
#         )
#         total_average_dfs.append(total_average_df)
#     except Exception as e:
#         print(e)

# all_total_average_df = pd.concat(total_average_dfs, ignore_index=True)
# numeric_cols = all_total_average_df.select_dtypes(include="number").columns
# all_total_average_df[numeric_cols] = all_total_average_df[numeric_cols].round(3)
# all_total_average_df.to_csv(
#     "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_all_task_metric_avg_perf.csv",
#     index=False,
# )


# %% Plot top RDKit features for each predicted Mordred descriptor
# %% Plot top RDKit features for predicting Mordred Autocorrelation descriptors
sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import *

# Get all Mordred descriptors belonging to Autocorrelation
res = getMordredGroups()
autocorrelation_descs = res["Autocorrelation"]

v = Visualise(save_all=False)

# original_data = pd.read_csv(
#     cfp_block["pred_mordred_tr_rdkit"] / "all_feature_importance.csv"
# )

# # Corresponding importance columns
# cols_to_avg = [
#     f"Importance_{desc}"
#     for desc in autocorrelation_descs
#     if f"Importance_{desc}" in original_data.columns
# ]

# # Mean importance of each RDKit feature across all
# # Autocorrelation Mordred descriptors
# original_data["Importance_Autocorrelation"] = original_data[cols_to_avg].mean(axis=1)

# v.plotFeatureImportance(
#     data=original_data,
#     x_col="Importance_Autocorrelation",
#     y_col="Feature",
#     top_n=25,
#     save_path=str(Path(__file__).parent),
# )

# %% Plot Mordred ATS atomic-mass autocorrelation descriptor distributions
import math
import re

import numpy as np

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
ATS_MASS_PATTERN = re.compile(r"^ATS(?P<lag>\d+)m(?:_mordred)?$")

mordred_features_path = Path(FULL_PATHING["full_features"]["all"]["mordred"])
if not mordred_features_path.exists():
    mordred_features_path = (
        PROJ_DIR
        / "run"
        / "test"
        / "expected_test_results"
        / "expected_mordred_features.csv"
    )

mordred_features = pd.read_csv(mordred_features_path, index_col=0, low_memory=False)

ats_mass_cols = []
for col in mordred_features.columns:
    match = ATS_MASS_PATTERN.match(col)
    if match:
        ats_mass_cols.append((int(match.group("lag")), col))

ats_mass_cols = [col for _, col in sorted(ats_mass_cols)]

if not ats_mass_cols:
    raise ValueError(
        "No ATS atomic-mass columns found. Expected names like "
        "ATS0m_mordred, ATS1m_mordred, ..."
    )

n_plot_cols = 3
n_plot_rows = math.ceil(len(ats_mass_cols) / n_plot_cols)
ats_mass_values = [
    pd.to_numeric(mordred_features[col], errors="coerce").dropna()
    for col in ats_mass_cols
]
all_ats_mass_values = pd.concat(ats_mass_values, ignore_index=True)
if all_ats_mass_values.empty:
    raise ValueError("ATS atomic-mass columns were found, but none contained numeric values.")

bin_min = all_ats_mass_values.min()
bin_max = all_ats_mass_values.max()
if bin_min == bin_max:
    bin_min -= 0.5
    bin_max += 0.5

bin_edges = np.linspace(bin_min, bin_max, 41)

fig, axes = plt.subplots(
    n_plot_rows,
    n_plot_cols,
    figsize=(4.8 * n_plot_cols, 3.6 * n_plot_rows),
    squeeze=False,
)
axes_flat = axes.ravel()

for ax, col, non_nan_values in zip(axes_flat, ats_mass_cols, ats_mass_values):
    ax.hist(
        non_nan_values,
        bins=bin_edges,
        edgecolor="black",
        color="steelblue",
        alpha=0.85,
    )
    ax.set_title(f"{col} (non-NaN n={len(non_nan_values)})", fontsize=11)
    ax.set_xlabel("Descriptor value")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

for ax in axes_flat[len(ats_mass_cols) :]:
    ax.axis("off")

fig.suptitle("Mordred Autocorrelation: ATS Atomic Mass", fontsize=14, weight="bold")
fig.tight_layout()

save_path = FILE_DIR / "ats_mass_histograms.png"
fig.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Loaded Mordred features from: {mordred_features_path}")
print(f"Plotted descriptors: {', '.join(ats_mass_cols)}")
print(f"Saved histogram plot to: {save_path}")
