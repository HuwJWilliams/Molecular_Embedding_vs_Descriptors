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
from config import FULL_PATHING

total_average_dfs = []

cfp_block = FULL_PATHING["prediction_output_dirs"][
    "lipinski_cross_feature_predictions"
]["all"]

for exp, exp_dir in cfp_block.items():
    exp_parts = exp.split("_")

    if len(exp_parts) < 4:
        continue

    pred_feature = exp_parts[1]

    if pred_feature != "mordred":
        continue

    try:
        total_average_df = averageExperimentPerformanceTotalDescriptors(
            exp=exp,
            exp_dir=exp_dir,
        )
        total_average_dfs.append(total_average_df)
    except Exception as e:
        print(e)

all_total_average_df = pd.concat(total_average_dfs, ignore_index=True)
all_total_average_df.to_csv(
    "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_all_task_metric_avg_perf.csv",
    index=False,
)
