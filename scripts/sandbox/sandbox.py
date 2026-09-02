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
    exp_list: list[str],
    cfp_dir: dict[str, str | Path],
    task_type: str | None = "regression",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Average each CFP experiment across all descriptor rows.

    This is similar to averageExperimentPerformance(), but instead of returning
    one averaged row per descriptor, it returns one averaged row per experiment.
    A second table gives the total average across those experiment-level means.
    """

    if not exp_list:
        raise ValueError("No experiments provided.")

    rows = []

    for exp in exp_list:
        exp_dir = Path(cfp_dir[exp])
        exp_path = exp_dir / f"{exp}.csv"

        if not exp_path.exists():
            raise FileNotFoundError(f"Could not find experiment file: {exp_path}")

        exp_df = pd.read_csv(exp_path, index_col=0)

        if task_type is not None and "task_type" in exp_df.columns:
            exp_df = exp_df.loc[exp_df["task_type"] == task_type].copy()

        numeric_cols = exp_df.select_dtypes(include="number").columns.tolist()
        mean_metrics = exp_df[numeric_cols].mean(numeric_only=True).to_dict()

        exp_parts = exp.split("_")
        rows.append(
            {
                "experiment": exp,
                "pred": exp_parts[1] if len(exp_parts) > 1 else pd.NA,
                "train": exp_parts[3] if len(exp_parts) > 3 else pd.NA,
                "task_type": task_type or "all",
                "n_descriptors": len(exp_df),
                **mean_metrics,
            }
        )

    descriptor_average_df = pd.DataFrame(rows).set_index("experiment")

    metric_cols = [
        col
        for col in descriptor_average_df.select_dtypes(include="number").columns
        if col != "n_descriptors"
    ]
    total_average_df = pd.DataFrame(
        [descriptor_average_df[metric_cols].mean(numeric_only=True)]
    )
    total_average_df.insert(0, "task_type", task_type or "all")
    total_average_df.insert(0, "n_experiments", len(descriptor_average_df))

    if save:
        exp_parts = [exp.split("_") for exp in exp_list]
        preds = [parts[1] if len(parts) > 1 else "unknown" for parts in exp_parts]
        pred = sorted(set(preds))[0] if preds else "unknown"
        save_dir = Path(cfp_dir[exp_list[0]]).parent / f"pred_{pred}_tr_total_avg"
        save_dir.mkdir(parents=True, exist_ok=True)

        descriptor_average_df.to_csv(save_dir / "descriptor_average_by_experiment.csv")
        total_average_df.to_csv(save_dir / "total_descriptor_average.csv", index=False)

    return descriptor_average_df, total_average_df


import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
sys.path.insert(0, str(CONFIG_DIR))
from config import FULL_PATHING

# Example:
exp_list = [
    "pred_mordred_tr_molformer-c3-1b",
    "pred_mordred_tr_ft-random-molformer-c3-1b",
]
cfp_dir = FULL_PATHING["prediction_output_dirs"]["lipinski_cross_feature_predictions"][
    "hole_re"
]
descriptor_average_df, total_average_df = averageExperimentPerformanceTotalDescriptors(
    exp_list=exp_list,
    cfp_dir=cfp_dir,
    task_type="regression",
)
