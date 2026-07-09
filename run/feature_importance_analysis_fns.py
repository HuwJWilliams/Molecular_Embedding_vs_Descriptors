"""
Fucntions to generate the feature importance analysis
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %% ===== Project Imports & Pathing Setup =====
from config import PATHING_JSON_PATH, SRC_DIR, PP_ANALYSIS_METRICS


def cleanFeatureName(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.str.replace(
        r"_(mordred|rdkit|maccs).*",
        "",
        regex=True,
    )
    return df


def AverageFeatureImportance(
    df_ls: list[str], save_path: str | Path, filename: str, top_n_feats=50
):

    loaded_dfs = [
        pd.read_csv(df, index_col="Feature", usecols=["Importance", "Feature"])
        for df in df_ls
    ]
    mean_df = sum(loaded_dfs) / len(loaded_dfs)
    mean_df = cleanFeatureName(mean_df)

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=mean_df.head(top_n_feats),
        x="Importance",
        y="Feature",
        palette="viridis",
        dodge=False,
        hue="Feature",
        legend=False,
    )

    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.savefig(f"{save_path}/{filename}.png", dpi=800)
    mean_df.to_csv(f"{save_path}/feature_importance_df.csv")

    return mean_df


def importanceVsPredictability(
    feature_importance_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    performance_metric: str = "r2",
):
    feat_imp_df = cleanFeatureName(feature_importance_df)

    pred_df = pred_df.copy()
    if "Feature" in pred_df.columns:
        pred_df = pred_df.set_index("Feature")

    pred_df = cleanFeatureName(pred_df)

    plot_df = pd.DataFrame(
        {
            "importance": feat_imp_df,
            performance_metric: pred_df[performance_metric],
        }
    ).dropna()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = range(len(plot_df))

    ax1.scatter(x, plot_df["importance"], color="tab:blue", label="Importance")
    ax1.set_ylabel("Importance", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.scatter(
        x,
        plot_df[performance_metric],
        color="tab:orange",
        label=f"Predictability ({performance_metric})",
    )
    ax2.set_ylabel(f"Predictability ({performance_metric})", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df.index, rotation=90)
    ax1.set_xlabel("Descriptor")

    ax1.set_title("Feature Importance vs Descriptor Predictability")

    fig.tight_layout()
    plt.show()

    return plot_df
