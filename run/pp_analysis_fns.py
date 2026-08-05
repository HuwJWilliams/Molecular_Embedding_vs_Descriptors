"""
Functions to analyse the individual property prediction (PP) results
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from glob import glob

# %% ===== Project Imports & Pathing Setup =====
from config import PATHING_JSON_PATH, SRC_DIR, PP_ANALYSIS_METRICS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import checkLipinskiCriteria

v = Visualise(save_all=False)


# %% ===== Function Definitions =====
def getPropertyPerformanceDfs(
    property_pathing: dict[str, Path],
    feature_sets: list[str],
    perf_fname: str = "rf_performance.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read internal and external mean/std performance from rf_performance.json
    files for each feature set.

    Returns
    -------
    int_property_performance_df:
        Rows for internal mean/std performance.

    ext_property_performance_df:
        Rows for external mean/std performance.
    """

    int_rows = []
    ext_rows = []

    for feat in feature_sets:
        feature_path = Path(property_pathing[feat])
        perf_json_path = feature_path / perf_fname

        with open(perf_json_path, "r") as f:
            perf_json = json.load(f)

        internal = perf_json["internal"]
        int_mean = internal["mean"]
        int_std = internal["std"]

        external = perf_json["external"]
        ext_mean = external["mean"]
        ext_std = external["std"]

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **int_mean,
            }
        )

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **int_std,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **ext_mean,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **ext_std,
            }
        )

    int_property_performance_df = pd.DataFrame(int_rows)
    ext_property_performance_df = pd.DataFrame(ext_rows)

    return int_property_performance_df, ext_property_performance_df


def getFeatureColourConfig(
    feature_ls: list[str],
    colour_map: dict[str, str],
):
    feature_colour_config = {}

    for feature in feature_ls:
        if feature.startswith("ft-"):
            hatch = "//"
            stripped_feature_name = feature.removeprefix("ft-")
            stripped_feature_name = stripped_feature_name.removeprefix("random-")
            stripped_feature_name = stripped_feature_name.removeprefix("scaffold-")
        else:
            hatch = None
            stripped_feature_name = feature

        colour = colour_map.get(stripped_feature_name, "#808080")
        feature_colour_config[feature] = (colour, hatch)

    return feature_colour_config


FT_FEATURE_PAIRS = {
    "ft-scaffold-chemberta": "chemberta",
    "ft-scaffold-chembertasey": "chembertasey",
    "ft-scaffold-molformer": "molformer",
    "ft-scaffold-molformer-c3-1b": "molformer-c3-1b",
    "ft-scaffold-selformer": "selformer",
    "ft-random-chemberta": "chemberta",
    "ft-random-chembertasey": "chembertasey",
    "ft-random-molformer": "molformer",
    "ft-random-molformer-c3-1b": "molformer-c3-1b",
    "ft-random-selformer": "selformer",
}

METRIC_ALIASES = {
    "pearson_r": ["pearson_r", "r_pearson", "Pearson_r"],
    "Pearson_r": ["Pearson_r", "pearson_r", "r_pearson"],
    "r_pearson": ["r_pearson", "pearson_r", "Pearson_r"],
    "rmse": ["rmse", "RMSE"],
    "RMSE": ["RMSE", "rmse"],
    "bias": ["bias", "Bias"],
    "Bias": ["Bias", "bias"],
    "sdep": ["sdep", "SDEP"],
    "SDEP": ["SDEP", "sdep"],
}


def getMetricColumn(metric: str, data: pd.DataFrame) -> str | None:
    for metric_alias in METRIC_ALIASES.get(metric, []):
        if metric_alias in data.columns:
            return metric_alias

    metric_lower = metric.lower()

    if metric in data.columns:
        return metric

    if metric_lower in data.columns:
        return metric_lower

    return None


def getFTDifferenceDf(
    data: pd.DataFrame,
    prop: str,
    split_name: str,
    metric_col: str,
) -> pd.DataFrame:
    mean_df = data.loc[
        data["stat"] == "mean",
        ["feature_set", metric_col],
    ].copy()

    mean_df[metric_col] = pd.to_numeric(mean_df[metric_col], errors="coerce")
    mean_by_feature = mean_df.set_index("feature_set")[metric_col]

    rows = []
    for ft_feature, base_feature in FT_FEATURE_PAIRS.items():
        if (
            ft_feature not in mean_by_feature.index
            or base_feature not in mean_by_feature.index
        ):
            continue

        rows.append(
            {
                "property": prop,
                "split": split_name,
                "metric": metric_col,
                "feature_pair": f"{ft_feature} - {base_feature}",
                "ft_feature": ft_feature,
                "base_feature": base_feature,
                "ft_value": mean_by_feature[ft_feature],
                "base_value": mean_by_feature[base_feature],
                "difference": mean_by_feature[ft_feature]
                - mean_by_feature[base_feature],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "property",
                "split",
                "metric",
                "feature_pair",
                "ft_feature",
                "base_feature",
                "ft_value",
                "base_value",
                "difference",
            ]
        )

    return pd.DataFrame(rows).dropna(subset=["difference"])


def plotFTDifferenceBar(
    data: pd.DataFrame,
    prop: str,
    split_name: str,
    metric_col: str,
    save_path: Path,
    colour_map: dict,
    dpi: int = 400,
) -> pd.DataFrame:
    diff_df = getFTDifferenceDf(
        data=data,
        prop=prop,
        split_name=split_name,
        metric_col=metric_col,
    )

    if diff_df.empty:
        print(f"No FT/base pairs for {prop} / {split_name} / {metric_col}")
        return diff_df

    diff_df = diff_df.sort_values("difference", ascending=False)
    bar_colours = [
        (
            colour_map.get(row["base_feature"], "#808080")
            if not isinstance(colour_map.get(row["base_feature"]), tuple)
            else colour_map[row["base_feature"]][0]
        )
        for _, row in diff_df.iterrows()
    ]

    plot_dir = save_path / prop / split_name / "ft_differences"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=diff_df,
        x="feature_pair",
        y="difference",
        hue="feature_pair",
        order=diff_df["feature_pair"].tolist(),
        palette=bar_colours,
        legend=False,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel(f"ft - base {metric_col}", fontsize=12, weight="bold")
    plt.xlabel("feature pair", fontsize=12, weight="bold")
    plt.title(
        f"{prop}: {split_name} {metric_col} FT difference",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        plot_dir / f"{prop}_{split_name}_{metric_col}_ft_difference_bar.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    return diff_df


def plotSummaryHeatmaps(
    summary_df: pd.DataFrame,
    split_name: str,
    save_path: Path,
    metrics: list[str],
    feature_order: list[str],
    dpi: int = 400,
) -> None:
    heatmap_dir = save_path / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    lower_is_better_metrics = {"rmse", "mse", "mae", "sdep"}

    for metric in metrics:
        if metric not in summary_df.columns:
            print(f"{metric} not in {split_name} summary columns")
            continue

        heatmap_df = summary_df.pivot(
            index="property",
            columns="feature_set",
            values=metric,
        )

        ordered_features = [
            feature for feature in feature_order if feature in heatmap_df.columns
        ]
        heatmap_df = heatmap_df[ordered_features]

        if heatmap_df.empty:
            print(f"No heatmap data for {split_name} / {metric}")
            continue

        figsize = (14, max(6, 0.45 * len(heatmap_df)))

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="white",
        )
        plt.title(
            f"{split_name} average {metric}",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_heatmap.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_df,
            annot=False,
            cmap="viridis",
            linewidths=0.5,
            linecolor="white",
        )
        plt.title(
            f"{split_name} average {metric}",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_heatmap_no_values.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        if metric.lower() == "bias":
            ranking_df = heatmap_df.abs().rank(
                axis=1,
                method="min",
                ascending=True,
            )
        else:
            ranking_df = heatmap_df.rank(
                axis=1,
                method="min",
                ascending=metric.lower() in lower_is_better_metrics,
            )

        plt.figure(figsize=figsize)
        sns.heatmap(
            ranking_df,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "placement rank (1 = best)"},
        )
        plt.title(
            f"{split_name} average {metric} placement",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_placement_heatmap.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()


def plotGroupedPropertyFeatureBar(
    summary_df: pd.DataFrame,
    split_name: str,
    save_path: Path,
    metric: str = "r2",
    feature_order: list[str] | None = None,
    colour_map: dict | None = None,
    dpi: int = 400,
) -> None:
    if metric not in summary_df.columns:
        print(f"{metric} not in {split_name} summary columns")
        return

    plot_df = summary_df[["property", "feature_set", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        print(f"No grouped bar data for {split_name} / {metric}")
        return

    if feature_order is None:
        feature_order = plot_df["feature_set"].drop_duplicates().tolist()

    feature_order = [
        feature
        for feature in feature_order
        if feature in plot_df["feature_set"].unique()
    ]
    property_order = plot_df["property"].drop_duplicates().tolist()

    if colour_map is None:
        colour_map = {}

    palette = {}
    hatches = {}
    for feature in feature_order:
        style = colour_map.get(feature, "#808080")
        if isinstance(style, tuple):
            colour, hatch = style
        elif isinstance(style, dict):
            colour = style.get("colour", "#808080")
            hatch = style.get("hatch")
        else:
            colour = style
            hatch = "//" if feature.startswith("ft-") else None

        palette[feature] = colour
        hatches[feature] = hatch

    bar_dir = save_path / "grouped_bars"
    bar_dir.mkdir(parents=True, exist_ok=True)

    figsize = (max(14, 0.85 * len(property_order)), 7)
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df,
        x="property",
        y=metric,
        hue="feature_set",
        order=property_order,
        hue_order=feature_order,
        palette=palette,
        errorbar=None,
    )

    for container, feature in zip(ax.containers, feature_order):
        hatch = hatches.get(feature)
        if hatch:
            for bar in container:
                bar.set_hatch(hatch)
                bar.set_edgecolor("black")
                bar.set_linewidth(0.8)

    if metric.lower() == "r2":
        ax.set_ylim(0, 1)

    ax.set_xlabel("property", fontsize=12, weight="bold")
    ax.set_ylabel(metric, fontsize=12, weight="bold")
    ax.set_title(
        f"{split_name} average {metric} by property and feature set",
        fontsize=16,
        weight="bold",
    )
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(
        title="feature_set",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(
        bar_dir / f"{split_name}_average_{metric}_grouped_property_feature_bar.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()


def plotFTDifferenceSummaryBars(
    ft_difference_df: pd.DataFrame,
    save_path: Path,
    metrics: list[str] | None = None,
    dpi: int = 400,
) -> None:
    if ft_difference_df.empty:
        print("No FT difference rows available for summary bar plots.")
        return

    if metrics is None:
        metrics = ["r2", "pearson_r"]

    plot_dir = save_path / "ft_differences"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_alias_lookup = {
        metric: METRIC_ALIASES.get(metric, []) + [metric] for metric in metrics
    }

    for split_name in sorted(ft_difference_df["split"].dropna().unique()):
        split_df = ft_difference_df.loc[ft_difference_df["split"] == split_name].copy()

        for metric, aliases in metric_alias_lookup.items():
            metric_df = split_df.loc[split_df["metric"].isin(aliases)].copy()
            metric_df["difference"] = pd.to_numeric(
                metric_df["difference"],
                errors="coerce",
            )
            metric_df = metric_df.dropna(subset=["difference"])

            if metric_df.empty:
                print(f"No FT difference summary rows for {split_name} / {metric}")
                continue

            property_order = metric_df["property"].drop_duplicates().tolist()
            feature_pair_order = metric_df["feature_pair"].drop_duplicates().tolist()

            plt.figure(figsize=(max(14, 0.9 * len(property_order)), 7))
            ax = sns.barplot(
                data=metric_df,
                x="property",
                y="difference",
                hue="feature_pair",
                order=property_order,
                hue_order=feature_pair_order,
                errorbar=None,
            )
            ax.axhline(0, color="black", linewidth=1)
            ax.set_xlabel("property", fontsize=12, weight="bold")
            ax.set_ylabel(f"ft - base {metric}", fontsize=12, weight="bold")
            ax.set_title(
                f"{split_name} {metric} FT differences",
                fontsize=16,
                weight="bold",
            )
            ax.tick_params(axis="x", labelrotation=45, labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            ax.legend(
                title="feature pair",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
            )
            plt.tight_layout()
            plt.savefig(
                plot_dir / f"{split_name}_{metric}_ft_difference_summary_bar.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()


def loadFeaturePattern(path: str | Path) -> pd.DataFrame:
    path = Path(path) if "*" not in str(path) else str(path)

    if "*" in str(path):
        files = sorted(glob(str(path)))
        if not files:
            raise FileNotFoundError(f"No files matched: {path}")
        return pd.concat(
            [pd.read_csv(file, index_col=0, low_memory=False) for file in files],
            axis=0,
        )

    return pd.read_csv(path, index_col=0, low_memory=False)


def calculateR2(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true.loc[valid]
    y_pred = y_pred.loc[valid]

    if len(y_true) < 2:
        return float("nan")

    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    if ss_total == 0:
        return float("nan")

    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - (ss_res / ss_total))


def getLipinskiFilteredExternalPerformanceDf(
    properties: list[str],
    feature_sets: list[str],
    full_pathing: dict,
    preds_dir: dict,
    target_columns: dict[str, str],
) -> pd.DataFrame:
    rows = []

    for prop in properties:
        if prop not in preds_dir:
            print(f"{prop} not in prediction output paths")
            continue

        try:
            rdkit_df = loadFeaturePattern(full_pathing["full_features"][prop]["rdkit"])
            lipinski_ids = pd.Index(checkLipinskiCriteria(rdkit_df)).astype(str)

            target_col = target_columns[prop]
            target_df = pd.read_csv(
                full_pathing["targets"][prop],
                index_col="ID",
            )
            if target_col not in target_df.columns:
                print(f"{target_col} not in target columns for {prop}")
                continue

            target_df = target_df[[target_col]].rename(columns={target_col: "true"})
            target_df.index = target_df.index.astype(str)

        except Exception as e:
            print(f"Could not prepare Lipinski IDs for {prop}: {e}")
            continue

        for feature_set in feature_sets:
            if feature_set not in preds_dir[prop]:
                continue

            pred_path = Path(preds_dir[prop][feature_set]) / "last_20pct_pred.csv.gz"
            if not pred_path.exists():
                print(
                    f"Missing last_20pct predictions for {prop} / {feature_set}: {pred_path}"
                )
                continue

            try:
                pred_df = pd.read_csv(pred_path, index_col=0)
                pred_df.index = pred_df.index.astype(str)

                pred_col = (
                    target_col if target_col in pred_df.columns else pred_df.columns[0]
                )
                pred_df = pred_df[[pred_col]].rename(columns={pred_col: "pred"})

                keep_ids = pred_df.index.intersection(target_df.index).intersection(
                    lipinski_ids
                )

                eval_df = pred_df.loc[keep_ids].join(
                    target_df.loc[keep_ids],
                    how="inner",
                )
                eval_df = eval_df.dropna(subset=["pred", "true"])

                if eval_df.empty:
                    print(f"No Lipinski-matched rows for {prop} / {feature_set}")
                    continue

                rows.append(
                    {
                        "property": prop,
                        "split": "external_lipinski",
                        "feature_set": feature_set,
                        "r2": calculateR2(eval_df["true"], eval_df["pred"]),
                        "n": len(eval_df),
                        "n_lipinski_ids": len(lipinski_ids),
                    }
                )

            except Exception as e:
                print(
                    f"Could not calculate Lipinski performance for {prop} / {feature_set}: {e}"
                )

    return pd.DataFrame(rows)


def get3xIQRFilteredExternalPerformanceDf(
    properties,
    feature_sets,
    full_pathing,
    preds_dir,
    target_columns,
):
    rows = []

    for prop in properties:
        target_col = target_columns[prop]

        target_df = pd.read_csv(full_pathing["targets"][prop], index_col="ID")
        target_df.index = target_df.index.astype(str)
        true = pd.to_numeric(target_df[target_col], errors="coerce")

        q1 = true.quantile(0.25)
        q3 = true.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (3 * iqr)
        upper = q3 + (3 * iqr)

        keep_target = true.loc[true.between(lower, upper)].rename("true")

        for feature_set in feature_sets:
            if feature_set not in preds_dir[prop]:
                print(f"{feature_set} not in prediction output paths for {prop}")
                continue

            pred_path = Path(preds_dir[prop][feature_set]) / "last_20pct_pred.csv.gz"

            if not pred_path.exists():
                print(f"Missing predictions for {prop} / {feature_set}: {pred_path}")
                continue

            pred_df = pd.read_csv(pred_path, index_col=0)
            pred_df.index = pred_df.index.astype(str)

            pred_col = (
                target_col if target_col in pred_df.columns else pred_df.columns[0]
            )
            pred = pd.to_numeric(pred_df[pred_col], errors="coerce").rename("pred")

            eval_df = pd.concat([keep_target, pred], axis=1, join="inner").dropna()

            if len(eval_df) < 2:
                continue

            err = eval_df["pred"] - eval_df["true"]

            rows.append(
                {
                    "property": prop,
                    "split": "external_3xIQR",
                    "feature_set": feature_set,
                    "stat": "mean",
                    "r2": calculateR2(eval_df["true"], eval_df["pred"]),
                    "pearson_r": eval_df["true"].corr(
                        eval_df["pred"], method="pearson"
                    ),
                    "rmse": (err.pow(2).mean()) ** 0.5,
                    "bias": err.mean(),
                    "sdep": (err - err.mean()).pow(2).mean() ** 0.5,
                    "n": len(eval_df),
                    "lower_3xIQR": lower,
                    "upper_3xIQR": upper,
                }
            )

    return pd.DataFrame(rows)
