from __future__ import annotations

from pathlib import Path
from glob import glob
import random as rand
from typing import Iterable, Tuple, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch

DataInput = Union[pd.DataFrame, str, Path]
DatasetSpec = Tuple[str, DataInput]  # (source_name, data)


def plotPCA(
    datasets: dict[str, Path],
    plot_dir: Path,
    n_components: int = 5,
    loadings_filename: str = "pca_loadings",
    pca_df_filename: str = "pca_components",
    kdep_sample_size: float = 0.33,
    contamination: float = 0.00001,
    plot_fname: str = "plotPCA",
    save_plot: bool = True,
    save_extra_data: bool = False,
    plot_area: bool = False,
    plot_scatter: bool = True,
    random_seed: int = None,
    plot_loadings: bool = False,
    plot_title: str = "PCA Plot",
    remove_outliers: bool = True,
    kdep_sample_ls: list = ["PyMolGen"],
    axis_fontsize: int = 20,
    tick_fontsize: int = 18,
    label_fontsize: int = 20,
    legend_fontsize: int = 20,
    kde_tick_dicts: list = None,
):
    """
    datasets: iterable of (source_name, input)
      - input can be:
          * pd.DataFrame (index must be 'ID' or provide 'ID' column)
          * CSV filepath (str/Path)
          * glob pattern (str) e.g. "/path/preds_*.csv"
    """

    if random_seed is None:
        random_seed = rand.randint(0, 2**31)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers ----
    def _load_one(inp: DataInput) -> pd.DataFrame:
        if isinstance(inp, pd.DataFrame):
            df = inp.copy()
            if df.index.name != "ID":
                # if ID exists as a column, set it; else assume index already is ID-like
                if "ID" in df.columns:
                    df = df.set_index("ID")
                else:
                    df.index.name = "ID"
            return df

        # str/Path
        inp_str = str(inp)

        # glob pattern?
        files = glob(inp_str)
        if len(files) > 0 and (("*" in inp_str) or ("?" in inp_str) or ("[" in inp_str)):
            parts = []
            for f in files:
                parts.append(pd.read_csv(f, index_col="ID"))
            return pd.concat(parts, axis=0) if parts else pd.DataFrame()

        # single file
        return pd.read_csv(inp_str, index_col="ID")

    def _remove_outliers(df: pd.DataFrame, columns: list[str], n_neighbors=20, contamination=contamination) -> pd.DataFrame:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(df[columns])
        return df[outlier_labels == 1]

    # ---- load datasets, tag sources ----
    loaded = []
    source_ls = []
    for src, inp in datasets.items():
        df = _load_one(inp)
        if df.empty:
            continue
        df["Source"] = src
        loaded.append(df)
        source_ls.append(src)

    if len(loaded) < 2:
        raise ValueError("Need at least 2 non-empty datasets to run PCA.")

    # ---- keep only common feature columns across all datasets ----
    common_cols = None
    for df in loaded:
        cols = set(df.columns)
        cols -= {"Source"}
        common_cols = cols if common_cols is None else (common_cols & cols)

    common_cols = list(common_cols) if common_cols else []
    if not common_cols:
        raise ValueError("No common descriptor columns found across datasets.")

    loaded = [df[common_cols + ["Source"]] for df in loaded]

    # ---- combine all (largest first to control plotting layer order) ----
    loaded_sorted = sorted(loaded, key=len, reverse=True)
    combined_df = pd.concat(loaded_sorted, axis=0).dropna()

    # ---- scale features only ----
    used_cols = [c for c in combined_df.columns if c != "Source"]
    scaler = StandardScaler()

    # only numeric columns, excluding Source
    used_cols = [
    c for c in combined_df.columns
    if c != "Source" and pd.api.types.is_numeric_dtype(combined_df[c])
]

    scaled_X = scaler.fit_transform(combined_df[used_cols])

    scaled_combined_df = pd.DataFrame(scaled_X, columns=used_cols, index=combined_df.index)
    scaled_combined_df["Source"] = combined_df["Source"]

    # ---- PCA ----
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_combined_df[used_cols])
    explained_variance = pca.explained_variance_ratio_ * 100

    # ---- loadings ----
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=used_cols)
    abs_loadings_df = loadings_df.abs()

    loadings_df.rename_axis("Features", inplace=True)
    abs_loadings_df.rename_axis("Features", inplace=True)

    if save_extra_data:
        loadings_df.to_csv(plot_dir / f"{loadings_filename}.csv", index_label="Features")
        abs_loadings_df.to_csv(plot_dir / f"{loadings_filename}_abs.csv", index_label="Features")

    # ---- PCA df ----
    pca_df = pd.DataFrame(
        principal_components,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=combined_df.index
    )
    pca_df["Source"] = combined_df["Source"].values

    # ---- colors ----
    dark_colours = sns.color_palette("dark", n_colors=max(len(source_ls), 3))
    source_color_map = {src: dark_colours[i] for i, src in enumerate(source_ls)}

    if save_extra_data:
        pca_df.to_csv(plot_dir / f"{pca_df_filename}.csv.gz", index_label="ID", compression="gzip")

    # ---- remove outliers (optional) ----
    if remove_outliers:
        pca_df = _remove_outliers(pca_df, [f"PC{i+1}" for i in range(n_components)])

    # ---- plotting ----
    fig, axs = plt.subplots(nrows=n_components, ncols=n_components, figsize=(20, 20))

    for i in range(n_components):
        for j in range(n_components):

            if i != j:
                if plot_scatter:
                    sns.scatterplot(
                        x=f"PC{j+1}",
                        y=f"PC{i+1}",
                        hue="Source",
                        data=pca_df,
                        ax=axs[i, j],
                        legend=False,
                        edgecolor="none",
                        palette=source_color_map,
                        alpha=0.5
                    )

                sorted_sources = [s for s in source_ls if s in pca_df["Source"].unique()]

                for source in sorted_sources:
                    source_data = pca_df[pca_df["Source"] == source]
                    area_colour = source_color_map[source]

                    if plot_area and len(source_data) >= 3:
                        points = source_data[[f"PC{j+1}", f"PC{i+1}"]].values
                        # ConvexHull requires at least 3 non-collinear points; guard lightly
                        try:
                            hull = ConvexHull(points)
                            hull_points = points[hull.vertices]
                            hull_points = np.vstack((hull_points, hull_points[0]))

                            axs[i, j].fill(
                                hull_points[:, 0],
                                hull_points[:, 1],
                                alpha=0.2,
                                color=area_colour,
                                edgecolor=area_colour,
                                linewidth=2,
                                label=f"{source} area"
                            )
                        except Exception:
                            pass

                axs[i, j].tick_params(axis="both", labelsize=tick_fontsize, pad=6)

                if i == n_components - 1:
                    axs[i, j].set_xlabel(
                        f"PC{j+1} ({explained_variance[j]:.2f}% Var)",
                        fontsize=axis_fontsize,
                        labelpad=10
                    )
                else:
                    axs[i, j].set_xlabel("")
                    axs[i, j].set_xticklabels([])

                if j == 0:
                    axs[i, j].set_ylabel(
                        f"PC{i+1} ({explained_variance[i]:.2f}% Var)",
                        fontsize=axis_fontsize,
                        labelpad=10
                    )
                else:
                    axs[i, j].set_ylabel("")
                    axs[i, j].set_yticklabels([])

            else:
                # Diagonal KDE: loop over ALL sources
                sampled_parts = []
                for source in source_ls:
                    src_data = pca_df[pca_df["Source"] == source]
                    if src_data.empty:
                        continue

                    if source in kdep_sample_ls:
                        n = max(1, int(len(src_data) * kdep_sample_size))
                        sampled_parts.append(src_data.sample(n=n, random_state=random_seed))
                    else:
                        sampled_parts.append(src_data)

                sampled_pca_df = pd.concat(sampled_parts, axis=0) if sampled_parts else pd.DataFrame()

                if not sampled_pca_df.empty:
                    sns.kdeplot(
                        x=f"PC{i+1}",
                        hue="Source",
                        data=sampled_pca_df,
                        common_norm=False,
                        fill=True,
                        ax=axs[i, i],
                        legend=False,
                        palette=source_color_map,
                    )

                axs[i, i].set_xlabel("")
                axs[i, i].set_ylabel("Density", fontsize=label_fontsize, labelpad=10)
                axs[i, i].tick_params(axis="both", labelsize=tick_fontsize)

                if kde_tick_dicts and i < len(kde_tick_dicts):
                    tick_info = kde_tick_dicts[i]
                    if "xticks" in tick_info:
                        axs[i, i].set_xticks(tick_info["xticks"])
                    if "yticks" in tick_info:
                        axs[i, i].set_yticks(tick_info["yticks"])

    # ---- legend (scatter + optional area handle per source) ----
    legend_handles = [
        Patch(facecolor=source_color_map[src], label=src)
        for src in pca_df["Source"].unique()
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=len(legend_handles),
        fontsize=legend_fontsize,
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.95,
        bottom=0.3,
        wspace=0.6,
        hspace=0.6
    )

    if save_plot:
        plt.savefig(plot_dir / f"{plot_fname}.png", dpi=600, bbox_inches="tight")

    # ---- optional loadings plot (kept same behaviour) ----
    if plot_loadings:
        loadings_df["Max"] = loadings_df.max(axis=1)
        if isinstance(loadings_df, pd.DataFrame):
            loadings_df = loadings_df[loadings_df["Max"] > 0.3]
            loadings_df.drop(columns=["Max"])

        fig2, ax = plt.subplots(n_components, 1, figsize=(25, 25), sharex=True)

        for n in range(1, n_components + 1):
            sns.barplot(x=abs_loadings_df.index, y=abs_loadings_df[f"PC{n}"], ax=ax[n-1])
            ax[n-1].set_ylabel(f"PC{n} Loadings", labelpad=10)

        ax[n-1].set_xticklabels(range(1, len(abs_loadings_df) + 1), rotation=90)

        legend_labels = [f"{i+1}: {feature}" for i, feature in enumerate(loadings_df.index)]
        fig2.legend(legend_labels, loc="center right", title="Feature Legend", fontsize=legend_fontsize)

        fig2.supxlabel("Features (Mapped to Index Numbers)")

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.85, top=0.95, wspace=0.4, hspace=0.4)

        plt.savefig(plot_dir / f"{plot_fname}_loadings.png", dpi=600, bbox_inches="tight")

    return fig
