import pandas as pd
from pathlib import Path
import sys
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import random as rand
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr, pearsonr
import json


# % ========= Constants =========
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]

sys.path.insert(0, f"{str(FILE_DIR.parents[1])}/path")
from get_paths import getPaths


DESCRIPTOR_SET = "rdkit"
paths = getPaths()
feat_paths = paths["full_features"]
rdkit_feats = {"bp":feat_paths['bp']['rdkit'],
                "logd":feat_paths['logd']['rdkit'], 
                "pka":feat_paths['pka']['rdkit'],
                "ld50":feat_paths['ld50']['rdkit'], 
                "pic50":feat_paths['pic50']['rdkit']}

mordred_feats = {"bp":feat_paths['bp']['mordred'],
                "logd":feat_paths['logd']['mordred'], 
                "pka":feat_paths['pka']['mordred'],
                "ld50":feat_paths['ld50']['mordred'], 
                "pic50":feat_paths['pic50']['mordred']}

chemberta_feats = {"bp":feat_paths['bp']['chemberta'],
                "logd":feat_paths['logd']['chemberta'], 
                "pka":feat_paths['pka']['chemberta'],
                "ld50":feat_paths['ld50']['chemberta'], 
                "pic50":feat_paths['pic50']['chemberta']}

molformer_feats = {"bp":feat_paths['bp']['molformer'],
                "logd":feat_paths['logd']['molformer'], 
                "pka":feat_paths['pka']['molformer'],
                "ld50":feat_paths['ld50']['molformer'], 
                "pic50":feat_paths['pic50']['molformer']}


BP_features = feat_paths["bp"][DESCRIPTOR_SET]
LOGD_features = feat_paths["logd"][DESCRIPTOR_SET]
PKA_features = feat_paths["pka"][DESCRIPTOR_SET]
LD50_features = feat_paths["ld50"][DESCRIPTOR_SET]
PIC50_features = feat_paths["pic50"][DESCRIPTOR_SET]


def getLowVarianceColumns(
        input_df: str | Path,
        threshold: float = 0.95,
        index_col: str | None = "ID",
        exclude_columns: list[str] | None = None,
    ) -> list[str]:
    """
    Identify columns where the most common value accounts for at least
    ``threshold`` of all rows.

    Parameters
    ----------
    input_df : str | Path
        Path to the CSV file to inspect.
    threshold : float, optional
        Minimum fraction of rows occupied by the most common value for a column
        to be flagged. Default = 0.95.
    index_col : str | None, optional
        Column to use as the index when reading the CSV. Default = "ID".
    exclude_columns : list[str] | None, optional
        Columns to skip, e.g. metadata such as "SMILES". Default = None.

    Returns
    -------
    list[str]
        Column names flagged as near-constant by the threshold rule.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    exclude_columns = set(exclude_columns or [])

    df = pd.read_csv(input_df, index_col=index_col)

    low_variance_cols = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        dominant_fraction = df[col].value_counts(normalize=True, dropna=False).iloc[0]

        if dominant_fraction >= threshold:
            low_variance_cols.append(col)

    return low_variance_cols


def getLowVarianceSummary(
        input_df: str | Path,
        threshold: float = 0.95,
        index_col: str | None = "ID",
        exclude_columns: list[str] | None = None,
    ) -> pd.DataFrame:
    """
    Summarise the dominant-value fraction for each column in a CSV file.

    Returns a dataframe sorted in descending order of dominant-value fraction so
    near-constant columns appear first.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    exclude_columns = set(exclude_columns or [])
    df = pd.read_csv(input_df, index_col=index_col)

    summary_rows = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        value_fractions = df[col].value_counts(normalize=True, dropna=False)
        dominant_value = value_fractions.index[0]
        dominant_fraction = value_fractions.iloc[0]

        summary_rows.append({
            "feature": col,
            "dominant_value": dominant_value,
            "dominant_fraction": dominant_fraction,
            "n_unique": df[col].nunique(dropna=False),
            "flag_low_variance": dominant_fraction >= threshold,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by="dominant_fraction",
        ascending=False,
    ).reset_index(drop=True)

    return summary_df


def plotLowVarianceColumns(
        input_df: str | Path,
        threshold: float = 0.95,
        index_col: str | None = "ID",
        exclude_columns: list[str] | None = None,
        output_path: str | Path | None = None,
        save_name: str = "low_variance_features",
        top_n: int | None = None,
    ) -> pd.DataFrame:
    """
    Plot the dominant-value fraction for each feature as a descending bar chart.
    """

    summary_df = getLowVarianceSummary(
        input_df=input_df,
        threshold=threshold,
        index_col=index_col,
        exclude_columns=exclude_columns,
    )

    plot_df = summary_df.head(top_n).copy() if top_n is not None else summary_df.copy()

    fig_width = max(12, len(plot_df) * 0.22)
    plt.figure(figsize=(fig_width, 6), dpi=150)

    bar_colors = [
        "tab:red" if is_flagged else "tab:blue"
        for is_flagged in plot_df["flag_low_variance"]
    ]

    plt.bar(
        plot_df["feature"],
        plot_df["dominant_fraction"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"threshold = {threshold:.2f}",
    )

    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Dominant value fraction", fontsize=12)
    plt.title("Low-variance feature summary", fontsize=13)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.legend(frameon=False)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / f"{save_name}.png")
        plot_df.to_csv(output_path / f"{save_name}.csv")

    plt.close()

    return summary_df

def getAnalysisDescriptors(
        descriptors: list[str]=[
            "MolWt",
            "NumHDonors",
            "NumHAcceptors",
            "NumAromaticRings",
            "NumRotatableBonds"],
        datasets: dict={
            "bp": str(BP_features),
            "logd": str(LOGD_features),
            "pka": str(PKA_features),
            "ld50": str(LD50_features),
            "pic50": str(PIC50_features),},
        out_path = paths["dataset_analysis"]["descriptor_analysis"]["rdkit"]
    ):
    """
    Description
    ------------
    This function obtains a set of defined descriptors from feature data frames

    Parameters
    ----------
    descriptors list[str]       A list of the names of RDKit descriptor which you want to evaluate the datasets against
    datasets    dict            A dictionary containing the property and file path for the data. (as per the getPaths function)
    out_path    Path            The path to save the data to
    """



    dfs = []

    for name, pattern in datasets.items():
        temp_df = pd.DataFrame()
        print(name)
        print(pattern)
        
        for file in glob(str(pattern)):
            df = pd.read_csv(file, index_col="ID")
            df = df[descriptors].copy()
            df["source"] = name
            
            temp_df = pd.concat([temp_df, df])

        temp_df = temp_df.sort_index(
            key=lambda x: pd.to_numeric(
                x.astype(str).str.extract(r"(\d+)")[0],
                errors="coerce"
            ).fillna(10**18)
        )

        dfs.append(temp_df)

    descriptor_df = pd.concat(dfs)

    descriptor_df.to_csv(out_path, index_label="ID")


def plotDescriptorAnalysis(
        input_df: Path = paths["dataset_analysis"]["descriptor_analysis"]["rdkit"],
        descriptor_columns: list[str] = [
            "MolWt",
            "NumHDonors",
            "NumHAcceptors",
            "NumAromaticRings",
            "NumRotatableBonds"
        ],
        output_path: Path = paths["dataset_analysis"]["descriptor_analysis"]["rdkit"].parent
    ):

    """
    Description
    -----------
    Function to plot the output from getAnalysisDescriptors

    Parameters
    ----------
    input_df                Path        The data generated from getAnalysisDescriptors
    descriptor_columns      list[str]   The RDKit descriptor names in the data to plot
    output_path             Path        The path to save plots to
    """

    df = pd.read_csv(input_df, index_col="ID")

    output_path.mkdir(parents=True, exist_ok=True)

    for col in descriptor_columns:
        plt.figure(figsize=(6, 4), dpi=150)

        plt.hist(df[col], bins=50)

        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.title(f"Distribution of {col}", fontsize=13)

        # Clean axis look
        plt.tick_params(axis="both", labelsize=10)
        for spine in ["top", "right"]:
            plt.gca().spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / f"total_{col}_desc_analysis.png")
        plt.close()

def plotDescriptorAnalysisViolin(
        input_df: Path,
        descriptor_columns: list[str],
        output_path: Path,
        save_name: str = "descriptor_violin_plot"
    ):
    df = pd.read_csv(input_df, index_col="ID")

    missing = [col for col in descriptor_columns if col not in df.columns]
    if missing:
        raise ValueError(f"These descriptor columns are missing from the input data: {missing}")

    output_path.mkdir(parents=True, exist_ok=True)

    plot_df = df[descriptor_columns].copy()
    long_df = plot_df.melt(var_name="Descriptor", value_name="Value")

    plt.figure(figsize=(max(10, len(descriptor_columns) * 1.2), 6), dpi=150)
    sns.violinplot(
        data=long_df,
        x="Descriptor",
        y="Value",
        inner=None,
        cut=0,
    )

    plt.xlabel("Descriptor", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Descriptor Distributions", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path / f"{save_name}.png")
    plt.close()

def plotDescriptorAnalysisBySourceViolin(
    input_df: Path = paths["dataset_analysis"]["descriptor_analysis"]["rdkit"],
    descriptor_columns: list[str] = [
        "MolWt",
        "NumHDonors",
        "NumHAcceptors",
        "NumAromaticRings",
        "NumRotatableBonds",
    ],
    output_path: Path = paths["dataset_analysis"]["descriptor_analysis"]["rdkit"].parent,
    source_col: str = "source",
    order: list[str] | None = None,
    source_colors : dict={
        "bp": "tab:blue",
        "logd": "tab:orange",
        "pka": "tab:green",
        "ld50": "tab:red",
        "pic50": "tab:purple",
    }
    ):

    """
    Description
    -----------
    Function to be able to separate descriptor analysis by source (e.g., by property) and plot the
    distribution using violin plots

    Parameters
    ----------
    input_df                Path        The data generated from getAnalysisDescriptors
    descriptor_columns      list[str]   The RDKit descriptor names in the data to plot
    output_path             Path        The path to save plots to
    source_col              str         Name of the column to distinguish groups by
    order                   list[str]   List to order the violins on the plot by, must contain the source names
    source_colours          dict        Dictionary to define the colours of each source

    """

    df = pd.read_csv(input_df, index_col="ID")
    output_path.mkdir(parents=True, exist_ok=True)

    if source_col not in df.columns:
        raise ValueError(f"Expected a '{source_col}' column in {input_df}")

    # Default order: alphabetical sources as they appear
    if order is None:
        order = sorted(df[source_col].dropna().unique().tolist())

    for col in descriptor_columns:
        if col not in df.columns:
            continue

        plt.figure(figsize=(8, 4.5), dpi=200)

        # Violin plot: descriptor distributions per source
        sns.violinplot(
            data=df,
            x=source_col,
            y=col,
            order=order,
            inner="quartile",
            palette=source_colors,
            cut=0,
            linewidth=1,

        )

        plt.xlabel("")  # sources shown as tick labels
        plt.ylabel(col, fontsize=12)
        plt.title(f"{col} by property", fontsize=13)

        ax = plt.gca()
        ax.tick_params(axis="both", labelsize=10)
        ax.tick_params(axis="x", rotation=45)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_path / "violin_plots" / f"violin_{col}_by_source.png")
        plt.close()


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
    remove_outliers: bool = True,
    kdep_sample_ls: list = ["bp"],
    axis_fontsize: int = 20,
    tick_fontsize: int = 18,
    label_fontsize: int = 20,
    legend_fontsize: int = 20,
    kde_tick_dicts: list = None,
):
    """
    Description
    -----------
    Function to calculate and plot Principal Component Analysis (PCA) of multiple datasets. On the x=y axis there are
    Kernel Density Estimate (KDE) Plots which show the distribution of points across each component. This KDE is calculated
    from a random sample of data from the datasets (due to processing time). This function can also remove outliers based
    on the defined 'contamination' using the LocalOutlierFactor from Scikit-Learn.

    Parameters
    ----------
    datasets            dict[str, Path]     Dictionary containing the source name and file path of datasets to perform analysis on
    plot_dir            Path                Directory to save the PCA plot to
    n_components        int                 Number of principal components to calculate (default is 5)
    loadings_filename   str                 File name to save the individual loadings as
    pca_df_filename     str                 File name to save the PCA data to
    kdep_sample_size    float               Size of sample to take to generate KDE plots on the x=y axis
    contamination       float               The fraction of outliers to remove (e.g., 0.00001 = 0.001 % of the data)
    plot_fname          str                 Name to save the plot under
    save_plot           bool                Flag to save the plot (True = Save)
    save_extra_data     bool                Flag to save extra data (e.g., the complete loadings data frames)
    plot_area           bool                Flag to plot the shaded area covered by each PC
    plot_scatter        bool                Flag to plot the scatter points for each PC
    random_seed         int                 Random seed to set random processes to (if None, generates random seed itself)
    plot_loadings       bool                Flad to plot the PC loadings (NOT OPTIMISED, MESSY & CROWDED)
    remove_outliers     bool                Flag to remove the outliers using LocalOutlierFactor based on contamination fraction
    kdep_sample_ls      list                The list of sources to include in the KDEP
    axis_fontsize       int                 Size of fonts on the axis
    tick_fontsize       int                 Size of fonts for the ticks
    label_fontsize      int                 Size of fonts for labels
    legend_fontsize     int                 Size of fonts in the legend
    kde_tick_dicts      list                Dictionary to be able to self allocate the ticks on the KDEP
    """

    if random_seed is None:                         # Generating random seed if None
        random_seed = rand.randint(0, 2**31)

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Function load data frames with each file type
    def _load_one(inp: pd.DataFrame | str | Path) -> pd.DataFrame:
        if isinstance(inp, pd.DataFrame):
            df = inp.copy()
            if df.index.name != "ID":
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

    # Function used to remove outliers from a specific dataset
    def _remove_outliers(df: pd.DataFrame, columns: list[str], n_neighbors=20, contamination=contamination) -> pd.DataFrame:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(df[columns])
        return df[outlier_labels == 1]

    # Loading each dataset
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

    # Dropping non-common features across all data sets
    common_cols = None
    for df in loaded:
        cols = set(df.columns)
        cols -= {"Source"}
        common_cols = cols if common_cols is None else (common_cols & cols)

    common_cols = list(common_cols) if common_cols else []
    if not common_cols:
        raise ValueError("No common descriptor columns found across datasets.")

    loaded = [df[common_cols + ["Source"]] for df in loaded]

    # Combining all data frames, making sure largest is plotted at the bottom to not
    # plot over smaller sets
    loaded_sorted = sorted(loaded, key=len, reverse=True)
    combined_df = pd.concat(loaded_sorted, axis=0).dropna()

    # Scaling all features
    used_cols = [c for c in combined_df.columns if c != "Source"]
    scaler = StandardScaler()

    # Using only the columns containing numeric cells
    used_cols = [
    c for c in combined_df.columns
    if c != "Source" and pd.api.types.is_numeric_dtype(combined_df[c])
]
    scaled_X = scaler.fit_transform(combined_df[used_cols])

    scaled_combined_df = pd.DataFrame(scaled_X, columns=used_cols, index=combined_df.index)
    scaled_combined_df["Source"] = combined_df["Source"]

    # Performing PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_combined_df[used_cols])
    explained_variance = pca.explained_variance_ratio_ * 100

    # Calculating loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(n_components)], index=used_cols)
    abs_loadings_df = loadings_df.abs()

    loadings_df.rename_axis("Features", inplace=True)
    abs_loadings_df.rename_axis("Features", inplace=True)

    # Save loadings, if set
    if save_extra_data:
        loadings_df.to_csv(plot_dir / f"{loadings_filename}.csv", index_label="Features")
        abs_loadings_df.to_csv(plot_dir / f"{loadings_filename}_abs.csv", index_label="Features")

    # Creating the PCA data frame
    pca_df = pd.DataFrame(
        principal_components,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=combined_df.index
    )
    pca_df["Source"] = combined_df["Source"].values

    # Assigning colours
    dark_colours = sns.color_palette("dark", n_colors=max(len(source_ls), 3))
    source_color_map = {src: dark_colours[i] for i, src in enumerate(source_ls)}

    # Saving PCA data frame, if set
    if save_extra_data:
        pca_df.to_csv(plot_dir / f"{pca_df_filename}.csv.gz", index_label="ID", compression="gzip")

    # Removing outliers, if set
    if remove_outliers:
        pca_df = _remove_outliers(pca_df, [f"PC{i+1}" for i in range(n_components)])

    # Plotting the PCA
    fig, axs = plt.subplots(nrows=n_components, ncols=n_components, figsize=(20, 20))

    for i in range(n_components):
        for j in range(n_components):

            if i != j:
                # Plotting Scatter
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

                    # Plotting Area
                    if plot_area and len(source_data) >= 3:#
                        points = source_data[[f"PC{j+1}", f"PC{i+1}"]].values
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

                # Formatting the axs labels
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
                # Creating KDEP for X=Y
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

    # Creating the legend
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
    
    # Small adjustments to plot format
    plt.tight_layout()
    plt.subplots_adjust(
        left=0.1,
        right=0.9,
        top=0.95,
        bottom=0.3,
        wspace=0.6,
        hspace=0.6
    )

    # Saving the PCA plot
    if save_plot:
        plt.savefig(plot_dir / f"{plot_fname}.png", dpi=600, bbox_inches="tight")

    # Plotting and saving the PCA loadings (UNOPTIMISED)
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


def getSimilarities(
    descriptor_sets: list[str],
    sample_set: str = "all",
    sample_size: int = 4000,
    data_paths: list[str | Path] = None,
    random_seed: int = 42,
    scale: bool = True,
    cor_save_path: str = str(PROJ_DIR / "datasets" / "all" / "descriptor_analysis")
):
    # Load tables
    if data_paths is None:
        data_paths = [feat_paths["all"][name] for name in descriptor_sets]

    if len(descriptor_sets) != len(data_paths):
        raise ValueError("descriptor_sets and data_paths must have the same length")

    tables = {}
    for name, data_path in zip(descriptor_sets, data_paths):
        tmp_df = pd.read_csv(data_path, index_col=0)
        tables[name] = tmp_df

    # Find shared index across all tables
    shared_idx = pd.Index(tables[descriptor_sets[0]].index)
    for name in descriptor_sets[1:]:
        shared_idx = shared_idx.intersection(tables[name].index)

    if sample_set != "all":
        mask = shared_idx.astype(str).str.contains(sample_set, regex=False)
        shared_idx = shared_idx[mask]

    if sample_size > len(shared_idx):
        raise ValueError(
            f"sample_size ({sample_size}) is larger than available rows ({len(shared_idx)})"
        )

    sampled_idx = (
        pd.Series(shared_idx)
        .sample(n=sample_size, random_state=random_seed, replace=False)
        .values
    )

    # Subset all tables to the same sampled molecules
    for name in descriptor_sets:
        tables[name] = tables[name].loc[sampled_idx]

    # Helpers
    def _clean_numeric_table(df: pd.DataFrame, scale: bool = True) -> pd.DataFrame:
        # Coerce everything possible to numeric
        df = df.apply(pd.to_numeric, errors="coerce")

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number]).copy()

        # Drop columns with any missing values
        df = df.dropna(axis=1)

        # Drop zero-variance columns
        nunique = df.nunique(dropna=False)
        df = df.loc[:, nunique > 1]

        if df.shape[1] == 0:
            raise ValueError("No usable numeric columns remain after cleaning.")

        if scale:
            scaler = StandardScaler()
            df[:] = scaler.fit_transform(df.values)

        return df

    def _cosine_similarity(similarity_df: pd.DataFrame) -> pd.DataFrame:
        sim = cosine_similarity(similarity_df.values)
        return pd.DataFrame(sim, index=similarity_df.index, columns=similarity_df.index)

    def _upper_triangle(similarity_df: pd.DataFrame) -> np.ndarray:
        arr = similarity_df.values
        iu = np.triu_indices_from(arr, k=1)
        return arr[iu]

    def _compare_similarity_spaces(tables: dict[str, pd.DataFrame]):
        cleaned = {}
        sims = {}

        for name, rep_df in tables.items():
            cleaned[name] = _clean_numeric_table(rep_df, scale=scale)
            sims[name] = _cosine_similarity(cleaned[name])

        names = list(sims.keys())
        correlations = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                va = _upper_triangle(sims[a])
                vb = _upper_triangle(sims[b])

                rho_s, pval_s = spearmanr(va, vb)
                rho_p, pval_p = pearsonr(va, vb)

                correlations[(a, b)] = {
                    "spearman_r": rho_s,
                    "spearman_p_value": pval_s,
                    "pearson_r": rho_p,
                    "pearson_p_value": pval_p,
                }

        return cleaned, sims, correlations

    cleaned, sims, correlations = _compare_similarity_spaces(tables)

    def _save_correlations_json(correlations, path):
        json_ready = {
            f"{a}__{b}": stats
            for (a,b), stats in correlations.items()
        }

        save_path = path + "/similarity_correlations.json"
        with open(save_path, "w") as f:
            json.dump(json_ready, f, indent=4)

    _save_correlations_json(correlations=correlations, path=cor_save_path)

    return cleaned, sims, correlations

def plotSimilarityHeatmaps(
    sims: dict[str, pd.DataFrame],
    out_dir: str | Path = PROJ_DIR / "datasets" / "all" / "descriptor_analysis",
    cmap: str = "viridis",
    figsize: tuple = (8, 6),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, sim_df in sims.items():
        plt.figure(figsize=figsize, dpi=200)

        sns.heatmap(
            sim_df,
            cmap=cmap,
            square=True,
            cbar_kws={"label": "Cosine similarity"},
            xticklabels=False,
            yticklabels=False,
        )

        plt.title(f"{name} similarity heatmap")
        plt.xlabel("Molecule index")
        plt.ylabel("Molecule index")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_similarity_heatmap.png")
        plt.close()


# cleaned, sims, correlations = getSimilarities(
#     descriptor_sets=["rdkit", "mordred", "molformer", "chemberta"],
#     sample_size=5000,
#     random_seed=42,
#     scale=True,
# )

# plotSimilarityHeatmaps(sims)

# plotDescriptorAnalysisViolin(
#         input_df = "/users/yhb18174/TL_project/datasets/all/all_rdkit.csv",
#         descriptor_columns=
#         ["Kappa3_rdkit"],
#         output_path=Path("/users/yhb18174/TL_project/datasets/all/descriptor_analysis"),
#         save_name="kappa3_distribution"
        
#     )



#getAnalysisDescriptors()
#plotDescriptorAnalysis()
#plotDescriptorAnalysisBySourceViolin()


# Plotting PCA for each dataset
plot_pcas=False
if plot_pcas:
    fig = plotPCA(
        datasets = rdkit_feats,
        plot_dir=paths["dataset_analysis"]["descriptor_analysis"]["rdkit"].parent,
        plot_fname="rdkit_PCA"
    )

    fig = plotPCA(
        datasets = mordred_feats,
        plot_dir=paths["dataset_analysis"]["descriptor_analysis"]["mordred"].parent,
        plot_fname="mordred_PCA"
    )

    fig = plotPCA(
        datasets = chemberta_feats,
        plot_dir=paths["dataset_analysis"]["descriptor_analysis"]["chemberta"].parent,
        plot_fname="chemberta_PCA"
    )

    fig = plotPCA(
        datasets = molformer_feats,
        plot_dir=paths["dataset_analysis"]["descriptor_analysis"]["molformer"].parent,
        plot_fname="molformer_PCA"
    )
