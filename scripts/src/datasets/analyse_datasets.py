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
import networkx as nx

# % ========= Constants =========
FILE_DIR = Path(__file__).parent
SRC_DIR = FILE_DIR.parent
PROJ_DIR = SRC_DIR.parent.parent

sys.path.insert(0, f"{str(SRC_DIR)}/pathing/")
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
        title: str="Low-variance feature summary"
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
    plt.title(title, fontsize=13)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=10)
    plt.legend(frameon=False)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path / f"{save_name}.png",
            dpi=600,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Description": f"Bar plot showing the value variance compared to the most common value \
                    for each feature. Threshold = {threshold}"
            }
            )
        plot_df.to_csv(output_path / f"{save_name}.csv")

    plt.close()

    return summary_df

def getOutlierSummary(df, exclude_columns=None):
    exclude_columns = set(exclude_columns or [])
    rows = []

    for col in df.select_dtypes(include=[np.number]).columns:
        if col in exclude_columns:
            continue

        s = df[col].dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        p50 = s.quantile(0.50)
        p95 = s.quantile(0.95)
        p99 = s.quantile(0.99)

        upper_iqr = q3 + 1.5 * iqr
        n_upper_outliers = (s > upper_iqr).sum()

        row = {
            "feature": col,
            "max": s.max(),
            "p95": p95,
            "p99": p99,
            "median": p50,
            "skew": s.skew(),
            "upper_iqr_bound": upper_iqr,
            "n_upper_outliers": int(n_upper_outliers),
            "max_to_p95": s.max() / p95 if p95 not in [0, np.nan] else np.nan,
            "p99_to_median": p99 / p50 if p50 not in [0, np.nan] else np.nan,
        }
        rows.append(row)
        print(row)

    return pd.DataFrame(rows).sort_values(
        by=["n_upper_outliers", "max_to_p95", "skew"],
        ascending=False
    )


def trimRowsByPercentile(
        input_df: str | Path | pd.DataFrame,
        columns: list[str] | None = None,
        percentile: float = 0.99,
        tail: str = "upper",
        index_col: str | None = "ID",
        exclude_columns: list[str] | None = None,
        return_removed_rows: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows whose values fall outside a percentile cutoff.

    This is intended for trimming rows with extreme feature values before model
    training. By default, it removes rows above the 99th percentile for the
    selected numeric columns.
    """

    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    if tail not in {"upper", "lower", "both"}:
        raise ValueError("tail must be one of: 'upper', 'lower', 'both'.")

    if isinstance(input_df, pd.DataFrame):
        df = input_df.copy()
    else:
        df = pd.read_csv(input_df, index_col=index_col)

    exclude_columns = set(exclude_columns or [])

    if columns is None:
        candidate_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in candidate_columns if col not in exclude_columns]
    else:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(f"These columns are missing from the input data: {missing}")

    if not columns:
        raise ValueError("No columns available for percentile trimming.")

    keep_mask = pd.Series(True, index=df.index)

    for col in columns:
        s = df[col]

        if not pd.api.types.is_numeric_dtype(s):
            raise TypeError(f"Column '{col}' must be numeric for percentile trimming.")

        lower_cutoff = s.quantile(1 - percentile)
        upper_cutoff = s.quantile(percentile)

        if tail == "upper":
            keep_mask &= s.le(upper_cutoff) | s.isna()
        elif tail == "lower":
            keep_mask &= s.ge(lower_cutoff) | s.isna()
        else:
            keep_mask &= s.between(lower_cutoff, upper_cutoff) | s.isna()

    trimmed_df = df.loc[keep_mask].copy()

    if return_removed_rows:
        removed_df = df.loc[~keep_mask].copy()
        return trimmed_df, removed_df

    return trimmed_df



def collect_outlier_details(
    feature_set: str,
    input_path: str | Path,
    percentile: float,
    tail: str,
    columns: list[str] | None=None,
) -> pd.DataFrame:
    input_df = pd.read_csv(input_path, index_col="ID", low_memory=False)

    _, removed_df = trimRowsByPercentile(
        input_df=input_df,
        columns=columns,
        percentile=percentile,
        tail=tail,
        return_removed_rows=True,
    )

    if columns is None:
        columns = input_df.select_dtypes(include="number").columns.tolist()

    detail_rows = []

    for col in columns:
        s = input_df[col]

        if not pd.api.types.is_numeric_dtype(s):
            continue

        lower_cutoff = s.quantile(1 - percentile)
        upper_cutoff = s.quantile(percentile)

        if tail == "upper":
            mask = s.gt(upper_cutoff)
            cutoff = upper_cutoff
            reason = f"value > {percentile:.0%} percentile"
        elif tail == "lower":
            mask = s.lt(lower_cutoff)
            cutoff = lower_cutoff
            reason = f"value < {1 - percentile:.0%} percentile"
        else:
            mask = s.lt(lower_cutoff) | s.gt(upper_cutoff)
            reason = "value outside percentile band"

        flagged = input_df.loc[mask, [col]].copy()
        if flagged.empty:
            continue

        for idx, value in flagged[col].items():
            row = {
                "ID": str(idx),
                "feature_set": feature_set,
                "column": col,
                "tail": tail,
                "value": value,
                "reason": reason,
            }

            if tail == "both":
                row["lower_cutoff"] = lower_cutoff
                row["upper_cutoff"] = upper_cutoff
            else:
                row["cutoff"] = cutoff

            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    detail_df = detail_df[detail_df["ID"].isin(removed_df.index.astype(str))].copy()
    detail_df = detail_df.sort_values(by=["ID", "feature_set", "column"]).reset_index(drop=True)

    return detail_df

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

def checkLipinskiCriteria(
        df: str | Path | pd.DataFrame,
        mw: int=600,
        logp: float=6,
        n_hbd: int=6,
        n_hba: int=11,
        columns: list[str]=[
        "MolWt_rdkit", "MolLogP_rdkit", "NumHDonors_rdkit", "NumHAcceptors_rdkit",
        ]
) -> list[str]:
    """Function to check how many molecules fit the Lipinski Ro5, default to 'relaxed' criteria"""
    
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df, index_col=0)

    n_orig = len(df)
    print(f"Number of original molecules:\n{n_orig}")
    
    trimmed_df = df
    criteria = [mw, logp, n_hbd, n_hba]

    for i, desc in enumerate(columns):
        trimmed_df = trimmed_df[trimmed_df[desc] < criteria[i]]
        if trimmed_df.empty:
            raise ValueError("Empty dataframe, make sure that the columns list\
                             is ordered: mw, logp, hb donor, hb acceptor")
        
    n_pass = len(trimmed_df)

    print(f"Number of molecules fitting Lipinski Criteria:\n \
          {n_pass} ({round((n_pass/n_orig)*100, 2)} %)")


    return trimmed_df.index

def featNetworkCorrelation(df, threshold=0.3, save_path="/users/yhb18174/TL_project/results/test_network.png"):
    corr_matrix = df.corr(numeric_only=True)

    G = nx.Graph()
    variables = list(corr_matrix.columns)
    G.add_nodes_from(variables)

    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            corr = corr_matrix.iloc[i, j]
            if pd.notna(corr) and abs(corr) > threshold:
                G.add_edge(variables[i], variables[j], weight=float(corr))

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=500, alpha=0.9)

    edges = G.edges(data=True)
    edge_colors = ["tab:red" if d["weight"] > 0 else "tab:blue" for _, _, d in edges]
    edge_widths = [1 + 4 * abs(d["weight"]) for _, _, d in edges]

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# featNetworkCorrelation(pd.read_csv(paths["full_features"]["all"]["rdkit"], index_col=0))

# def featArcCorrelationDiagram(
#     df,
#     threshold=0.3,
#     max_labels=80,
#     figsize=(18, 6),
#     save_path="/users/yhb18174/TL_project/results/test_arc_network.png",
# ):
#     """
#     Arc diagram: features on a line, arcs connect correlated feature pairs.
#     Red = positive correlation, Blue = negative correlation.
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     from matplotlib.patches import Arc

#     corr_matrix = df.corr(numeric_only=True)
#     features = list(corr_matrix.columns)

#     # Optional cap for readability
#     if len(features) > max_labels:
#         var_rank = corr_matrix.var(axis=0).sort_values(ascending=False).index[:max_labels]
#         corr_matrix = corr_matrix.loc[var_rank, var_rank]
#         features = list(corr_matrix.columns)

#     n = len(features)
#     x_pos = np.arange(n)

#     # Collect edges above threshold
#     edges = []
#     for i in range(n):
#         for j in range(i + 1, n):
#             c = corr_matrix.iloc[i, j]
#             if pd.notna(c) and abs(c) >= threshold:
#                 edges.append((i, j, float(c)))

#     plt.figure(figsize=figsize)
#     ax = plt.gca()

#     # Draw baseline and nodes
#     ax.hlines(0, 0, n - 1, color="black", linewidth=1)
#     ax.scatter(x_pos, np.zeros(n), s=25, color="black", zorder=3)

#     # Draw arcs
#     for i, j, c in edges:
#         mid = (i + j) / 2
#         width = j - i
#         height = max(0.2, width * 0.35)
#         color = "tab:red" if c > 0 else "tab:blue"
#         lw = 0.6 + 2.5 * abs(c)
#         arc = Arc((mid, 0), width=width, height=height, angle=0,
#                   theta1=0, theta2=180, color=color, linewidth=lw, alpha=0.6)
#         ax.add_patch(arc)

#     # Labels
#     for i, f in enumerate(features):
#         ax.text(i, -0.08, f, rotation=90, ha="right", va="top", fontsize=7)

#     ax.set_xlim(-1, n)
#     ax.set_ylim(-0.2, max([1.0] + [((j - i) * 0.35) for i, j, _ in edges]) + 0.5)
#     ax.axis("off")
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300)
#     plt.close()


from group_descriptors import getGroups
import matplotlib.pyplot as plt

def build_feature_group_map(feature_set: str):
    group_map = getGroups(feature_set)  # {group_name: [feature1, feature2, ...]}
    feat_to_group = {}
    for g, feats in group_map.items():
        for f in feats:
            feat_to_group[f] = g
    return feat_to_group

def _strip_last_token(name: str) -> str:
    """Split by '_' and drop the final token: emb_1_chemberta -> emb_1."""
    parts = str(name).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 else str(name)

def _resolve_group(feature_name: str, feat_to_group: dict) -> str | None:
    """Resolve group by exact feature name, then by stripped fallback."""
    if feature_name in feat_to_group:
        return feat_to_group[feature_name]
    stripped = _strip_last_token(feature_name)
    return feat_to_group.get(stripped)


def featArcCorrelationDiagram(
    df,
    feature_set="rdkit",
    threshold=0.6,
    max_labels=120,
    figsize=(20, 7),
    label_fontsize=12,
    connection_width=1.0,
    save_path="/users/yhb18174/TL_project/results/test_arc_network.png",
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from group_descriptors import getGroups

    # --- correlations
    corr_matrix = df.corr(numeric_only=True)

    # --- feature->group map from getGroups
    group_map = getGroups(feature_set)  # {group: [feature names]}
    feat_to_group = {}
    for g, feats in group_map.items():
        for f in feats:
            feat_to_group[f] = g

    # keep only columns that can be resolved to a descriptor group
    features = [f for f in corr_matrix.columns if _resolve_group(f, feat_to_group) is not None]
    if len(features) == 0:
        raise ValueError(f"No columns matched groups for feature_set='{feature_set}'")

    # optional cap by variance for readability
    if len(features) > max_labels:
        var_rank = corr_matrix[features].var(axis=0).sort_values(ascending=False).index[:max_labels]
        features = list(var_rank)

    # sort by group, then name (makes grouped blocks on x-axis)
    features = sorted(features, key=lambda f: (_resolve_group(f, feat_to_group) or "zzz", f))
    corr_matrix = corr_matrix.loc[features, features]

    # group colors
    groups = sorted({_resolve_group(f, feat_to_group) for f in features})
    cmap = plt.get_cmap("tab20")
    group_color = {g: cmap(i % 20) for i, g in enumerate(groups)}
    node_colors = [group_color[_resolve_group(f, feat_to_group)] for f in features]

    # build edge list
    edges = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if pd.notna(c) and abs(c) >= threshold:
                edges.append((i, j, float(c)))

    def _draw_arc_plot(features_local, node_colors_local, edges_local, out_path):
        n_local = len(features_local)
        x = np.arange(n_local, dtype=float)

        fig, ax = plt.subplots(figsize=figsize)
        ax.hlines(0, 0, max(0, n_local - 1), color="black", linewidth=0.5, alpha=0.4)
        ax.scatter(x, np.zeros(n_local), s=24, c=node_colors_local, zorder=3)

        def draw_gradient_arc(i, j, corr):
            x0, x1 = float(i), float(j)
            mid = 0.5 * (x0 + x1)
            h = max(0.10, (x1 - x0) * 0.20)
            t = np.linspace(0, 1, 45)
            xs = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid + t ** 2 * x1
            ys = 2 * (1 - t) * t * h

            segs = np.stack(
                [np.column_stack([xs[:-1], ys[:-1]]), np.column_stack([xs[1:], ys[1:]])],
                axis=1
            )

            c0 = np.array(node_colors_local[i])
            c1 = np.array(node_colors_local[j])
            cols = np.array([c0 * (1 - u) + c1 * u for u in np.linspace(0, 1, len(segs))])

            lw = connection_width * (0.2 + 0.8 * abs(corr))  # thin lines
            lc = LineCollection(segs, colors=cols, linewidths=lw, alpha=0.55, zorder=2)
            ax.add_collection(lc)

        for i, j, corr in edges_local:
            draw_gradient_arc(i, j, corr)

        degree = {i: 0 for i in range(n_local)}
        for i, j, _ in edges_local:
            degree[i] += 1
            degree[j] += 1

        for i, f in enumerate(features_local):
            ax.text(
                i,
                -0.06,
                f"{_strip_last_token(f)} ({degree.get(i, 0)})",
                rotation=90,
                ha="right",
                va="top",
                fontsize=label_fontsize,
            )

        if features_local:
            prev = _resolve_group(features_local[0], feat_to_group)
            for i, f in enumerate(features_local[1:], start=1):
                g = _resolve_group(f, feat_to_group)
                if g != prev:
                    ax.vlines(i - 0.5, -0.02, 0.18, color="grey", linewidth=0.6, alpha=0.6)
                    prev = g

        ymax = max([0.6] + [max(0.10, (j - i) * 0.20) for i, j, _ in edges_local]) + 0.2
        ax.set_xlim(-1, n_local)
        ax.set_ylim(-0.18, ymax)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # Full plot output path
    base_path = Path(save_path)
    full_path = base_path.with_name(f"{base_path.stem}_full{base_path.suffix}")
    connected_only_path = base_path.with_name(f"{base_path.stem}_connected_only{base_path.suffix}")

    # Plot 1: all features
    _draw_arc_plot(features, node_colors, edges, full_path)

    # Plot 2: remove completely uncorrelated features (degree 0)
    connected_idx = set()
    for i, j, _ in edges:
        connected_idx.add(i)
        connected_idx.add(j)

    kept = sorted(connected_idx)
    if kept:
        features_connected = [features[i] for i in kept]
        colors_connected = [node_colors[i] for i in kept]
        idx_map = {old_i: new_i for new_i, old_i in enumerate(kept)}
        edges_connected = [(idx_map[i], idx_map[j], c) for i, j, c in edges if i in idx_map and j in idx_map]
        _draw_arc_plot(features_connected, colors_connected, edges_connected, connected_only_path)
    else:
        # If no edges pass threshold, still save an empty skeleton plot for transparency.
        _draw_arc_plot([], [], [], connected_only_path)

    print(f"Saved full arc diagram to: {full_path}")
    print(f"Saved connected-only arc diagram to: {connected_only_path}")

    return str(full_path), str(connected_only_path)



def featCircularCorrelationDiagram(
    df,
    feature_set="rdkit",
    threshold=0.6,
    max_labels=200,
    figsize=(10, 10),
    label_fontsize=12,
    connection_width=1.0,
    save_path="/users/yhb18174/TL_project/results/test_circular_network.png",
):
    """
    Circular correlation diagram with grouped node colours.
    Saves two plots:
    - *_full.png
    - *_connected_only.png (removes degree-0 nodes)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    corr_matrix = df.corr(numeric_only=True)

    group_map = getGroups(feature_set)
    feat_to_group = {}
    for g, feats in group_map.items():
        for f in feats:
            feat_to_group[f] = g

    features = [f for f in corr_matrix.columns if _resolve_group(f, feat_to_group) is not None]
    if len(features) == 0:
        raise ValueError(f"No columns matched groups for feature_set='{feature_set}'")

    if len(features) > max_labels:
        var_rank = corr_matrix[features].var(axis=0).sort_values(ascending=False).index[:max_labels]
        features = list(var_rank)

    features = sorted(features, key=lambda f: (_resolve_group(f, feat_to_group) or "zzz", f))
    corr_matrix = corr_matrix.loc[features, features]

    groups = sorted({_resolve_group(f, feat_to_group) for f in features})
    cmap = plt.get_cmap("tab20")
    group_color = {g: cmap(i % 20) for i, g in enumerate(groups)}
    node_colors = [group_color[_resolve_group(f, feat_to_group)] for f in features]

    edges = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if pd.notna(c) and abs(c) >= threshold:
                edges.append((i, j, float(c)))

    def _draw_circular_plot(features_local, node_colors_local, edges_local, out_path):
        n_local = len(features_local)
        fig, ax = plt.subplots(figsize=figsize)

        if n_local == 0:
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            return

        theta = np.linspace(0, 2 * np.pi, n_local, endpoint=False)
        radius = 1.0
        positions = {i: np.array([radius * np.cos(t), radius * np.sin(t)]) for i, t in enumerate(theta)}

        for i, j, corr in edges_local:
            p0 = positions[i]
            p2 = positions[j]
            control = 0.18 * (p0 + p2)

            t = np.linspace(0, 1, 40)
            pts = ((1 - t)[:, None] ** 2) * p0 + (2 * (1 - t)[:, None] * t[:, None]) * control + (t[:, None] ** 2) * p2
            segs = np.stack([pts[:-1], pts[1:]], axis=1)

            c0 = np.array(node_colors_local[i])
            c1 = np.array(node_colors_local[j])
            cols = np.array([c0 * (1 - u) + c1 * u for u in np.linspace(0, 1, len(segs))])

            lw = connection_width * (0.2 + 0.8 * abs(corr))
            lc = LineCollection(segs, colors=cols, linewidths=lw, alpha=0.5, zorder=1)
            ax.add_collection(lc)

        degree = {i: 0 for i in range(n_local)}
        for i, j, _ in edges_local:
            degree[i] += 1
            degree[j] += 1

        xy = np.array([positions[i] for i in range(n_local)])
        ax.scatter(xy[:, 0], xy[:, 1], s=28, c=node_colors_local, zorder=3)

        for i, feat in enumerate(features_local):
            x, y = positions[i]
            x_lab, y_lab = 1.12 * x, 1.12 * y
            angle = np.degrees(np.arctan2(y, x))
            rotation = angle if -90 <= angle <= 90 else angle + 180
            ha = "left" if x >= 0 else "right"
            ax.text(
                x_lab,
                y_lab,
                f"{_strip_last_token(feat)} ({degree.get(i, 0)})",
                fontsize=label_fontsize,
                rotation=rotation,
                rotation_mode="anchor",
                ha=ha,
                va="center",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    base_path = Path(save_path)
    full_path = base_path.with_name(f"{base_path.stem}_full{base_path.suffix}")
    connected_only_path = base_path.with_name(f"{base_path.stem}_connected_only{base_path.suffix}")

    _draw_circular_plot(features, node_colors, edges, full_path)

    connected_idx = set()
    for i, j, _ in edges:
        connected_idx.add(i)
        connected_idx.add(j)
    kept = sorted(connected_idx)

    if kept:
        features_connected = [features[i] for i in kept]
        colors_connected = [node_colors[i] for i in kept]
        idx_map = {old_i: new_i for new_i, old_i in enumerate(kept)}
        edges_connected = [(idx_map[i], idx_map[j], c) for i, j, c in edges if i in idx_map and j in idx_map]
        _draw_circular_plot(features_connected, colors_connected, edges_connected, connected_only_path)
    else:
        _draw_circular_plot([], [], [], connected_only_path)

    print(f"Saved full circular diagram to: {full_path}")
    print(f"Saved connected-only circular diagram to: {connected_only_path}")

    return str(full_path), str(connected_only_path)

def featLeftRightCorrelationDiagram(
    df_left,
    df_right,
    left_feature_set="rdkit",
    right_feature_set="chemberta",
    threshold=0.6,
    max_labels=120,
    figsize=(14, 12),
    save_path="/users/yhb18174/TL_project/results/left_right_corr.png",
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from group_descriptors import getGroups

    def _strip_last_token(name: str) -> str:
        parts = str(name).split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else str(name)

    def _build_group_map(feature_set: str):
        gmap = getGroups(feature_set)
        out = {}
        for g, feats in gmap.items():
            for f in feats:
                out[f] = g
        return out

    def _resolve_group(name: str, fmap: dict):
        if name in fmap:
            return fmap[name]
        return fmap.get(_strip_last_token(name), None)

    # Numeric only
    L = df_left.select_dtypes(include=[np.number]).copy()
    R = df_right.select_dtypes(include=[np.number]).copy()

    # Align molecules by index
    common_ids = L.index.intersection(R.index)
    L = L.loc[common_ids]
    R = R.loc[common_ids]

    if len(common_ids) == 0:
        raise ValueError("No shared molecule IDs between left and right dataframes.")

    # Optional cap for readability
    if L.shape[1] > max_labels:
        L = L[L.var().sort_values(ascending=False).index[:max_labels]]
    if R.shape[1] > max_labels:
        R = R[R.var().sort_values(ascending=False).index[:max_labels]]

    # Cross-correlation matrix (left cols x right cols)
    X = np.corrcoef(L.to_numpy().T, R.to_numpy().T)
    nL = L.shape[1]
    cross = X[:nL, nL:]  # shape: (left_features, right_features)

    left_feats = list(L.columns)
    right_feats = list(R.columns)

    left_group = _build_group_map(left_feature_set)
    right_group = _build_group_map(right_feature_set)

    # Sort by group then name
    left_feats = sorted(left_feats, key=lambda f: (_resolve_group(f, left_group) or "zzz", f))
    right_feats = sorted(right_feats, key=lambda f: (_resolve_group(f, right_group) or "zzz", f))

    # Reindex cross matrix to sorted order
    iL = [list(L.columns).index(f) for f in left_feats]
    iR = [list(R.columns).index(f) for f in right_feats]
    cross = cross[np.ix_(iL, iR)]

    # Colors by group; fallback black if ungrouped
    cmap = plt.get_cmap("tab20")

    left_groups = sorted({g for g in (_resolve_group(f, left_group) for f in left_feats) if g is not None})
    right_groups = sorted({g for g in (_resolve_group(f, right_group) for f in right_feats) if g is not None})

    left_group_color = {g: cmap(i % 20) for i, g in enumerate(left_groups)}
    right_group_color = {g: cmap(i % 20) for i, g in enumerate(right_groups)}

    left_colors = [left_group_color.get(_resolve_group(f, left_group), "black") for f in left_feats]
    right_colors = [right_group_color.get(_resolve_group(f, right_group), "black") for f in right_feats]

    # Positions
    yL = np.linspace(1, 0, len(left_feats))
    yR = np.linspace(1, 0, len(right_feats))
    xL, xR = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # Nodes
    ax.scatter(np.full(len(left_feats), xL), yL, c=left_colors, s=40, zorder=3)
    ax.scatter(np.full(len(right_feats), xR), yR, c=right_colors, s=40, zorder=3)

    # Labels
    for i, f in enumerate(left_feats):
        ax.text(xL - 0.02, yL[i], _strip_last_token(f), ha="right", va="center", fontsize=11)
    for j, f in enumerate(right_feats):
        ax.text(xR + 0.02, yR[j], _strip_last_token(f), ha="left", va="center", fontsize=11)

    # Edges (only strong cross-correlations)
    segs = []
    cols = []
    widths = []
    for i in range(len(left_feats)):
        for j in range(len(right_feats)):
            c = cross[i, j]
            if np.isfinite(c) and abs(c) >= threshold:
                segs.append([(xL, yL[i]), (xR, yR[j])])
                cols.append("tab:red" if c > 0 else "tab:blue")
                widths.append(0.2 + 1.2 * abs(c))

    if segs:
        lc = LineCollection(segs, colors=cols, linewidths=widths, alpha=0.35, zorder=1)
        ax.add_collection(lc)

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def featBipartiteCorrelationDiagram(
    df_left,
    df_right,
    left_feature_set="rdkit",
    right_feature_set="chemberta",
    threshold=0.6,
    max_labels=120,
    figsize=(14, 12),
    label_fontsize=11,
    connection_width=1.0,
    save_path="/users/yhb18174/TL_project/results/left_right_corr.png",
):
    """
    Bipartite left-right correlation diagram.

    - Left feature set nodes are arranged vertically on the left.
    - Right feature set nodes are arranged vertically on the right.
    - Edges are drawn for cross-feature correlations with |r| >= threshold.
    - Nodes not resolved to descriptor groups are coloured black.
    - Saves two plots each call:
      *_full.png and *_connected_only.png
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    L = df_left.select_dtypes(include=[np.number]).copy()
    R = df_right.select_dtypes(include=[np.number]).copy()

    common_ids = L.index.intersection(R.index)
    L = L.loc[common_ids]
    R = R.loc[common_ids]

    if len(common_ids) == 0:
        raise ValueError("No shared molecule IDs between left and right dataframes.")

    if L.shape[1] > max_labels:
        keep_left = L.var(axis=0).sort_values(ascending=False).index[:max_labels]
        L = L[keep_left]
    if R.shape[1] > max_labels:
        keep_right = R.var(axis=0).sort_values(ascending=False).index[:max_labels]
        R = R[keep_right]

    try:
        left_group_map = build_feature_group_map(left_feature_set)
    except Exception:
        left_group_map = {}

    try:
        right_group_map = build_feature_group_map(right_feature_set)
    except Exception:
        right_group_map = {}

    left_features = [f for f in L.columns]
    right_features = [f for f in R.columns]

    left_features = sorted(left_features, key=lambda f: (_resolve_group(f, left_group_map) or "zzz", f))
    right_features = sorted(right_features, key=lambda f: (_resolve_group(f, right_group_map) or "zzz", f))

    L = L[left_features]
    R = R[right_features]

    corr_full = np.corrcoef(L.to_numpy(dtype=float).T, R.to_numpy(dtype=float).T)
    nL = L.shape[1]
    cross_corr = corr_full[:nL, nL:]

    left_groups = sorted({g for g in (_resolve_group(f, left_group_map) for f in left_features) if g is not None})
    right_groups = sorted({g for g in (_resolve_group(f, right_group_map) for f in right_features) if g is not None})

    cmap = plt.get_cmap("tab20")
    left_group_color = {g: cmap(i % 20) for i, g in enumerate(left_groups)}
    right_group_color = {g: cmap(i % 20) for i, g in enumerate(right_groups)}

    left_colors = [left_group_color.get(_resolve_group(f, left_group_map), "black") for f in left_features]
    right_colors = [right_group_color.get(_resolve_group(f, right_group_map), "black") for f in right_features]

    edges = []
    for i in range(len(left_features)):
        for j in range(len(right_features)):
            c = cross_corr[i, j]
            if np.isfinite(c) and abs(c) >= threshold:
                edges.append((i, j, float(c)))

    def _draw_bipartite_plot(l_feats, r_feats, l_cols, r_cols, e_list, out_path):
        fig, ax = plt.subplots(figsize=figsize)

        if len(l_feats) == 0 and len(r_feats) == 0:
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            return

        yL = np.linspace(1.0, 0.0, max(1, len(l_feats)))
        yR = np.linspace(1.0, 0.0, max(1, len(r_feats)))
        xL, xR = 0.0, 1.0

        left_degree = {i: 0 for i in range(len(l_feats))}
        right_degree = {j: 0 for j in range(len(r_feats))}
        for i, j, _ in e_list:
            if i in left_degree:
                left_degree[i] += 1
            if j in right_degree:
                right_degree[j] += 1

        if len(l_feats) > 0:
            ax.scatter(np.full(len(l_feats), xL), yL, c=l_cols, s=32, zorder=3)
            for i, feat in enumerate(l_feats):
                ax.text(
                    xL - 0.02,
                    yL[i],
                    f"{_strip_last_token(feat)} ({left_degree.get(i, 0)})",
                    ha="right",
                    va="center",
                    fontsize=label_fontsize,
                )

        if len(r_feats) > 0:
            ax.scatter(np.full(len(r_feats), xR), yR, c=r_cols, s=32, zorder=3)
            for j, feat in enumerate(r_feats):
                ax.text(
                    xR + 0.02,
                    yR[j],
                    f"{_strip_last_token(feat)} ({right_degree.get(j, 0)})",
                    ha="left",
                    va="center",
                    fontsize=label_fontsize,
                )

        if len(e_list) > 0:
            segs, cols, widths = [], [], []
            for i, j, c in e_list:
                segs.append([(xL, yL[i]), (xR, yR[j])])
                c0 = np.array(l_cols[i]) if not isinstance(l_cols[i], str) else np.array((0, 0, 0, 1))
                c1 = np.array(r_cols[j]) if not isinstance(r_cols[j], str) else np.array((0, 0, 0, 1))
                cols.append(0.5 * c0 + 0.5 * c1)
                widths.append(connection_width * (0.15 + 0.75 * abs(c)))

            lc = LineCollection(segs, colors=cols, linewidths=widths, alpha=0.38, zorder=1)
            ax.add_collection(lc)

        ax.set_xlim(-0.3, 1.3)
        ax.set_ylim(-0.05, 1.05)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    base_path = Path(save_path)
    full_path = base_path.with_name(f"{base_path.stem}_full{base_path.suffix}")
    connected_only_path = base_path.with_name(f"{base_path.stem}_connected_only{base_path.suffix}")

    _draw_bipartite_plot(left_features, right_features, left_colors, right_colors, edges, full_path)

    connected_left = sorted({i for i, _, _ in edges})
    connected_right = sorted({j for _, j, _ in edges})

    if connected_left and connected_right:
        l_map = {old_i: new_i for new_i, old_i in enumerate(connected_left)}
        r_map = {old_j: new_j for new_j, old_j in enumerate(connected_right)}

        left_feats_conn = [left_features[i] for i in connected_left]
        right_feats_conn = [right_features[j] for j in connected_right]
        left_cols_conn = [left_colors[i] for i in connected_left]
        right_cols_conn = [right_colors[j] for j in connected_right]

        edges_conn = [
            (l_map[i], r_map[j], c)
            for i, j, c in edges
            if i in l_map and j in r_map
        ]

        _draw_bipartite_plot(
            left_feats_conn,
            right_feats_conn,
            left_cols_conn,
            right_cols_conn,
            edges_conn,
            connected_only_path,
        )
    else:
        _draw_bipartite_plot([], [], [], [], [], connected_only_path)

    print(f"Saved full bipartite diagram to: {full_path}")
    print(f"Saved connected-only bipartite diagram to: {connected_only_path}")

    return str(full_path), str(connected_only_path)



# region Testing Space


# if __name__ == "__main__":
#     featCircularCorrelationDiagram(
#     pd.read_csv(paths["full_features"]["all"]["mordred"], index_col=0),
#     feature_set="mordred",
#     threshold=0.8,
#     max_labels=2000
# )

# featArcCorrelationDiagram(
#     pd.read_csv(paths["full_features"]["all"]["rdkit"], index_col=0),
#     threshold=0.8
# )


# left_df = pd.read_csv(paths["full_features"]["all"]["rdkit"], index_col=0)
# right_df = pd.read_csv(paths["full_features"]["all"]["mordred"], index_col=0)

# featBipartiteCorrelationDiagram(
#     df_left=left_df,
#     df_right=right_df,
#     left_feature_set="rdkit",
#     right_feature_set="mordred",
#     threshold=0.75,
#     max_labels=150,
#     connection_width=6,
#     save_path="/users/yhb18174/TL_project/results/rdkit_vs_mordred_corr.png",
# )
