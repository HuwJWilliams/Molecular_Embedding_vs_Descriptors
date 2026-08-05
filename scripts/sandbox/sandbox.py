# =============================================================================
# ChemBERTa Importance Analysis
# Goal: Identify which ChemBERTa embeddings are unused/redundant via SHAP/RF
# =============================================================================

# region Imports
from pathlib import Path
from glob import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from scipy.stats import gaussian_kde, pearsonr
from sklearn.metrics import root_mean_squared_error, r2_score

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/misc/")
from vis import Visualise
from misc_fns import molid2Smiles
from group_descriptors import *

v = Visualise(save_all=False)
SANDBOX = Path("/users/yhb18174/TL_project/scripts/sandbox")

paths = getPaths()

lipinski_ids = set(
    pd.read_csv(paths["full_features"]["fit_lipinski"]["rdkit"], index_col=0).index
)

# endregion


# region Function definitions
def calc_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.Descriptors.MolWt(mol)


def plot_true_vs_pred(
    true_test_data,
    pred_df,
    true_col: str,
    pred_col: str = "Prediction",
    save_plot: bool = False,
    save_path: str | Path = ".",
    save_fname: str = "true_vs_pred_scatter",
    dpi: int = 300,
    remove_pred_outliers: bool = False,
    remove_true_outliers: bool = False,
    lower_pct: float = 1,
    upper_pct: float = 99,
    show_plot: bool = False,
    model_name: str = None,
):
    common_idx = true_test_data.index.intersection(pred_df.index)

    y_true = true_test_data.loc[common_idx, true_col]
    y_pred = pred_df.loc[common_idx, pred_col]

    plot_df = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()

    if remove_pred_outliers:
        pred_low = plot_df["pred"].quantile(lower_pct / 100)
        pred_high = plot_df["pred"].quantile(upper_pct / 100)
        plot_df = plot_df[plot_df["pred"].between(pred_low, pred_high)]

    if remove_true_outliers:
        true_low = plot_df["true"].quantile(lower_pct / 100)
        true_high = plot_df["true"].quantile(upper_pct / 100)
        plot_df = plot_df[plot_df["true"].between(true_low, true_high)]

    y_true = plot_df["true"]
    y_pred = plot_df["pred"]

    residuals = y_pred - y_true
    bias = residuals.mean()
    sdep = np.sqrt(np.mean((residuals - bias) ** 2))
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pear_r, pear_p = pearsonr(y_true, y_pred)

    metrics = {
        "RMSE": float(rmse),
        "r2": float(r2),
        "Pearson_r": float(pear_r),
        "Pearson_p": float(pear_p),
        "Bias": float(bias),
        "SDEP": float(sdep),
        "n": int(len(plot_df)),
    }

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="black", linewidth=0.4)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="red",
        linewidth=1,
    )

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"True vs Predicted ({model_name})")

    table_data = [
        [
            f"{rmse:.3f}",
            f"{r2:.3f}",
            f"{pear_r:.3f}",
            f"{bias:.3f}",
            f"{sdep:.3f}",
        ]
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["RMSE", "R2", "Pearson r", "Bias", "SDEP"],
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.32, 1.0, 0.16],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    if save_plot:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{save_fname}.png", dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)

    return fig, ax, metrics


def plot_train_test_pred_distribution(
    train_data,
    true_test_data,
    pred_df,
    train_col: str,
    true_col: str,
    pred_col: str = "Prediction",
    bins: int = 40,
    alpha: float = 0.45,
    density: bool = False,
    label_sparse_bins: bool = False,
    sparse_bin_max_count: int = 5,
    save_plot: bool = False,
    save_path: str | Path = ".",
    save_fname: str = "train_test_pred_distribution",
    dpi: int = 300,
    show_plot: bool = False,
):
    common_idx = true_test_data.index.intersection(pred_df.index)

    train_vals = train_data[train_col].dropna()
    test_vals = true_test_data.loc[common_idx, true_col].dropna()
    pred_vals = pred_df.loc[common_idx, pred_col].dropna()

    all_vals = pd.concat([train_vals, test_vals, pred_vals])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    set_colors = {
        "train": "tab:blue",
        "test": "tab:orange",
        "pred": "tab:green",
    }

    ax.hist(
        train_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Train true (n={len(train_vals)})",
        edgecolor="black",
        color=set_colors["train"],
    )
    ax.hist(
        test_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Test true (n={len(test_vals)})",
        edgecolor="black",
        color=set_colors["test"],
    )
    ax.hist(
        pred_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Predicted (n={len(pred_vals)})",
        edgecolor="black",
        color=set_colors["pred"],
    )

    sparse_ids = {
        "train": {},
        "test": {},
        "pred": {},
    }

    if label_sparse_bins:
        sparse_sets = [
            (train_vals, "train", 0),
            (test_vals, "test", 1),
            (pred_vals, "pred", 2),
        ]

        y_max = ax.get_ylim()[1]
        y_step = y_max * 0.06

        for vals, label_prefix, stack_i in sparse_sets:
            counts, _ = np.histogram(vals, bins=bin_edges)

            for bin_i, count in enumerate(counts):
                if 0 < count <= sparse_bin_max_count:
                    left = bin_edges[bin_i]
                    right = bin_edges[bin_i + 1]

                    if bin_i == len(counts) - 1:
                        in_bin = vals[(vals >= left) & (vals <= right)]
                    else:
                        in_bin = vals[(vals >= left) & (vals < right)]

                    ids = in_bin.index.tolist()

                    sparse_ids[label_prefix][bin_i] = {
                        "left": float(left),
                        "right": float(right),
                        "count": int(count),
                        "ids": ids,
                    }

                    ids_text = ", ".join(map(str, ids))
                    bin_mid = (left + right) / 2

                    ax.text(
                        bin_mid,
                        count + (stack_i + 1) * y_step,
                        f"{label_prefix}: {ids_text}",
                        rotation=45,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        color=set_colors[label_prefix],
                    )

        ax.set_ylim(top=y_max * 1.35)

    ax.set_xlabel(true_col)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Train/Test/Prediction Distribution")
    ax.legend()

    plt.tight_layout()

    if save_plot:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{save_fname}.png", dpi=dpi, bbox_inches="tight")

    if show_plot:
        plt.show()

    plt.close(fig)

    return fig, ax, sparse_ids


def plot_two_df_ridgeplots(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    descriptors: list[str],
    label_a: str = "A",
    label_b: str = "B",
    save_plot: bool = False,
    save_path: str | Path = ".",
    save_prefix: str = "two_df_ridgeplot",
    dpi: int = 400,
    show_plot: bool = False,
    figsize: tuple = (10, 4),
    ridge_height: float = 0.9,
    alpha: float = 0.75,
):
    data_dict = {
        label_a: df_a,
        label_b: df_b,
    }

    save_path = Path(save_path)
    if save_plot:
        save_path.mkdir(parents=True, exist_ok=True)

    colors = {
        label_a: "tab:blue",
        label_b: "tab:orange",
    }

    figs = {}

    for desc in descriptors:
        desc_data = {}

        for label, df in data_dict.items():
            if desc not in df.columns:
                print(f"Skipping {label} for {desc}: descriptor not found.")
                continue

            vals = pd.to_numeric(df[desc], errors="coerce").dropna()

            if vals.empty:
                print(f"Skipping {label} for {desc}: no valid values.")
                continue

            desc_data[label] = vals

        if not desc_data:
            print(f"Skipping {desc}: no valid data.")
            continue

        all_vals = pd.concat(desc_data.values())
        x_min = all_vals.min()
        x_max = all_vals.max()

        if x_min == x_max:
            x_min -= 1
            x_max += 1

        x_vals = np.linspace(x_min, x_max, 500)

        fig, ax = plt.subplots(figsize=figsize)

        y_ticks = []
        y_labels = []

        for i, label in enumerate([label_a, label_b]):
            if label not in desc_data:
                continue

            vals = desc_data[label]
            values = vals.to_numpy()

            if np.unique(values).size == 1:
                val = values[0]
                width = max((x_max - x_min) * 0.02, 0.001)
                y_vals = np.exp(-0.5 * ((x_vals - val) / width) ** 2)
            else:
                kde = gaussian_kde(values)
                y_vals = kde(x_vals)

            y_scaled = y_vals / y_vals.max() * ridge_height
            baseline = i

            ax.fill_between(
                x_vals,
                baseline,
                baseline + y_scaled,
                color=colors[label],
                alpha=alpha,
                edgecolor="black",
                linewidth=0.5,
            )

            ax.plot(
                x_vals,
                baseline + y_scaled,
                color="black",
                linewidth=1,
            )

            y_ticks.append(baseline + ridge_height / 2)
            y_labels.append(f"{label} (n={len(values)})")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel(desc)
        ax.set_ylabel("Dataset")
        ax.set_title(f"{desc} Distribution")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save_plot:
            clean_desc = desc.replace("/", "_").replace(" ", "_")
            fig.savefig(
                save_path / f"{save_prefix}_{clean_desc}.png",
                dpi=dpi,
                bbox_inches="tight",
            )

        if show_plot:
            plt.show()

        plt.close(fig)
        figs[desc] = fig

    return figs


# endregion

# region

run_1 = False

if run_1:
    for prop, col in [
        # ("bp", "Boiling_Point"),
        # ("pka", "pKa"),
        # ("pka_paper1_basic", "pKa"),
        ("pka_paper1_acidic", "pKa"),
        # ("log_ld50", "LOG_LD50"),
        # ("pic50", "pIC50")
    ]:

        results = paths["prediction_output_dirs"]["rf"][prop]
        save_path = paths["prediction_output_dirs"]["rf"][prop]["rdkit"].parent
        json_paths = {}
        trimmed_perf = {}

        for k, p in results.items():
            try:
                if "molformer-c3-1b" in str(k) or "molformer-c3-1b" in str(p):
                    continue
                json_paths[k] = str(p / "last_20_pct_perf.json")

                pred_path = str(p / "last_20pct_pred.csv.gz")
                pred_df = pd.read_csv(pred_path, index_col=0)
                # pred_df = pred_df[pred_df.index.isin(lipinski_ids)]

                true_path = paths["targets"][prop]
                true_df = pd.read_csv(true_path, index_col=0)
                true_test_data = true_df.loc[pred_df.index]

                _, _, metrics = plot_true_vs_pred(
                    true_test_data=true_test_data,
                    pred_df=pred_df,
                    true_col=col,
                    pred_col=col,
                    save_path=save_path,
                    # remove_true_outliers=True,
                    # remove_pred_outliers=True,
                    # save_fname=f"true_vs_pred_scatter_{k}_99_true_trim",
                    # save_fname=f"true_vs_pred_scatter_{k}_99_pred_trim",
                    save_fname=f"true_vs_pred_scatter_{k}",
                    save_plot=True,
                    model_name=k,
                    upper_pct=95,
                    lower_pct=1,
                )
                trimmed_perf[k] = metrics
            except Exception as e:
                print(e)

        v.plotModelPerformanceBars(
            model_jsons=trimmed_perf,
            model_labels=list(trimmed_perf.keys()),
            metrics=["r2", "Pearson_r", "RMSE", "Bias", "SDEP"],
            show_plots=False,
            save_plot=True,
            save_path=save_path,
            # save_fname="model_performance_99_true_trim",
            # save_fname=f"model_performance_99_pred_trim",
            save_fname="model_performance",
        )


run_2 = False

if run_2:
    prop = "bp"
    col = "Boiling_Point"

    res_dir = Path(
        f"/users/yhb18174/TL_project/results/{prop.upper()}_predictions_rf/rdkit"
    )
    pred_df = pd.read_csv(res_dir / "last_20pct_pred.csv.gz", index_col=0)
    train_df = pd.read_csv(
        res_dir / "training_data" / "training_targets.csv.gz", index_col=0
    )
    true_path = paths["targets"][prop]
    true_df = pd.read_csv(true_path, index_col=0).loc[pred_df.index]

    _, _, sparse_ids = plot_train_test_pred_distribution(
        train_data=train_df,
        true_test_data=true_df,
        pred_df=pred_df,
        train_col=col,
        true_col=col,
        pred_col=col,
        save_plot=False,
        save_path="./",
        save_fname=f"{prop}_train_test_pred_distribution",
        show_plot=True,
        label_sparse_bins=True,
    )

    sparse_molids = []
    for set_bins in sparse_ids.values():
        for bin_info in set_bins.values():
            sparse_molids.extend(bin_info["ids"])

    sparse_molids = list(dict.fromkeys(sparse_molids))

    mols = []
    legends = []
    missing_ids = []
    for molid in sparse_molids:
        try:
            smi = molid2Smiles(molid)
        except IndexError:
            missing_ids.append(molid)
            continue

        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            legends.append(str(molid))

    img = Chem.Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=4,
        subImgSize=(250, 250),
    )

    if missing_ids:
        print(f"Missing SMILES for IDs: {missing_ids}")

    img.show()

run_3 = False

if run_3:
    desc = "mordred"
    res_dir = Path(f"/users/yhb18174/TL_project/results/BP_predictions_rf/{desc}")
    pred_ids = list(pd.read_csv(res_dir / "last_20pct_pred.csv.gz", index_col=0).index)
    train_df = pd.read_csv(
        res_dir / "training_data" / "training_features.csv.gz", index_col=0
    )
    true_path = true_path = str(paths["full_features"]["fit_lipinski"][desc])
    test_df = pd.read_csv(true_path, index_col=0).loc[pred_ids]

    figs = plot_two_df_ridgeplots(
        df_a=train_df,
        df_b=test_df,
        # descriptors=["MolWt_rdkit", "NumAromaticRings_rdkit", "MolLogP_rdkit", "NumHDonors_rdkit", "NumHAcceptors_rdkit"],
        descriptors=[
            "BertzCT_mordred",
            "naRing_mordred",
            "SLogP_mordred",
            "nHBDon_mordred",
            "nHBAcc_mordred",
            "MW_mordred",
            "C1SP3_mordred",
            "C2SP3_mordred",
        ],
        save_plot=True,
        save_path="/users/yhb18174/TL_project/scripts/sandbox",
        show_plot=True,
    )


run_4 = False

if run_4:
    prop = "bp"
    col = "Boiling_Point"

    res_dir = Path(
        f"/users/yhb18174/TL_project/results/{prop.upper()}_predictions_rf/rdkit"
    )

    pred_df = pd.read_csv(res_dir / "last_20pct_pred.csv.gz", index_col=0)
    train_df = pd.read_csv(
        res_dir / "training_data" / "training_targets.csv.gz", index_col=0
    )

    true_path = paths["targets"][prop]
    all_true_df = pd.read_csv(true_path, index_col=0)

    common_ids = all_true_df.index.intersection(pred_df.index)
    true_df = all_true_df.loc[common_ids].copy()
    pred_df = pred_df.loc[common_ids].copy()

    pred_col = col if col in pred_df.columns else "Prediction"

    error_df = pd.DataFrame(index=common_ids)
    error_df["true"] = true_df[col]
    error_df["pred"] = pred_df[pred_col]
    error_df["error"] = error_df["pred"] - error_df["true"]
    error_df["abs_error"] = error_df["error"].abs()
    error_df["SMILES"] = true_df["SMILES"]

    train_ids = train_df.index.intersection(all_true_df.index)
    train_smiles = all_true_df.loc[train_ids, "SMILES"].dropna()

    morgan_gen = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    def smi_to_fp(smi):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return morgan_gen.GetFingerprint(mol)

    train_fp_items = []
    for molid, smi in train_smiles.items():
        fp = smi_to_fp(smi)
        if fp is not None:
            train_fp_items.append((molid, fp))

    nn_sims = []
    nn_ids = []

    train_ids_only = [molid for molid, fp in train_fp_items]
    train_fps = [fp for molid, fp in train_fp_items]

    for molid, smi in error_df["SMILES"].items():
        fp = smi_to_fp(smi)

        if fp is None or not train_fps:
            nn_sims.append(np.nan)
            nn_ids.append(np.nan)
            continue

        sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        best_i = int(np.argmax(sims))

        nn_sims.append(float(sims[best_i]))
        nn_ids.append(train_ids_only[best_i])

    error_df["nearest_train_tanimoto"] = nn_sims
    error_df["nearest_train_id"] = nn_ids
    error_df = error_df.dropna(subset=["nearest_train_tanimoto", "abs_error"])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        error_df["nearest_train_tanimoto"],
        error_df["abs_error"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.4,
    )

    ax.set_xlabel("Nearest-neighbour Tanimoto similarity to training set")
    ax.set_ylabel(f"Absolute error in {col}")
    ax.set_title(f"{prop.upper()} error vs nearest training-set similarity")

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    min_sim = error_df["nearest_train_tanimoto"].min()
    max_sim = error_df["nearest_train_tanimoto"].max()

    ax.text(
        0.05,
        0.95,
        f"Min similarity: {min_sim:.3f}\nMax similarity: {max_sim:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig("/users/yhb18174/TL_project/scripts/sandbox/tanimoto_vs_error.png")
    plt.close(fig)

    perfect_match_df = error_df[
        np.isclose(error_df["nearest_train_tanimoto"], 1.0)
    ].copy()

    perfect_match_df["test_id"] = perfect_match_df.index
    perfect_match_df["train_id"] = perfect_match_df["nearest_train_id"]
    perfect_match_df["SMILES_1"] = perfect_match_df["SMILES"]
    perfect_match_df["SMILES_2"] = perfect_match_df["nearest_train_id"].map(
        all_true_df["SMILES"]
    )

    perfect_match_df = perfect_match_df[
        [
            "test_id",
            "train_id",
            "SMILES_1",
            "SMILES_2",
            "abs_error",
        ]
    ]
    perfect_match_df.to_csv(
        "/users/yhb18174/TL_project/scripts/sandbox/tanimoto_vs_error.csv"
    )


run_5 = False

if run_5:
    prop = "bp"
    col = "Boiling_Point"

    res_dir = Path(
        f"/users/yhb18174/TL_project/results/{prop.upper()}_predictions_rf/rdkit"
    )

    pred_df = pd.read_csv(res_dir / "last_20pct_pred.csv.gz", index_col=0)
    train_df = pd.read_csv(
        res_dir / "training_data" / "training_targets.csv.gz", index_col=0
    )

    true_path = paths["targets"][prop]
    all_true_df = pd.read_csv(true_path, index_col=0)

    common_ids = all_true_df.index.intersection(pred_df.index)
    true_df = all_true_df.loc[common_ids].copy()
    pred_df = pred_df.loc[common_ids].copy()

    train_ids = train_df.index.intersection(all_true_df.index)
    train_true_df = all_true_df.loc[train_ids].copy()

    test_smiles = true_df["SMILES"].astype(str)
    train_smiles = train_true_df["SMILES"].astype(str)

    exact_match_df = (
        test_smiles.reset_index()
        .rename(columns={"ID": "test_id", "SMILES": "SMILES"})
        .merge(
            train_smiles.reset_index().rename(
                columns={"ID": "train_id", "SMILES": "SMILES"}
            ),
            on="SMILES",
            how="inner",
        )
    )

    print(f"Exact SMILES matches: {len(exact_match_df)}")
    print(exact_match_df.head())


run_6 = False
if run_6:
    rdkit_df_p = "/users/yhb18174/TL_project/datasets/descriptors/BP_descriptors/bp_rdkit_1.csv"  # paths["full_features"]["fit_lipinski"]["rdkit"]
    # "/users/yhb18174/TL_project/datasets/all/all_mordred_with_nans.csv" #paths["full_features"]["all"]["mordred"]

    rdkit_df = pd.read_csv(rdkit_df_p, index_col="ID")
    # print(rdkit_df["NumAromaticRings_rdkit"].max())
    print(len(rdkit_df.columns))


run_7 = False
if run_7:
    import selfies
    from transformers import AutoTokenizer

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/config/")
    from pipeline_config import TRANSFORMER_FEATURE_SPECS

    p = "/users/yhb18174/TL_project/datasets/reorganisation_energies/DS3_8000.csv"
    df = pd.read_csv(p)
    df = df.dropna(subset=["SMILES"]).copy()
    df["SMILES"] = df["SMILES"].astype(str)

    def smiles_to_selfies(smi):
        try:
            return selfies.encoder(smi)
        except selfies.EncoderError:
            return None

    for feature_set, spec in TRANSFORMER_FEATURE_SPECS.items():
        if spec["input_kind"] == "smiles":
            input_texts = df["SMILES"]
            stats_df = df
        elif spec["input_kind"] == "selfies":
            selfies_df = df.copy()
            selfies_df["SELFIES"] = selfies_df["SMILES"].apply(smiles_to_selfies)
            invalid_selfies = selfies_df["SELFIES"].isna().sum()
            selfies_df = selfies_df.dropna(subset=["SELFIES"]).copy()
            input_texts = selfies_df["SELFIES"]
            stats_df = selfies_df
        else:
            continue

        tokenizer = AutoTokenizer.from_pretrained(
            spec["tokeniser"], trust_remote_code=True
        )
        token_lengths = input_texts.apply(
            lambda text: len(
                tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            )
        )
        max_idx = token_lengths.idxmax()
        p025, p975 = token_lengths.quantile([0.025, 0.975])

        print(f"\n{feature_set}")
        print(f"Tokenizer: {spec['tokeniser']}")
        if spec["input_kind"] == "selfies":
            print(f"Invalid SELFIES conversions skipped: {invalid_selfies}")
        print(f"Min token length: {token_lengths.min()}")
        print(f"Average token length: {token_lengths.mean():.2f}")
        print(f"Max token length: {token_lengths.loc[max_idx]}")
        print(f"Central 95% token range: {p025:.0f}-{p975:.0f}")
        print(f"Row index: {max_idx}")
        if "ID" in stats_df.columns:
            print(f"ID: {stats_df.loc[max_idx, 'ID']}")
        print(f"SMILES: {stats_df.loc[max_idx, 'SMILES']}")
        if spec["input_kind"] == "selfies":
            print(f"SELFIES: {stats_df.loc[max_idx, 'SELFIES']}")


run_8 = False

if run_8:
    import pandas as pd

    import matplotlib.pyplot as plt

    rdkit = pd.read_csv(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_rdkit.csv",
        index_col="ID",
    )

    count = 0

    for col in rdkit.columns:
        n_unique = rdkit[col].nunique(dropna=True)

        if n_unique < 6 and n_unique > 2:
            print(col, n_unique)
            count += 1

    print(count)

    # print(np.min(rdkit["NumAromaticRings_rdkit"]))
    # print(np.max(rdkit["NumAromaticRings_rdkit"]))

    # x_col = "MolWt_rdkit"
    # y_col = "AvgIpc_rdkit"

    # plot_df = rdkit[[x_col, y_col]].dropna()

    # ipc_cutoff = plot_df[y_col].quantile(0.90)
    # plot_df = plot_df.loc[plot_df[y_col] <= ipc_cutoff]

    # plt.figure(figsize=(7, 5))
    # plt.scatter(plot_df[x_col], plot_df[y_col], s=8, alpha=0.35)
    # plt.xlabel("MolWt")
    # plt.ylabel("Ipc")
    # plt.title("MolWt vs AvgIpc")
    # plt.grid(alpha=0.25)
    # plt.tight_layout()
    # plt.savefig("/users/yhb18174/TL_project/datasets/all/descriptor_analysis/MolWt_vs_AvgIpc.png", dpi=300, bbox_inches="tight")


run_9 = False
if run_9:

    from glob import glob

    files = glob(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred*/pred*.csv"
    )

    for f in files:
        print(f"processing file: {f}")
        df = pd.read_csv(f, index_col=0)

        task_type = df["task_type"]

        mask = (
            task_type.isna()
            | task_type.astype(str).str.strip().eq("")
            | task_type.astype(str).str.strip().str.lower().eq("nan")
        )

        print("rows to update:", mask.sum())

        df.loc[mask, "task_type"] = "regression"
        df.index.name = "Feature"
        df.to_csv(f, index=True)


run_10 = False

if run_10:
    from chembl_webresource_client.new_client import new_client
    import pandas as pd

    activity = new_client.activity

    # records = activity.filter(
    #     target_chembl_id="CHEMBL203",
    #     standard_type=True"IC50",
    #     standard_value__isnull=False
    # ).only([
    #     "molecule_chembl_id",
    #     "canonical_smiles",
    #     "standard_type",
    #     "standard_relation",
    #     "standard_value",
    #     "standard_units",
    #     "pchembl_value",
    #     "assay_chembl_id",
    #     "target_chembl_id",
    #     "target_organism",
    #     "document_chembl_id"
    # ])

    df = pd.DataFrame.from_records(records)

    print(df.head())
    print(df.shape)

    df_ic50 = df[(df["standard_units"] == "nM") & (df["standard_value"].notna())].copy()

    df_ic50["standard_value"] = df_ic50["standard_value"].astype(float)

    print(df_ic50.head())
    print(df_ic50.shape)

    df_ic50.to_csv("/users/yhb18174/Downloads/CHEMBL203_IC50.csv", index=False)

    df = pd.read_csv("/users/yhb18174/Downloads/CHEMBL203_IC50.csv")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])

    df = df[
        (df["standard_relation"] == "=")
        & (df["standard_units"] == "nM")
        & (df["standard_type"] == "IC50")
    ].copy()

    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df[df["standard_value"] > 0]

    df["pIC50"] = 9 - np.log10(df["standard_value"])
    df.to_csv(
        "/users/yhb18174/Downloads/CHEMBL203_pIC50.csv", index_label="assay_chembl_id"
    )


run_11 = False
if run_11:

    import sys
    import pandas as pd
    import numpy as np

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/config")
    from pipeline_config import DEFAULT_TARGET_COLUMNS

    for prop, prop_path in paths["targets"].items():
        feats = paths["full_features"][prop]["rdkit"]
        feats = str(feats).replace("*", "1")
        rdkit = pd.read_csv(feats, index_col="ID")

        prop_col = DEFAULT_TARGET_COLUMNS[prop]

        prop_df = pd.read_csv(prop_path, index_col="ID")

        if prop_col not in prop_df.columns:
            print(f"{prop}: missing target column {prop_col}")
            continue

        joined = rdkit.join(prop_df[[prop_col]], how="inner")
        joined[prop_col] = pd.to_numeric(joiWNSA3ned[prop_col], errors="coerce")
        joined = joined.dropna(subset=[prop_col])

        results = []

        for feat in rdkit.columns:
            x = pd.to_numeric(joined[feat], errors="coerce")
            y = joined[prop_col]

            valid = x.notna() & y.notna()

            if valid.sum() < 10:
                continue

            if x[valid].nunique() < 2 or y[valid].nunique() < 2:
                continue

            r = x[valid].corr(y[valid], method="pearson")

            results.append(
                {
                    "property": prop,
                    "target_col": prop_col,
                    "descriptor": feat,
                    "pearson_r": r,
                    "abs_pearson_r": abs(r),
                    "n": int(valid.sum()),
                }
            )

        corr_df = pd.DataFrame(results).sort_values("abs_pearson_r", ascending=False)

        corr_df = (
            pd.DataFrame(results)
            .sort_values("abs_pearson_r", ascending=False)
            .query("abs_pearson_r >= 0.7")
        )

        print("\n" + "=" * 80)
        print(prop, prop_col)
        print(corr_df)

run_12 = False
if run_12:
    from pathlib import Path
    import pandas as pd

    root = Path("/users/yhb18174/TL_project/results")

    for f in root.rglob("all_feature_importance.csv"):
        df = pd.read_csv(f)

        if "Feature" not in df.columns:
            df = df.reset_index().rename(columns={"index": "Feature"})

        keep_cols = ["Feature"] + [
            col
            for col in df.columns
            if col.startswith("Importance_")
            and not col.endswith("_x")
            and not col.endswith("_y")
        ]

        cleaned = df[keep_cols].drop_duplicates(subset=["Feature"], keep="first")
        cleaned.to_csv(f, index=False)

        print(f"Cleaned {f}")

run_13 = False
if run_13:
    res_dir = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
    csv_paths = [
        "pred_rdkit_tr_maccs/pred_rdkit_tr_maccs.csv",
        "pred_rdkit_tr_morgan/pred_rdkit_tr_morgan.csv",
        "pred_rdkit_tr_chemberta/pred_rdkit_tr_chemberta.csv",
        "pred_rdkit_tr_chembertasey/pred_rdkit_tr_chembertasey.csv",
        "pred_rdkit_tr_molformer/pred_rdkit_tr_molformer.csv",
        "pred_rdkit_tr_molformer-c3-1b/pred_rdkit_tr_molformer-c3-1b.csv",
        "pred_rdkit_tr_selformer/pred_rdkit_tr_selformer.csv",
    ]

    for p in csv_paths:
        csv_path = res_dir + p
        df = pd.read_csv(csv_path, index_col=0)

        mask = df["task_type"] == "binary_classification"

        df.loc[mask, "Balanced_Accuracy"] = (
            pd.to_numeric(df.loc[mask, "Sensitivity"], errors="coerce")
            + pd.to_numeric(df.loc[mask, "Specificity"], errors="coerce")
        ) / 2

        df.to_csv(csv_path)

run_14 = False
if run_14:
    import re
    import sys
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    csv_path = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
        "pred_mordred_avg_transformers/avg_stats_across_embedding_models.csv"
    )

    # Change this to "Autocorrelation" or "MoRSE"
    desc_group = "MoRSE"

    save_dir = csv_path.parent

    r2_cutoff = 0.00
    avg_r2_line = 0.66

    df = pd.read_csv(csv_path, index_col=0)

    desc_features = getGroups("mordred")[desc_group]

    desc_df = df.loc[
        df.index.isin(desc_features) & (df["task_type"] == "regression")
    ].copy()

    desc_df["r2"] = pd.to_numeric(desc_df["r2"], errors="coerce")
    desc_df = desc_df.dropna(subset=["r2"])

    def clean_name(name):
        return str(name).replace("_mordred", "").replace("_rdkit", "")

    def get_lag(descriptor):
        """
        Extract the first number in the descriptor name.

        Examples:
            ATS1m   -> 1
            AATS2v  -> 2
            Mor01m  -> 1
            Mor32se -> 32
        """
        match = re.search(r"\d+", clean_name(descriptor))
        return int(match.group()) if match else None

    def get_family(descriptor):
        """
        Extract the descriptor family before the first number.

        Examples:
            ATS1m   -> ATS
            AATS2v  -> AATS
            MATS3se -> MATS
            GATS4p  -> GATS
            Mor01m  -> Mor
        """
        name = clean_name(descriptor)
        match = re.match(r"([A-Za-z]+)\d+", name)
        return match.group(1) if match else "Unknown"

    def get_property(descriptor):
        """
        Extract property suffix after the first number.

        Examples:
            Mor01m   -> m
            Mor02v   -> v
            Mor03se  -> se
            Mor04     -> unweighted
            ATS1m    -> m
            AATS2se  -> se
        """
        name = clean_name(descriptor)

        # Remove any separators just in case
        name = name.replace("-", "").replace("_", "")

        match = re.match(r"[A-Za-z]+\.?\d+([A-Za-z]*)$", name)

        if not match:
            print(f"Could not parse property from: {descriptor} -> {name}")
            return "Unknown"

        suffix = match.group(1)

        if suffix == "":
            return "unweighted"

        return suffix

    # ------------------------------------------------------------------
    # 1. Bar plot of low-r2 descriptors
    # ------------------------------------------------------------------
    low_r2_df = desc_df.loc[desc_df["r2"] < r2_cutoff].sort_values(
        "r2",
        ascending=True,
    )

    print(f"\n{desc_group} descriptors with r2 < {r2_cutoff}:")
    for descriptor, row in low_r2_df.iterrows():
        print(f"{descriptor}: {row['r2']:.4f}")

    labels = [clean_name(idx) for idx in low_r2_df.index]

    plt.figure(figsize=(max(10, 0.45 * len(low_r2_df)), 6))
    plt.bar(labels, low_r2_df["r2"], edgecolor="black")
    plt.axhline(r2_cutoff, color="red", linestyle="--", linewidth=1)
    plt.ylabel("r2")
    plt.xlabel(f"{desc_group} feature")
    plt.title(f"{desc_group} Features Below {r2_cutoff} r2")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.tight_layout()

    save_path = (
        save_dir / f"{desc_group}_below_{str(r2_cutoff).replace('.', 'p')}_r2.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")
    print(f"Features plotted: {len(low_r2_df)}")

    # ------------------------------------------------------------------
    # 2. Box plot by lag / signal index
    # ------------------------------------------------------------------
    lag_df = desc_df.copy()
    lag_df["lag"] = [get_lag(idx) for idx in lag_df.index]
    lag_df = lag_df.dropna(subset=["lag"])
    lag_df["lag"] = lag_df["lag"].astype(int)

    print(f"\n{desc_group} r2 by lag/index:")
    print(
        lag_df.groupby("lag")["r2"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=lag_df, x="lag", y="r2", color="steelblue")
    # plt.axhline(avg_r2_line, color="red", linestyle="--", linewidth=1)

    if desc_group == "MoRSE":
        plt.xlabel("MoRSE signal index")
        plt.title("MoRSE Descriptor r2 by Signal Index")
    else:
        plt.xlabel(f"{desc_group} lag / distance")
        plt.title(f"{desc_group} Descriptor r2 by Lag")

    plt.ylabel("r2")
    plt.tight_layout()

    save_path = save_dir / f"{desc_group}_r2_by_lag_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")

    # ------------------------------------------------------------------
    # 3. Box plot by descriptor family
    # ------------------------------------------------------------------
    family_df = desc_df.copy()
    family_df["family"] = [get_family(idx) for idx in family_df.index]

    print(f"\n{desc_group} r2 by descriptor family:")
    print(
        family_df.groupby("family")["r2"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values("median", ascending=False)
    )

    family_order = (
        family_df.groupby("family")["r2"].median().sort_values(ascending=False).index
    )

    plt.figure(figsize=(max(10, 0.7 * len(family_order)), 6))
    sns.boxplot(
        data=family_df,
        x="family",
        y="r2",
        order=family_order,
        color="steelblue",
    )
    # plt.axhline(avg_r2_line, color="red", linestyle="--", linewidth=1)
    plt.xlabel(f"{desc_group} descriptor family")
    plt.ylabel("r2")
    plt.title(f"{desc_group} Descriptor r2 by Family")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = save_dir / f"{desc_group}_r2_by_descriptor_family_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")

    # ------------------------------------------------------------------
    # 4. Scatter plots by property suffix, one plot per family
    #    colour = lag
    #    horizontal line = average r2 for that family
    # ------------------------------------------------------------------
    property_df = desc_df.copy()
    property_df["property"] = [get_property(idx) for idx in property_df.index]
    property_df["lag"] = [get_lag(idx) for idx in property_df.index]
    property_df["family"] = [get_family(idx) for idx in property_df.index]

    property_df = property_df.dropna(subset=["property", "r2", "lag", "family"])
    property_df["lag"] = property_df["lag"].astype(int)

    print(f"\n{desc_group} r2 by family and property suffix:")
    print(
        property_df.groupby(["family", "property"])["r2"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_values(["family", "median"], ascending=[True, False])
    )

    # Keep lag colours consistent across all family plots
    unique_lags = sorted(property_df["lag"].unique())
    palette = sns.color_palette("husl", n_colors=len(unique_lags))
    lag_to_color = {lag: palette[i] for i, lag in enumerate(unique_lags)}

    families = sorted(property_df["family"].unique())

    scatter_save_dir = save_dir / f"{desc_group}_family_property_suffix_scatter_plots"
    scatter_save_dir.mkdir(parents=True, exist_ok=True)

    saved_family_plots = []

    for family in families:
        family_df = property_df[property_df["family"] == family].copy()

        if family_df.empty:
            continue

        family_avg_r2 = family_df["r2"].mean()

        property_order = (
            family_df.groupby("property")["r2"]
            .median()
            .sort_values(ascending=False)
            .index
        )

        family_df["property"] = pd.Categorical(
            family_df["property"],
            categories=property_order,
            ordered=True,
        )

        family_df = family_df.sort_values(["property", "lag", "r2"])

        property_to_x = {prop: i for i, prop in enumerate(property_order)}

        family_df["x_pos"] = family_df["property"].map(property_to_x).astype(float)

        rng = np.random.default_rng(0)
        family_df["x_jitter"] = family_df["x_pos"] + rng.normal(
            loc=0,
            scale=0.08,
            size=len(family_df),
        )

        fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(property_order)), 6))

        for lag in unique_lags:
            lag_subset = family_df[family_df["lag"] == lag]

            if lag_subset.empty:
                continue

            ax.scatter(
                lag_subset["x_jitter"],
                lag_subset["r2"],
                s=55,
                color=lag_to_color[lag],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.9,
                label=f"Lag {lag}",
            )

        ax.axhline(
            family_avg_r2,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean r2 = {family_avg_r2:.3f}",
        )

        ax.set_xlabel("Property suffix")
        ax.set_ylabel("r2")
        ax.set_title(
            f"{desc_group} {family} Descriptors\n"
            f"r2 by Property Suffix, Coloured by Lag"
        )

        ax.set_xticks(range(len(property_order)))
        ax.set_xticklabels(property_order, rotation=45, ha="right")

        ax.legend(
            title="Lag / average",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )

        fig.tight_layout()

        save_path = (
            scatter_save_dir
            / f"{desc_group}_{family}_r2_by_property_suffix_scatter_by_lag.png"
        )

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_family_plots.append(save_path)
        print(f"Saved plot to: {save_path}")


run_15 = False

if run_15:
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    mordred_path = Path(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_mordred.csv"
    )
    save_dir = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_rdkit"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    keep_families = {"GATS", "ATSC", "AATSC", "MATS"}
    min_distance = 0

    df = pd.read_csv(mordred_path, index_col="ID")
    df = df.drop(columns=["SMILES"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")

    autocorr_features = [
        c for c in getGroups("mordred")["Autocorrelation"] if c in df.columns
    ]

    def clean_name(name):
        return str(name).replace("_mordred", "").replace("_rdkit", "")

    def parse_autocorr_descriptor(descriptor):
        name = clean_name(descriptor)

        # Examples:
        #   AATSC4Z  -> family=AATSC, distance=4, suffix=Z
        #   GATS8se  -> family=GATS, distance=8, suffix=se
        match = re.match(r"([A-Za-z]+)(\d+)(.+)$", name)

        if not match:
            return None, None, None

        family = match.group(1)
        distance = int(match.group(2))
        suffix = match.group(3)

        return family, distance, suffix

    selected = []

    for feat in autocorr_features:
        family, distance, suffix = parse_autocorr_descriptor(feat)

        if (
            family in keep_families
            and distance is not None
            and distance >= min_distance
        ):
            selected.append(
                {
                    "feature": feat,
                    "family": family,
                    "distance": distance,
                    "suffix": suffix,
                }
            )

    meta_df = pd.DataFrame(selected)

    print("\nSelected descriptor counts by property suffix:")
    print(
        meta_df.groupby(["family", "suffix"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["family", "suffix"])
    )

    grouped = {
        (family, suffix): group["feature"].tolist()
        for (family, suffix), group in meta_df.groupby(["family", "suffix"])
    }

    for (family, suffix), feats in sorted(grouped.items()):
        feats = sorted(feats, key=lambda x: parse_autocorr_descriptor(x)[1])

        family_df = df[feats].copy()

        # Z-score each descriptor so boxplots are comparable.
        family_df = (family_df - family_df.mean()) / family_df.std(ddof=0)
        family_df = family_df.replace([np.inf, -np.inf], np.nan)

        long_df = family_df.melt(
            var_name="descriptor",
            value_name="z_value",
        ).dropna()

        long_df["descriptor_label"] = long_df["descriptor"].map(clean_name)

        if long_df.empty:
            print(f"Skipping {family} suffix {suffix}: no finite values")
            continue

        descriptor_order = [clean_name(c) for c in feats]

        plt.figure(figsize=(max(10, 0.55 * len(descriptor_order)), 6))

        sns.boxplot(
            data=long_df,
            x="descriptor_label",
            y="z_value",
            order=descriptor_order,
            color="steelblue",
            showfliers=False,
        )

        plt.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        plt.xlabel("Descriptor")
        plt.ylabel("Z-scored descriptor value")
        plt.title(
            f"{family} Autocorrelation Distributions, "
            f"Suffix {suffix}, Distance > {min_distance}"
        )
        plt.xticks(rotation=60, ha="right", fontsize=8)
        plt.tight_layout()

        safe_suffix = str(suffix).replace("/", "_").replace("\\", "_")
        save_path = save_dir / (
            f"autocorrelation_{family}_suffix_{safe_suffix}"
            f"_distance_gt_{min_distance}_boxplot.png"
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {family} suffix {suffix} boxplot to: {save_path}")


run_16 = False

if run_16:
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    mordred_path = Path(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_mordred.csv"
    )
    save_dir = Path("/users/yhb18174/TL_project/datasets/all/descriptor_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / "autocorrelation_descriptor_correlation_heatmap.png"

    df = pd.read_csv(mordred_path, index_col="ID")
    df = df.drop(columns=["SMILES"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")

    autocorr_features = [
        c for c in getGroups("mordred")["Autocorrelation"] if c in df.columns
    ]

    autocorr_df = df[autocorr_features].copy()
    autocorr_df = autocorr_df.replace([np.inf, -np.inf], np.nan)

    # Drop descriptors with no variance or too many missing values.
    autocorr_df = autocorr_df.dropna(axis=1, how="all")
    autocorr_df = autocorr_df.loc[:, autocorr_df.nunique(dropna=True) > 1]

    corr = autocorr_df.corr(method="pearson")

    plt.figure(figsize=(18, 16))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson correlation"},
    )

    plt.title("Correlation Heatmap of Mordred Autocorrelation Descriptors")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Autocorrelation descriptors plotted: {corr.shape[0]}")
    print(f"Saved heatmap to: {save_path}")

# run_17 (= True)

# if run_17:
#     import re
#     import sys
#     from pathlib import Path

#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import seaborn as sns

#     sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
#     from group_descriptors import getGroups

#     mordred_path = Path(
#         "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_mordred.csv"
#     )
#     save_dir = Path(
#         "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_rdkit"
#     )
#     save_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(mordred_path, index_col="ID")
#     df = df.drop(columns=["SMILES"], errors="ignore")
#     df = df.apply(pd.to_numeric, errors="coerce")
#     df = df.replace([np.inf, -np.inf], np.nan)

#     autocorr_features = [
#         c for c in getGroups("mordred")["Autocorrelation"]
#         if c in df.columns
#     ]

#     def clean_name(name):
#         return str(name).replace("_mordred", "").replace("_rdkit", "")

#     def get_family(descriptor):
#         # Examples: MATS1c -> MATS, AATSC3v -> AATSC
#         match = re.match(r"([A-Za-z]+)\d+", clean_name(descriptor))
#         return match.group(1) if match else "Unknown"

#     records = []

#     for feat in autocorr_features:
#         family = get_family(feat)
#         values = df[feat].dropna()

#         if values.empty:
#             continue

#         q1 = values.quantile(0.25)
#         q3 = values.quantile(0.75)

#         records.extend([
#             {"family": family, "descriptor": feat, "stat": "min", "value": values.min()},
#             {"family": family, "descriptor": feat, "stat": "max", "value": values.max()},
#             {"family": family, "descriptor": feat, "stat": "avg", "value": values.mean()},
#             {"family": family, "descriptor": feat, "stat": "IQR", "value": q3 - q1},
#         ])

#     stat_df = pd.DataFrame(records)

#     print("\nAutocorrelation descriptor summary stats by family:")
#     print(
#         stat_df.groupby(["family", "stat"])["value"]
#         .agg(["count", "mean", "median", "min", "max"])
#         .round(4)
#     )

#     stat_order = ["min", "max", "avg", "IQR"]

#     plt.figure(figsize=(12, 6))
#     sns.stripplot(
#         data=stat_df,
#         x="family",
#         y="value",
#         hue="stat",
#         hue_order=stat_order,
#         dodge=True,
#         alpha=0.65,
#         size=4,
#     )

#     plt.xlabel("Autocorrelation descriptor family")
#     plt.ylabel("Statistic value")
#     plt.title("Min, Max, Average, and IQR per Autocorrelation Descriptor")
#     plt.xticks(rotation=45, ha="right")
#     plt.legend(title="Statistic", bbox_to_anchor=(1.02, 1), loc="upper left")
#     plt.tight_layout()

#     save_path = save_dir / "autocorrelation_family_descriptor_stats_scatter.png"
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"\nSaved plot to: {save_path}")

run_17 = False

if run_17:
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    mordred_path = Path(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_mordred.csv"
    )
    save_dir = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_rdkit"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(mordred_path, index_col="ID")
    df = df.drop(columns=["SMILES"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)

    autocorr_features = [
        c for c in getGroups("mordred")["Autocorrelation"] if c in df.columns
    ]

    def clean_name(name):
        return str(name).replace("_mordred", "").replace("_rdkit", "")

    def get_family(descriptor):
        match = re.match(r"([A-Za-z]+)\d+", clean_name(descriptor))
        return match.group(1) if match else "Unknown"

    def fmt_value(x):
        if pd.isna(x):
            return "nan"
        if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 0.001):
            return f"{x:.2e}"
        return f"{x:.3g}"

    def dominant_value_label(values):
        values = values.dropna()

        if values.empty:
            return "DV=nan, 0.0%"

        counts = values.value_counts(dropna=True)
        dominant_value = counts.index[0]
        dominant_count = counts.iloc[0]
        dominant_pct = 100 * dominant_count / len(values)

        return f"DV={fmt_value(dominant_value)}, {dominant_pct:.1f}%"

    records = []

    for feat in autocorr_features:
        family = get_family(feat)
        values = df[feat].dropna()

        if values.empty:
            continue

        descriptor_label = f"{clean_name(feat)}\n({dominant_value_label(values)})"

        records.extend(
            [
                {
                    "family": family,
                    "descriptor": feat,
                    "descriptor_label": descriptor_label,
                    "stat": "min",
                    "value": values.min(),
                },
                {
                    "family": family,
                    "descriptor": feat,
                    "descriptor_label": descriptor_label,
                    "stat": "max",
                    "value": values.max(),
                },
                {
                    "family": family,
                    "descriptor": feat,
                    "descriptor_label": descriptor_label,
                    "stat": "avg",
                    "value": values.mean(),
                },
            ]
        )

    stat_df = pd.DataFrame(records)

    stat_order = ["min", "max", "avg"]

    for family, family_df in stat_df.groupby("family"):
        descriptor_order = (
            family_df.loc[family_df["stat"] == "avg"]
            .sort_values("value", ascending=False)["descriptor_label"]
            .tolist()
        )

        plt.figure(figsize=(max(12, 0.55 * len(descriptor_order)), 7))

        sns.scatterplot(
            data=family_df,
            x="descriptor_label",
            y="value",
            hue="stat",
            hue_order=stat_order,
            style="stat",
            style_order=stat_order,
            s=35,
        )

        plt.xticks(
            ticks=range(len(descriptor_order)),
            labels=descriptor_order,
            rotation=75,
            ha="right",
            fontsize=7,
        )

        plt.xlabel("Descriptor")
        plt.ylabel("Statistic value")
        plt.title(f"{family} Descriptor Summary Statistics")
        plt.legend(title="Statistic", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        save_path = save_dir / f"autocorrelation_{family}_descriptor_stats_scatter.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {family} plot to: {save_path}")

run_19 = False

if run_19:
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
    from group_descriptors import getGroups

    rdkit_path = Path(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_rdkit.csv"
    )
    mordred_path = Path(
        "/users/yhb18174/TL_project/datasets/all/fit_lipinski/all_mordred.csv"
    )
    save_dir = Path("/users/yhb18174/TL_project/datasets/all/descriptor_analysis")
    save_dir.mkdir(parents=True, exist_ok=True)

    rdkit_df = pd.read_csv(rdkit_path)
    mordred_df = pd.read_csv(mordred_path)

    # Descriptor columns
    rdkit_cols = [c for c in rdkit_df.columns if str(c).endswith("_rdkit")]
    mordred_ats_cols = [
        c for c in mordred_df.columns if str(c).endswith("_mordred") and "ATS" in str(c)
    ]

    print(f"Found {len(rdkit_cols)} RDKit descriptor columns.")
    print(f"Found {len(mordred_ats_cols)} Mordred ATS descriptor columns.")

    rdkit_data = rdkit_df[rdkit_cols].apply(pd.to_numeric, errors="coerce")
    mordred_data = mordred_df[mordred_ats_cols].apply(pd.to_numeric, errors="coerce")

    # Drop unusable columns
    rdkit_data = rdkit_data.dropna(axis=1, how="all")
    mordred_data = mordred_data.dropna(axis=1, how="all")

    rdkit_data = rdkit_data.loc[:, rdkit_data.nunique(dropna=True) > 1]
    mordred_data = mordred_data.loc[:, mordred_data.nunique(dropna=True) > 1]

    print(f"Using {rdkit_data.shape[1]} non-constant RDKit descriptors.")
    print(f"Using {mordred_data.shape[1]} non-constant Mordred ATS descriptors.")

    if rdkit_data.empty:
        print("No valid RDKit descriptor columns found.")
    elif mordred_data.empty:
        print("No valid Mordred ATS descriptor columns found.")
    else:
        # Make sure rows align.
        # This assumes both CSVs are in the same molecule/order.
        n = min(len(rdkit_data), len(mordred_data))
        rdkit_data = rdkit_data.iloc[:n].reset_index(drop=True)
        mordred_data = mordred_data.iloc[:n].reset_index(drop=True)

        combined = pd.concat([rdkit_data, mordred_data], axis=1)

        corr_matrix = combined.corr(method="spearman")

        # Rows = RDKit descriptors, columns = Mordred ATS descriptors
        heatmap_matrix = corr_matrix.loc[
            rdkit_data.columns,
            mordred_data.columns,
        ]

        matrix_path = save_dir / "rdkit_vs_mordred_ATS_spearman_matrix.csv"
        heatmap_matrix.to_csv(matrix_path)

        print(f"Saved Spearman matrix to: {matrix_path}")

        # Optional cleaner labels for plotting only
        y_labels = [re.sub(r"_rdkit$", "", str(c)) for c in heatmap_matrix.index]
        x_labels = [re.sub(r"_mordred$", "", str(c)) for c in heatmap_matrix.columns]

        fig_width = max(10, 0.35 * len(heatmap_matrix.columns))
        fig_height = max(8, 0.28 * len(heatmap_matrix.index))

        # Cap figure size to avoid huge/corrupt images
        fig_width = min(fig_width, 50)
        fig_height = min(fig_height, 50)

        plt.figure(figsize=(fig_width, fig_height))

        im = plt.imshow(
            heatmap_matrix.values,
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )

        plt.colorbar(im, label="Spearman correlation")

        plt.xticks(
            ticks=np.arange(len(x_labels)),
            labels=x_labels,
            rotation=90,
            fontsize=6,
        )

        plt.yticks(
            ticks=np.arange(len(y_labels)),
            labels=y_labels,
            fontsize=6,
        )

        plt.xlabel("Mordred ATS descriptors")
        plt.ylabel("RDKit descriptors")
        plt.title("Spearman correlation: RDKit descriptors vs Mordred ATS descriptors")

        plt.tight_layout()

        heatmap_path = save_dir / "rdkit_vs_mordred_ATS_spearman_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved heatmap to: {heatmap_path}")

run_20 = False
if run_20:
    path = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv"
    df = pd.read_csv(path)

    task_type_means = (
        df.groupby("task_type", dropna=False).mean(numeric_only=True).reset_index()
    )

    print(task_type_means.to_string(index=False))


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

run_21 = False

if run_21:
    path = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
    )

    rdkit_df = pd.read_csv(
        path / "pred_mordred_tr_rdkit/regression_group_perf.csv", index_col=0
    )
    maccs_df = pd.read_csv(
        path / "pred_mordred_tr_maccs/regression_group_perf.csv", index_col=0
    )
    morgan_df = pd.read_csv(
        path / "pred_mordred_tr_morgan/regression_group_perf.csv", index_col=0
    )

    # Make sure all three dfs are aligned by the same index
    common_index = rdkit_df.index.intersection(maccs_df.index).intersection(
        morgan_df.index
    )

    rdkit_df = rdkit_df.loc[common_index]
    maccs_df = maccs_df.loc[common_index]
    morgan_df = morgan_df.loc[common_index]

    rdkit_df = rdkit_df.drop(columns=["avg_Bias", "avg_RMSE", "avg_Pearson_r"])
    maccs_df = maccs_df.drop(columns=["avg_Bias", "avg_RMSE", "avg_Pearson_r"])
    morgan_df = morgan_df.drop(columns=["avg_Bias", "avg_RMSE", "avg_Pearson_r"])

    # Numeric metric columns only
    metric_cols = rdkit_df.select_dtypes(include="number").columns

    rdkit_minus_maccs = rdkit_df[metric_cols].subtract(maccs_df[metric_cols])
    rdkit_minus_morgan = rdkit_df[metric_cols].subtract(morgan_df[metric_cols])

    # Give the index a name for plotting
    group_col = "group"
    rdkit_minus_maccs.index.name = group_col
    rdkit_minus_morgan.index.name = group_col

    rdkit_minus_maccs_long = rdkit_minus_maccs.reset_index().melt(
        id_vars=group_col, var_name="metric", value_name="normalised_difference"
    )
    rdkit_minus_maccs_long["comparison"] = "RDKit - MACCS"

    rdkit_minus_morgan_long = rdkit_minus_morgan.reset_index().melt(
        id_vars=group_col, var_name="metric", value_name="normalised_difference"
    )
    rdkit_minus_morgan_long["comparison"] = "RDKit - Morgan"

    plot_df = pd.concat(
        [rdkit_minus_maccs_long, rdkit_minus_morgan_long], ignore_index=True
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=plot_df,
        x=group_col,
        y="normalised_difference",
        hue="comparison",
        style="metric",
        markers=True,
        dashes=False,
    )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel(group_col)
    plt.ylabel("RDKit-normalised difference")
    plt.title("Normalised group performance differences")
    plt.xticks(rotation=45, ha="right")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        "/users/yhb18174/TL_project/scripts/sandbox/norm_fing.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

run_21 = False
if run_21:
    path = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_*/*_group_perf.csv"
    files = glob(path)

    for f in files:
        df = pd.read_csv(f, index_col=0, low_memory=False)

        averaged_cols = df.select_dtypes(include="number").mean().to_dict()

        print(f)
        print(averaged_cols)


run_22 = False
if run_22:
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    mordred_df = pd.read_csv(
        "/users/yhb18174/TL_project/datasets/all/all_mordred.csv",
        index_col=0,
        low_memory=False,
    )

    save_path = Path("/users/yhb18174/TL_project/datasets/all/descriptor_analysis/")
    save_path.mkdir(parents=True, exist_ok=True)

    columns = [
        "nAcid_mordred",
        "nBase_mordred",
        "nRot_mordred",
        "RotRatio_mordred",
        "GeomPetitjeanIndex_mordred",
        "GeomShapeIndex_mordred",
        "PBF_mordred",
    ]

    summary_rows = []

    for column in columns:
        if column not in mordred_df.columns:
            print(f"Skipping {column}: column not found")
            continue

        s = pd.to_numeric(mordred_df[column], errors="coerce").dropna()

        if s.empty:
            print(f"Skipping {column}: no numeric values")
            continue

        min_val = s.min()
        max_val = s.max()
        avg_val = s.mean()
        med_val = s.median()
        q10 = s.quantile(0.10)
        q90 = s.quantile(0.90)
        percent_at_median = (s == med_val).mean() * 100

        summary_rows.append(
            {
                "column": column,
                "min": min_val,
                "max": max_val,
                "average": avg_val,
                "median": med_val,
                "q10": q10,
                "q90": q90,
                "percent_at_median": percent_at_median,
            }
        )

        print(f"\n{column}")
        print(f"min: {min_val}")
        print(f"max: {max_val}")
        print(f"average: {avg_val}")
        print(f"median: {med_val}")
        print(f"10th percentile: {q10}")
        print(f"90th percentile: {q90}")
        print(f"% at median: {percent_at_median:.2f}%")

        vals = s.values.astype(float)

        fig, ax = plt.subplots(figsize=(7, 3))

        # Shade central 80% of the data
        ax.axvspan(q10, q90, alpha=0.2, label=f"Central 80% = [{q10:.2f}, {q90:.2f}]")

        # KDE needs at least 2 unique points
        if len(np.unique(vals)) > 1:
            kde = gaussian_kde(vals)
            x = np.linspace(min_val - 0.5, max_val + 0.5, 500)
            y = kde(x)

            ax.fill_between(x, y, alpha=0.7)
            ax.plot(x, y, linewidth=1.5)
        else:
            # fallback if all values are identical
            x = np.array([vals[0] - 0.5, vals[0], vals[0] + 0.5])
            y = np.array([0, 1, 0])
            ax.fill_between(x, y, alpha=0.7)
            ax.plot(x, y, linewidth=1.5)

        ax.axvline(avg_val, linestyle="--", linewidth=1, label=f"Mean = {avg_val:.2f}")
        ax.axvline(med_val, linestyle=":", linewidth=1, label=f"Median = {med_val:.2f}")

        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Density")
        ax.set_yticks([])

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        ax.legend()
        fig.tight_layout()

        out_file = save_path / f"{column}_ridgeplot_kde.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        save_path / "selected_descriptor_distribution_summary.csv",
        index=False,
    )


run_23 = False

if run_23:
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter
    from pathlib import Path
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    def atom_label(atom):
        """
        Return atom symbol.
        Uses lowercase for aromatic atoms, e.g. aromatic carbon = c.
        """
        symbol = atom.GetSymbol()

        if atom.GetIsAromatic():
            return symbol.lower()

        return symbol

    def get_rotatable_bond_types_from_mol(mol):
        """
        Return bond-type labels for all rotatable bonds in one molecule.
        """
        if mol is None:
            return []

        bond_types = []

        for atom_idx_1, atom_idx_2 in mol.GetSubstructMatches(
            Lipinski.RotatableBondSmarts
        ):
            atom_1 = mol.GetAtomWithIdx(atom_idx_1)
            atom_2 = mol.GetAtomWithIdx(atom_idx_2)

            label_1 = atom_label(atom_1)
            label_2 = atom_label(atom_2)

            # Sort so C-N and N-C are counted as the same type
            bond_type = "-".join(sorted([label_1, label_2]))

            bond_types.append(bond_type)

        return bond_types

    def plot_top_rotatable_bond_types(
        df,
        smiles_col="SMILES",
        top_n=5,
        save_path=None,
    ):
        """
        Count rotatable bond atom-pair types across a dataframe of SMILES strings,
        plot top N as a pie chart, and group all remaining types as Other.
        """

        all_bond_types = []
        invalid_smiles = 0
        molecules_with_no_rotatable_bonds = 0

        for smi in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                invalid_smiles += 1
                continue

            bond_types = get_rotatable_bond_types_from_mol(mol)

            if len(bond_types) == 0:
                molecules_with_no_rotatable_bonds += 1

            all_bond_types.extend(bond_types)

        counts = Counter(all_bond_types)

        if not counts:
            raise ValueError("No rotatable bonds found in the dataframe.")

        top_counts = counts.most_common(top_n)

        top_labels = [label for label, count in top_counts]
        top_values = [count for label, count in top_counts]

        other_count = sum(
            count for label, count in counts.items() if label not in top_labels
        )

        plot_labels = top_labels.copy()
        plot_values = top_values.copy()

        if other_count > 0:
            plot_labels.append("Other")
            plot_values.append(other_count)

        summary_df = pd.DataFrame(
            {
                "rotatable_bond_type": plot_labels,
                "count": plot_values,
            }
        )

        summary_df["percentage"] = summary_df["count"] / summary_df["count"].sum() * 100

        print("\nTop rotatable bond types:")
        print(summary_df)

        print(f"\nInvalid SMILES skipped: {invalid_smiles}")
        print(
            "Molecules with no rotatable bonds: " f"{molecules_with_no_rotatable_bonds}"
        )
        print(f"Total rotatable bonds counted: {sum(plot_values)}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 7))
        plt.pie(
            summary_df["count"],
            labels=summary_df["rotatable_bond_type"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title(f"Top {top_n} Rotatable Bond Types Across Dataset")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nSaved pie chart to: {save_path}")

            summary_path = save_path.with_suffix(".csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary CSV to: {summary_path}")

        plt.show()

        return summary_df, counts

    df = pd.read_csv("/users/yhb18174/TL_project/datasets/all/all_rdkit.csv")
    summary_df, all_counts = plot_top_rotatable_bond_types(
        df,
        smiles_col="SMILES",
        top_n=5,
        save_path="/users/yhb18174/TL_project/scripts/sandbox/rot_bond.png",
    )

run_24 = False

if run_24:
    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import Counter
    from pathlib import Path

    from rdkit import Chem
    from rdkit.Chem import Lipinski
    from rdkit.Chem.rdchem import HybridizationType

    def carbon_hybrid_class(atom):
        """
        Return carbon hybridisation class for carbon atoms.
        Non-carbon atoms return None.
        """
        if atom.GetAtomicNum() != 6:
            return None

        hyb = atom.GetHybridization()

        if hyb == HybridizationType.SP:
            return "Csp1"
        elif hyb == HybridizationType.SP2:
            return "Csp2"
        elif hyb == HybridizationType.SP3:
            return "Csp3"
        else:
            return "C_other"

    def classify_rotatable_bond_by_carbon_hybridisation(mol):
        """
        Classify each RDKit-defined rotatable bond by whether it involves
        sp3 carbon, sp2 carbon, both, or neither.
        """
        if mol is None:
            return []

        classes = []

        for atom_idx_1, atom_idx_2 in mol.GetSubstructMatches(
            Lipinski.RotatableBondSmarts
        ):
            atom_1 = mol.GetAtomWithIdx(atom_idx_1)
            atom_2 = mol.GetAtomWithIdx(atom_idx_2)

            hybrids = {
                carbon_hybrid_class(atom_1),
                carbon_hybrid_class(atom_2),
            }

            hybrids.discard(None)

            has_sp2 = "Csp2" in hybrids
            has_sp3 = "Csp3" in hybrids
            has_sp1 = "Csp1" in hybrids

            if has_sp2 and has_sp3:
                classes.append("Csp2-Csp3")
            elif has_sp3:
                classes.append("Csp3-involving")
            elif has_sp2:
                classes.append("Csp2-involving")
            elif has_sp1:
                classes.append("Csp1-involving")
            else:
                classes.append("No Csp1/Csp2/Csp3")

        return classes

    def plot_rotatable_bond_carbon_hybridisation_ratios(
        df,
        smiles_col="SMILES",
        save_path=None,
    ):
        """
        Plot dataset-level ratios of rotatable bonds involving Csp3, Csp2,
        both Csp2-Csp3, Csp1, or none.
        """

        all_classes = []
        invalid_smiles = 0
        molecules_with_no_rotatable_bonds = 0

        for smi in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                invalid_smiles += 1
                continue

            classes = classify_rotatable_bond_by_carbon_hybridisation(mol)

            if len(classes) == 0:
                molecules_with_no_rotatable_bonds += 1

            all_classes.extend(classes)

        counts = Counter(all_classes)

        if not counts:
            raise ValueError("No rotatable bonds found in the dataframe.")

        summary_df = pd.DataFrame(
            {
                "rotatable_bond_class": list(counts.keys()),
                "count": list(counts.values()),
            }
        )

        summary_df["percentage"] = summary_df["count"] / summary_df["count"].sum() * 100

        order = [
            "Csp3-involving",
            "Csp2-involving",
            "Csp2-Csp3",
            "Csp1-involving",
            "No Csp1/Csp2/Csp3",
        ]

        summary_df["order"] = summary_df["rotatable_bond_class"].apply(
            lambda x: order.index(x) if x in order else len(order)
        )
        summary_df = summary_df.sort_values("order").drop(columns="order")

        print("\nRotatable bond carbon-hybridisation ratios:")
        print(summary_df)

        print(f"\nInvalid SMILES skipped: {invalid_smiles}")
        print(f"Molecules with no rotatable bonds: {molecules_with_no_rotatable_bonds}")
        print(f"Total rotatable bonds counted: {summary_df['count'].sum()}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(7, 7))
        plt.pie(
            summary_df["count"],
            labels=summary_df["rotatable_bond_class"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Rotatable Bond Ratios by Carbon Hybridisation")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nSaved pie chart to: {save_path}")

            summary_path = save_path.with_suffix(".csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary CSV to: {summary_path}")

        plt.show()

        return summary_df, counts

    df = pd.read_csv("/users/yhb18174/TL_project/datasets/all/all_rdkit.csv")

    summary_df, all_counts = plot_rotatable_bond_carbon_hybridisation_ratios(
        df,
        smiles_col="SMILES",
        save_path="/users/yhb18174/TL_project/scripts/sandbox/rot_bond_carbon_hybridisation_ratios.png",
    )


run_25 = False

if run_25:
    from collections import Counter
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem.rdchem import HybridizationType

    COMMON_ATOMS = ["O", "N", "S", "P", "F", "Cl", "Br", "I"]

    def atom_class_label(atom):
        """
        Classify atoms as:
            Csp1, Csp2, Csp3, C_other,
            O, N, S, P, F, Cl, Br, I,
            Other
        """
        symbol = atom.GetSymbol()

        if symbol == "C":
            hyb = atom.GetHybridization()

            if hyb == HybridizationType.SP:
                return "Csp1"
            elif hyb == HybridizationType.SP2:
                return "Csp2"
            elif hyb == HybridizationType.SP3:
                return "Csp3"
            else:
                return "C_other"

        if symbol in COMMON_ATOMS:
            return symbol

        return "Other"

    def plot_dataset_atom_composition_pie(
        df,
        smiles_col="SMILES",
        save_path=None,
    ):
        """
        Count all atoms across the full dataframe and plot a pie chart
        of atom composition.
        """

        atom_counts = Counter()
        invalid_smiles = 0

        for smi in df[smiles_col].dropna():
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                invalid_smiles += 1
                continue

            for atom in mol.GetAtoms():
                atom_counts[atom_class_label(atom)] += 1

        if not atom_counts:
            raise ValueError("No valid atoms found in the dataframe.")

        summary_df = pd.DataFrame(
            {
                "atom_class": list(atom_counts.keys()),
                "count": list(atom_counts.values()),
            }
        )

        summary_df["percentage"] = summary_df["count"] / summary_df["count"].sum() * 100

        summary_df = summary_df.sort_values("count", ascending=False)

        print("\nDataset atom composition:")
        print(summary_df)

        print(f"\nInvalid SMILES skipped: {invalid_smiles}")
        print(f"Total atoms counted: {summary_df['count'].sum()}")

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 8))
        plt.pie(
            summary_df["count"],
            labels=summary_df["atom_class"],
            autopct="%1.1f%%",
            startangle=90,
        )
        plt.title("Dataset Atom Composition")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"\nSaved pie chart to: {save_path}")

            summary_path = save_path.with_suffix(".csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"Saved summary CSV to: {summary_path}")

        plt.show()

        return summary_df, atom_counts

    df = pd.read_csv("/users/yhb18174/TL_project/datasets/all/all_rdkit.csv")

    atom_summary_df, atom_counts = plot_dataset_atom_composition_pie(
        df,
        smiles_col="SMILES",
        save_path="/users/yhb18174/TL_project/scripts/sandbox/atom_composition_pie.png",
    )


run_26 = False
if run_26:

    desc_features = getGroups("mordred")["EState"]
    print(len(desc_features))
    df_p = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_avg_transformers/avg_stats_across_embedding_models.csv"
    df = pd.read_csv(df_p, index_col=0)

    in_df = []
    n_reg = 0
    n_reg_not_na = 0
    n_bc = 0
    n_mc = 0
    n_other = 0

    out_df = []

    for desc in desc_features:
        if desc in df.index:
            in_df.append(desc)

            task_type = df.loc[desc, "task_type"]

            if task_type == "regression":
                n_reg += 1

                r2_value = pd.to_numeric(df.loc[desc, "r2"], errors="coerce")
                if pd.notna(r2_value):
                    n_reg_not_na += 1

            elif task_type == "binary_classification":
                n_bc += 1
            elif task_type == "multiclass_classification":
                n_mc += 1
            else:
                n_other += 1
        else:
            out_df.append(desc)

    print(f"In avg_df: {len(in_df)}")
    print(in_df)

    print("\nOut of avg_df:")
    print(f"Out: {len(out_df)}")
    print(out_df)

    print("\nTask type counts among EState descriptors present in avg_df:")
    print(f"Regression: {n_reg}")
    print(f"Regression not na: {n_reg_not_na}")
    print(f"Binary classification: {n_bc}")
    print(f"Multiclass classification: {n_mc}")
    print(f"Other/unknown: {n_other}")

run_26 = False
if run_26:
    rd = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
    pred = "mordred"
    group_map = getGroups(pred)
    models = ["molformer", "molformer-c3-1b", "chemberta", "chembertasey", "selformer"]

    model_by_task_dict = {}

    for m in models:
        exp = f"pred_{pred}_tr_{m}"

        df = pd.read_csv(rd[exp] / f"{exp}.csv", index_col=0)
        bc = df[df["task_type"] == "binary_classification"]["Balanced_Accuracy"]
        mc = df[df["task_type"] == "multiclass_classification"]["Balanced_Accuracy"]
        r = df[df["task_type"] == "regression"]["r2"]

        model_by_task_dict[m] = {
            "regression": r,
            "binary_classification": bc,
            "multiclass_classification": mc,
        }

    regression_df = pd.DataFrame(
        {
            model: task_dict["regression"]
            for model, task_dict in model_by_task_dict.items()
        }
    )
    regression_df["Average"] = regression_df.mean(axis=1, skipna=True)

    binary_classification_df = pd.DataFrame(
        {
            model: task_dict["binary_classification"]
            for model, task_dict in model_by_task_dict.items()
        }
    )
    binary_classification_df["Average"] = binary_classification_df.mean(
        axis=1, skipna=True
    )

    multiclass_classification_df = pd.DataFrame(
        {
            model: task_dict["multiclass_classification"]
            for model, task_dict in model_by_task_dict.items()
        }
    )
    multiclass_classification_df["Average"] = multiclass_classification_df.mean(
        axis=1, skipna=True
    )

    avg_df_ls = [regression_df, binary_classification_df, multiclass_classification_df]

    reg_desc_groups = {}
    bin_desc_groups = {}
    mul_desc_groups = {}

    regression_df = avg_df_ls[0]
    binary_classification_df = avg_df_ls[1]
    multiclass_classification_df = avg_df_ls[2]

    def getDescriptorGroupInfo(
        task_df, group_map, threshold_low=0.5, threshold_high=0.7
    ):
        rows = []

        for group, descriptors in group_map.items():
            values = task_df.reindex(descriptors)["Average"].dropna()

            n_0_to_0p5 = int(((values > 0) & (values <= threshold_low)).sum())
            n_0p5_to_0p7 = int(
                ((values > threshold_low) & (values <= threshold_high)).sum()
            )
            n_above_0p7 = int((values > threshold_high).sum())

            rows.append(
                {
                    "group": group,
                    "descriptors": values.index.tolist(),
                    "values": values.tolist(),
                    "n_total": len(values),
                    "n_0_to_0p5": n_0_to_0p5,
                    "n_0p5_to_0p7": n_0p5_to_0p7,
                    "n_above_0p7": n_above_0p7,
                }
            )

        full_df = pd.DataFrame(rows)

        n_total = full_df["n_total"].sum()
        n_0_to_0p5 = full_df["n_0_to_0p5"].sum()
        n_0p5_to_0p7 = full_df["n_0p5_to_0p7"].sum()
        n_above_0p7 = full_df["n_above_0p7"].sum()

        return [full_df, n_total, n_0_to_0p5, n_0p5_to_0p7, n_above_0p7]

    reg_info = getDescriptorGroupInfo(regression_df, group_map)
    reg_info[0].to_csv("/users/yhb18174/TL_project/scripts/sandbox/reg_info.csv")
    bin_info = getDescriptorGroupInfo(binary_classification_df, group_map)
    mul_info = getDescriptorGroupInfo(multiclass_classification_df, group_map)

    def plotDescriptorGroupStackedBins(
        group_info,
        task_name,
        metric_name,
        out_path=None,
        sort_by="n_total",
    ):
        full_df = group_info[0].copy()
        task_total = group_info[1]

        # Remove groups with no descriptors
        full_df = full_df.loc[full_df["n_total"] > 0].copy()

        if sort_by is not None:
            full_df = full_df.sort_values(sort_by, ascending=False)

        x = np.arange(len(full_df))

        labels = full_df["group"].astype(str).tolist()

        red_counts = full_df["n_0_to_0p5"].values
        orange_counts = full_df["n_0p5_to_0p7"].values
        green_counts = full_df["n_above_0p7"].values

        fig_width = max(10, 0.45 * len(full_df))
        fig, ax = plt.subplots(figsize=(fig_width, 6))

        ax.bar(
            x,
            red_counts,
            label="0 < x <= 0.5",
            color="red",
        )

        ax.bar(
            x,
            orange_counts,
            bottom=red_counts,
            label="0.5 < x <= 0.7",
            color="orange",
        )

        ax.bar(
            x,
            green_counts,
            bottom=red_counts + orange_counts,
            label="0.7 < x",
            color="green",
        )

        ax.set_ylabel("Number of descriptors")
        ax.set_xlabel("Descriptor group")
        ax.set_title(
            f"{task_name}: descriptor performance bins by group\n"
            f"Metric = {metric_name}; total descriptors in task = {task_total}"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="right")

        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        fig.tight_layout()

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")

        plt.show()

        return full_df

    def plotDescriptorGroupBars(
        task_df,
        group_map,
        task_name,
        metric_name,
        out_dir,
        value_col="Average",
        sort_values=True,
        show=False,
    ):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_plots = []

        for group, descriptors in group_map.items():
            group_df = task_df.reindex(descriptors).copy()

            if value_col not in group_df.columns:
                raise ValueError(f"{value_col} not found in task_df columns")

            group_df = group_df[[value_col]].dropna()

            if group_df.empty:
                continue

            if sort_values:
                group_df = group_df.sort_values(value_col, ascending=False)

            labels = group_df.index.astype(str).tolist()
            values = group_df[value_col].values

            fig_width = max(8, 0.35 * len(group_df))
            fig, ax = plt.subplots(figsize=(fig_width, 5))

            ax.bar(np.arange(len(group_df)), values)

            ax.set_title(f"{task_name}: {group}\n" f"Metric = {metric_name}")
            ax.set_ylabel(metric_name)
            ax.set_xlabel("Descriptor")

            ax.set_xticks(np.arange(len(group_df)))
            ax.set_xticklabels(labels, rotation=60, ha="right")

            ax.grid(axis="y", linestyle="--", alpha=0.3)

            ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.5)
            ax.axhline(0.7, linestyle="--", linewidth=1, alpha=0.5)

            fig.tight_layout()

            out_path = out_dir / f"{pred}_{task_name}_{group}_barplot.png"
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            saved_plots.append(out_path)

            if show:
                plt.show()
            else:
                plt.close(fig)

        return saved_plots

    plot_dir = Path("descriptor_group_stacked_bin_plots")

    reg_plot_df = plotDescriptorGroupStackedBins(
        reg_info,
        task_name="Regression",
        metric_name="r2",
        out_path=plot_dir / f"{pred}_regression_descriptor_group_bins.png",
    )

    bin_plot_df = plotDescriptorGroupStackedBins(
        bin_info,
        task_name="Binary classification",
        metric_name="Balanced Accuracy",
        out_path=plot_dir / f"{pred}_binary_classification_descriptor_group_bins.png",
    )

    mul_plot_df = plotDescriptorGroupStackedBins(
        mul_info,
        task_name="Multiclass classification",
        metric_name="Balanced Accuracy",
        out_path=plot_dir
        / f"{pred}_multiclass_classification_descriptor_group_bins.png",
    )

    bar_plot_dir = Path("/users/yhb18174/TL_project/scripts/sandbox/")

    reg_bar_paths = plotDescriptorGroupBars(
        regression_df,
        group_map,
        task_name="Regression",
        metric_name="r2",
        out_dir=bar_plot_dir / f"{pred}_regression",
    )

    bin_bar_paths = plotDescriptorGroupBars(
        binary_classification_df,
        group_map,
        task_name="Binary classification",
        metric_name="Balanced Accuracy",
        out_dir=bar_plot_dir / f"{pred}_binary_classification",
    )

    mul_bar_paths = plotDescriptorGroupBars(
        multiclass_classification_df,
        group_map,
        task_name="Multiclass classification",
        metric_name="Balanced Accuracy",
        out_dir=bar_plot_dir / f"{pred}_multiclass_classification",
    )


run_27 = False

if run_27:
    mordred = pd.read_csv(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
        "pred_mordred_tr_rdkit/pred_mordred_tr_rdkit.csv",
        index_col=0,
    )

    rdkit = pd.read_csv(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
        "pred_rdkit_tr_mordred/pred_rdkit_tr_mordred.csv",
        index_col=0,
    )

    maccs = pd.read_csv(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/"
        "pred_maccs_tr_mordred/pred_maccs_tr_mordred.csv",
        index_col=0,
    )

    dfs = {
        "Mordred": mordred,
        "RDKit": rdkit,
        "MACCS": maccs,
    }

    task_types = [
        "regression",
        "binary_classification",
        "multiclass_classification",
    ]

    for name, df in dfs.items():
        print(f"\n{name}")

        for task in task_types:
            n = len(df.loc[df["task_type"] == task])
            print(f"  n_{task} = {n}")


run_28 = False

if run_28:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    path = "/users/yhb18174/TL_project/datasets/"

    dfs = [
        # ("boiling_point/cleaned_boiling_point.csv", "bp", "Boiling_Point"),
        ("pka/cleaned_pka.csv", "pKa", "pKa"),
        #     ("pka/cleaned_pka_paper1_acidic.csv", "pKa_acid", "pKa"),
        #     ("pka/cleaned_pka_paper1_basic.csv", "pKa_basic", "pKa"),
        #     ("pic50/cleaned_pic50.csv", "pIC50", "pIC50"),
        #     ("LOG_LD50/cleaned_log_ld50.csv", "log_LD50", "LOG_LD50"),
        #     ("logD/cleaned_logd.csv", "logD", "LogD"),
        #     ("hole_re/hole_re_cleaned.csv", "hole_re", "Hole_Reorganisation_Energy"),
        #     ("elec_re/elec_re_cleaned.csv", "elec_re", "Electron_Reorganisation_Energy"),
        #     ("aq_sol/aq_sol_cleaned.csv", "aq_sol", "Solubility"),
        #     ("homo_lumo_gap/homo_lumo_gap_cleaned.csv", "homo_lumo_gap", "homolumogap"),
        #     ("egfr_pic50/egfr_pic50.csv", "egfr_pIC50", "pIC50"),
    ]

    for p, save_name, col in dfs[:3]:
        df = pd.read_csv(path + p, index_col=0)

        values = df[col].dropna()

        mean_val = values.mean()
        min_val = values.min()
        max_val = values.max()
        data_range = max_val - min_val

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_iqr = q1 - 3.0 * iqr
        upper_iqr = q3 + 3.0 * iqr

        outlier_mask = (df[col] < lower_iqr) | (df[col] > upper_iqr)
        outliers = df.loc[outlier_mask].copy()

        plt.figure(figsize=(8, 5))

        # Low-tint box showing non-outlier region by 3x IQR
        plt.axvspan(
            lower_iqr,
            upper_iqr,
            alpha=0.12,
            label=f"3x IQR bounds: {lower_iqr:.2f} to {upper_iqr:.2f}",
        )

        sns.histplot(values, kde=True, bins=30)

        # Keep only the mean line
        plt.axvline(
            mean_val, linestyle="-", linewidth=2, label=f"Mean = {mean_val:.2f}"
        )

        plt.xlabel(save_name)
        plt.ylabel("Count")
        plt.title(
            f"Distribution of {save_name} values\n"
            f"Range: {min_val:.2f} to {max_val:.2f} | "
            f"Outliers flagged: {outlier_mask.sum()}"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(f"{save_name}_dist_3xIQR.png", dpi=300, bbox_inches="tight")
        plt.show()

        outliers.to_csv(f"{save_name}_target_outliers_3xIQR.csv")

        print(f"{save_name}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Range: {data_range:.4f}")
        print(f"  Q1: {q1:.4f}")
        print(f"  Q3: {q3:.4f}")
        print(f"  IQR: {iqr:.4f}")
        print(f"  3x IQR bounds: {lower_iqr:.4f} to {upper_iqr:.4f}")
        print(f"  Outliers flagged: {outlier_mask.sum()} / {df[col].notna().sum()}")
        print()

run_30 = False
if run_30:
    import sys
    import pandas as pd

    SRC_DIR = Path("/users/yhb18174/TL_project/scripts/src/")
    sys.path.insert(0, str(SRC_DIR / "datasets"))
    from group_descriptors import getGroups

    slogp_descriptors = getGroups("mordred")["LogS"]

    mordred_df = pd.read_csv(
        "/users/yhb18174/TL_project/datasets/descriptors/LOGD_descriptors/logd_mordred_1.csv",
        index_col=0,
        usecols=["ID"] + slogp_descriptors,
    )

    aq_sol_df = pd.read_csv(
        paths["targets"]["logd"],
        index_col=0,
    )

    joined_df = mordred_df.join(aq_sol_df[["LogD"]], how="inner")

    corrs = joined_df[slogp_descriptors].corrwith(joined_df["LogD"])

    avg_corr = corrs.mean()

    print(corrs.sort_values(ascending=False))
    print(f"Average SLogP-group correlation: {avg_corr}")

run_31 = False
if run_31:

    def plotGroupedRFPerformanceBar(
        base_ls: list[str],
        base_plus_ls: list[str],
        base_label: str,
        base_plus_label: str,
        metric_path: tuple[str, str, str] = ("external", "mean", "r2"),
        save_path: str | Path = "grouped_rf_performance_bar.png",
        y_label: str = "External R2",
        title: str = None,
    ):
        import json
        from pathlib import Path

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        def get_property_from_path(path: str | Path) -> str:
            path = Path(path)
            pred_dir = next(
                part for part in path.parts if part.endswith("_predictions_rf")
            )
            return pred_dir.replace("_predictions_rf", "")

        def read_metric(perf: dict, keys: tuple[str, ...]):
            value = perf
            for key in keys:
                value = value[key]
            return value

        rows = []

        for label, paths in {
            base_label: base_ls,
            base_plus_label: base_plus_ls,
        }.items():
            for p in paths:
                p = Path(p)
                prop = get_property_from_path(p)

                with open(p) as f:
                    perf = json.load(f)

                rows.append(
                    {
                        "property": prop,
                        "feature_set": label,
                        "metric": read_metric(perf, metric_path),
                    }
                )

        plot_df = pd.DataFrame(rows)

        summary_df = plot_df.pivot_table(
            index="property",
            columns="feature_set",
            values="metric",
            aggfunc="first",
        ).sort_index()

        print(summary_df)
        print("\nMissing paired results:")
        print(summary_df[summary_df.isna().any(axis=1)])

        property_order = summary_df.index.tolist()
        hue_order = [base_label, base_plus_label]

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(14, 6))
        sns.barplot(
            data=plot_df,
            x="property",
            y="metric",
            hue="feature_set",
            order=property_order,
            hue_order=hue_order,
        )

        plt.ylim(0, 1)
        plt.xlabel("Property")
        plt.ylabel(y_label)
        plt.title(title or f"{base_label} vs {base_plus_label}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()

        print(f"Saved to: {save_path}")

        return plot_df, summary_df

    base_feature = "ft-molformer-c3-1b"

    base_ls = glob(
        f"/users/yhb18174/TL_project/results/*_predictions_rf/"
        f"{base_feature}/rf_performance.json"
    )

    base_plus_mordred_ls = glob(
        f"/users/yhb18174/TL_project/results/*_predictions_rf/"
        f"{base_feature}/additional_features/mordred/rf_performance.json"
    )

    plotGroupedRFPerformanceBar(
        base_ls=base_ls,
        base_plus_ls=base_plus_mordred_ls,
        base_label="ft-molformer-c3-1b",
        base_plus_label="ft-molformer-c3-1b + Mordred",
        save_path=(
            "/users/yhb18174/TL_project/results/pp_analysis/"
            "ft_molformer_c3_1b_mordred_grouped_bar.png"
        ),
        y_label="External R2",
        title="ft-molformer-c3-1b vs ft-molformer-c3-1b + Mordred",
    )


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path

run_32 = False

if run_32:
    pka_targ = pd.read_csv(
        "/users/yhb18174/TL_project/datasets/pka/cleaned_pka.csv",
        index_col=0,
    )

    pka_preds_ls = glob(
        "/users/yhb18174/TL_project/results/PKA_predictions_rf/*/last_20pct_pred.csv.gz"
    )

    def bin_pka(pka: float) -> str:
        if pka < 0:
            return "strong_acid"
        elif pka < 5:
            return "moderate_acid"
        elif pka < 7:
            return "weak_acid"
        elif pka < 9:
            return "weak_base"
        elif pka < 12:
            return "moderate_base"
        else:
            return "strong_base"

    pka_targ["pKa"] = pd.to_numeric(pka_targ["pKa"], errors="coerce")
    pka_targ = pka_targ.dropna(subset=["pKa"]).copy()
    pka_targ["pka_bin"] = pka_targ["pKa"].apply(bin_pka)

    bin_order = [
        "strong_acid",
        "moderate_acid",
        "weak_acid",
        "weak_base",
        "moderate_base",
        "strong_base",
    ]

    bin_rows = []
    min_max_rows = []
    hist_rows = []

    for pred_path in pka_preds_ls:
        pred_path = Path(pred_path)
        feature_set = pred_path.parts[-2]

        pred_df = pd.read_csv(pred_path, index_col=0)

        pred_col = "pKa" if "pKa" in pred_df.columns else pred_df.columns[0]
        pred_df = pred_df[[pred_col]].rename(columns={pred_col: "pred_pKa"})

        joined_df = pred_df.join(
            pka_targ[["pKa", "pka_bin"]].rename(columns={"pKa": "true_pKa"}),
            how="inner",
        )

        min_max_rows.append(
            {
                "feature_set": feature_set,
                "n_matched": len(joined_df),
                "true_min_pKa": joined_df["true_pKa"].min(),
                "true_max_pKa": joined_df["true_pKa"].max(),
            }
        )

        hist_rows.append(
            joined_df[["true_pKa"]]
            .rename(columns={"true_pKa": "pKa"})
            .assign(feature_set=feature_set)
        )

        for pka_bin, bin_df in joined_df.groupby("pka_bin"):
            bin_rows.append(
                {
                    "feature_set": feature_set,
                    "pka_bin": pka_bin,
                    "mean_prediction": bin_df["pred_pKa"].mean(),
                    "n": len(bin_df),
                    "true_min_pKa": bin_df["true_pKa"].min(),
                    "true_max_pKa": bin_df["true_pKa"].max(),
                }
            )

    plot_df = pd.DataFrame(bin_rows)
    min_max_df = pd.DataFrame(min_max_rows).sort_values("feature_set")
    hist_df = pd.concat(hist_rows, axis=0).reset_index(drop=True)

    print(min_max_df.to_string(index=False))

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=plot_df,
        x="pka_bin",
        y="mean_prediction",
        hue="feature_set",
        order=bin_order,
    )
    plt.xlabel("True pKa bin")
    plt.ylabel("Mean predicted pKa")
    plt.title("Mean PKA Predictions by True pKa Bin")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("pka_anal_test.png", dpi=400, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=hist_df,
        x="pKa",
        hue="feature_set",
        bins=30,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.xlabel("True pKa")
    plt.ylabel("Count")
    plt.title("True pKa Distribution for IDs in last_20pct_pred")
    plt.tight_layout()
    plt.savefig("pka_hist_test.png", dpi=400, bbox_inches="tight")
    plt.close()

run_33 = False
if run_33:
    from pathlib import Path
    import shutil

    results_root = Path("/users/yhb18174/TL_project/results")

    nested_dirs = sorted(results_root.glob("*_predictions_rf/*/*_pred_*"))

    for nested_dir in nested_dirs:
        if not nested_dir.is_dir():
            continue

        feature_dir = nested_dir.parent

        print(f"Unpacking: {nested_dir} -> {feature_dir}")

        for item in nested_dir.iterdir():
            dest = feature_dir / item.name

            if dest.exists():
                print(f"  SKIP exists: {dest}")
                continue

            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    print("Done.")


run_35 = False
if run_35:
    paths = getPaths()
    from rdkit.Chem import rdFingerprintGenerator

    tanimoto_dir = SANDBOX / "tanimoto_dataset_similarity"
    tanimoto_dir.mkdir(parents=True, exist_ok=True)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    def smilesToMorganFP(smiles: str):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return morgan_gen.GetFingerprint(mol)

    def getSmilesColumn(df: pd.DataFrame) -> str | None:
        for col in ["SMILES", "smiles", "Smiles"]:
            if col in df.columns:
                return col
        return None

    def pairwiseTanimotoSummary(
        df: pd.DataFrame,
        dataset_name: str,
        bins: int = 50,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        smiles_col = getSmilesColumn(df)
        if smiles_col is None:
            raise ValueError(f"No SMILES column found for {dataset_name}")

        smiles_df = df[[smiles_col]].dropna().copy()
        smiles_df.index = smiles_df.index.astype(str)

        fps = []
        ids = []
        for mol_id, smiles in smiles_df[smiles_col].items():
            fp = smilesToMorganFP(smiles)
            if fp is None:
                continue
            ids.append(mol_id)
            fps.append(fp)

        n_mols = len(fps)
        if n_mols < 2:
            raise ValueError(f"Need at least 2 valid molecules for {dataset_name}")

        hist_edges = np.linspace(0, 1, bins + 1)
        hist_counts = np.zeros(bins, dtype=np.int64)
        nearest = np.zeros(n_mols, dtype=float)

        n_pairs = 0
        sim_sum = 0.0
        sim_min = 1.0
        sim_max = 0.0

        for i, fp in enumerate(fps[:-1]):
            sims = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :]),
                dtype=float,
            )
            if sims.size == 0:
                continue

            n_pairs += sims.size
            sim_sum += float(sims.sum())
            sim_min = min(sim_min, float(sims.min()))
            sim_max = max(sim_max, float(sims.max()))
            hist_counts += np.histogram(sims, bins=hist_edges)[0]

            nearest[i] = max(nearest[i], float(sims.max()))
            nearest[i + 1 :] = np.maximum(nearest[i + 1 :], sims)

        hist_df = pd.DataFrame(
            {
                "dataset": dataset_name,
                "bin_left": hist_edges[:-1],
                "bin_right": hist_edges[1:],
                "count": hist_counts,
            }
        )
        hist_df["proportion"] = hist_df["count"] / hist_df["count"].sum()
        hist_df["bin_mid"] = (hist_df["bin_left"] + hist_df["bin_right"]) / 2

        approx_median = hist_df.loc[
            hist_df["count"].cumsum().ge(n_pairs / 2),
            "bin_mid",
        ].iloc[0]

        summary = {
            "dataset": dataset_name,
            "n_molecules": n_mols,
            "n_pairs": n_pairs,
            "mean_pairwise_tanimoto": sim_sum / n_pairs,
            "approx_median_pairwise_tanimoto": approx_median,
            "min_pairwise_tanimoto": sim_min,
            "max_pairwise_tanimoto": sim_max,
            "mean_nearest_neighbour_tanimoto": float(nearest.mean()),
            "min_nearest_neighbour_tanimoto": float(nearest.min()),
            "max_nearest_neighbour_tanimoto": float(nearest.max()),
        }

        nearest_df = pd.DataFrame(
            {
                "dataset": dataset_name,
                "ID": ids,
                "nearest_neighbour_tanimoto": nearest,
            }
        )

        return summary, hist_df, nearest_df

    summary_rows = []
    hist_dfs = []
    nearest_dfs = []

    for dataset_name, target_path in paths["targets"].items():
        try:
            print(f"Processing Tanimoto similarity for {dataset_name}")
            target_df = pd.read_csv(target_path, index_col=0)
            summary, hist_df, nearest_df = pairwiseTanimotoSummary(
                df=target_df,
                dataset_name=dataset_name,
            )
            summary_rows.append(summary)
            hist_dfs.append(hist_df)
            nearest_dfs.append(nearest_df)
            print(summary)
        except Exception as e:
            print(f"Skipping {dataset_name}: {e}")

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    hist_df = pd.concat(hist_dfs, ignore_index=True)
    nearest_df = pd.concat(nearest_dfs, ignore_index=True)

    summary_df.to_csv(tanimoto_dir / "dataset_tanimoto_summary.csv", index=False)
    hist_df.to_csv(tanimoto_dir / "dataset_tanimoto_histograms.csv", index=False)
    nearest_df.to_csv(
        tanimoto_dir / "dataset_nearest_neighbour_tanimoto.csv", index=False
    )

    plt.figure(figsize=(14, 7))
    sns.lineplot(
        data=hist_df,
        x="bin_mid",
        y="proportion",
        hue="dataset",
    )
    plt.xlabel("Pairwise Tanimoto similarity")
    plt.ylabel("Proportion of molecule pairs")
    plt.title("Pairwise Morgan Tanimoto Similarity Across Datasets")
    plt.tight_layout()
    plt.savefig(
        tanimoto_dir / "dataset_pairwise_tanimoto_histograms.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(14, 6))
    sns.barplot(
        data=summary_df,
        x="dataset",
        y="mean_nearest_neighbour_tanimoto",
        hue="dataset",
        legend=False,
    )
    plt.ylim(0, 1)
    plt.xlabel("Dataset")
    plt.ylabel("Mean nearest-neighbour Tanimoto")
    plt.title("Mean Nearest-Neighbour Tanimoto Similarity by Dataset")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(
        tanimoto_dir / "dataset_mean_nearest_neighbour_tanimoto.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(14, 7))
    sns.histplot(
        data=nearest_df,
        x="nearest_neighbour_tanimoto",
        hue="dataset",
        bins=50,
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.xlabel("Nearest-neighbour Tanimoto similarity")
    plt.ylabel("Number of molecules")
    plt.title("Nearest-neighbour Tanimoto Similarity Distribution")
    plt.tight_layout()
    plt.savefig(
        tanimoto_dir / "dataset_nearest_neighbour_tanimoto_histogram_counts.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()

    for dataset_name, dataset_nearest_df in nearest_df.groupby("dataset"):
        plt.figure(figsize=(10, 6))
        sns.histplot(
            data=dataset_nearest_df,
            x="nearest_neighbour_tanimoto",
            bins=50,
            stat="count",
        )
        plt.xlabel("Nearest-neighbour Tanimoto similarity")
        plt.ylabel("Number of molecules")
        plt.title(f"{dataset_name} nearest-neighbour Tanimoto distribution")
        plt.tight_layout()
        plt.savefig(
            tanimoto_dir
            / f"{dataset_name}_nearest_neighbour_tanimoto_histogram_counts.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close()

    def averageTanimotoWithin(fps: list) -> tuple[float, int]:
        if len(fps) < 2:
            return np.nan, 0

        sim_sum = 0.0
        n_pairs = 0

        for i, fp in enumerate(fps[:-1]):
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps[i + 1 :])
            sim_sum += float(np.sum(sims))
            n_pairs += len(sims)

        return sim_sum / n_pairs, n_pairs

    def averageTanimotoBetween(fps_a: list, fps_b: list) -> tuple[float, int]:
        if not fps_a or not fps_b:
            return np.nan, 0

        sim_sum = 0.0
        n_pairs = 0

        for fp in fps_a:
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps_b)
            sim_sum += float(np.sum(sims))
            n_pairs += len(sims)

        return sim_sum / n_pairs, n_pairs

    def buildFingerprintMap(df: pd.DataFrame, dataset_name: str) -> dict[str, object]:
        smiles_col = getSmilesColumn(df)
        if smiles_col is None:
            raise ValueError(f"No SMILES column found for {dataset_name}")

        fp_map = {}
        for mol_id, smiles in df[smiles_col].dropna().items():
            fp = smilesToMorganFP(smiles)
            if fp is not None:
                fp_map[str(mol_id)] = fp

        return fp_map

    def readSplitIDs(split_path: Path) -> list[str]:
        split_df = pd.read_csv(split_path)
        id_col = "ID" if "ID" in split_df.columns else split_df.columns[0]
        return split_df[id_col].astype(str).tolist()

    def getRepresentativeSplitDirs() -> list[tuple[str, str, Path]]:
        feature_priority = [
            "rdkit",
            "mordred",
            "chemberta",
            "molformer",
            "molformer-c3-1b",
            "selformer",
        ]

        split_dirs = []
        rf_paths = paths["prediction_output_dirs"]["rf"]

        for dataset_name in paths["targets"]:
            if dataset_name not in rf_paths:
                continue

            for feature_name in feature_priority:
                result_dir = Path(rf_paths[dataset_name].get(feature_name, ""))
                if (result_dir / "repeats").exists():
                    split_dirs.append((dataset_name, feature_name, result_dir))
                    break

        return split_dirs

    split_rows = []
    split_dirs = getRepresentativeSplitDirs()

    for dataset_name, feature_name, result_dir in split_dirs:
        try:
            print(
                "Processing split Tanimoto similarity for "
                f"{dataset_name} / {feature_name}"
            )
            target_df = pd.read_csv(paths["targets"][dataset_name], index_col=0)
            fp_map = buildFingerprintMap(target_df, dataset_name)

            repeat_dirs = sorted((result_dir / "repeats").glob("repeat_*"))
            for repeat_dir in repeat_dirs:
                train_path = repeat_dir / "training_data" / "split_train_ids.csv"
                validation_path = (
                    repeat_dir / "training_data" / "split_validation_ids.csv"
                )

                if not train_path.exists() or not validation_path.exists():
                    continue

                train_ids = readSplitIDs(train_path)
                validation_ids = readSplitIDs(validation_path)

                train_fps = [fp_map[mol_id] for mol_id in train_ids if mol_id in fp_map]
                validation_fps = [
                    fp_map[mol_id] for mol_id in validation_ids if mol_id in fp_map
                ]

                train_avg, train_pairs = averageTanimotoWithin(train_fps)
                validation_avg, validation_pairs = averageTanimotoWithin(validation_fps)
                between_avg, between_pairs = averageTanimotoBetween(
                    train_fps,
                    validation_fps,
                )

                repeat_name = repeat_dir.name
                split_rows.extend(
                    [
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_name,
                            "repeat": repeat_name,
                            "comparison": "train_train",
                            "avg_tanimoto": train_avg,
                            "n_molecules_a": len(train_fps),
                            "n_molecules_b": len(train_fps),
                            "n_pairs": train_pairs,
                            "result_dir": str(result_dir),
                        },
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_name,
                            "repeat": repeat_name,
                            "comparison": "validation_validation",
                            "avg_tanimoto": validation_avg,
                            "n_molecules_a": len(validation_fps),
                            "n_molecules_b": len(validation_fps),
                            "n_pairs": validation_pairs,
                            "result_dir": str(result_dir),
                        },
                        {
                            "dataset": dataset_name,
                            "feature_set": feature_name,
                            "repeat": repeat_name,
                            "comparison": "train_validation",
                            "avg_tanimoto": between_avg,
                            "n_molecules_a": len(train_fps),
                            "n_molecules_b": len(validation_fps),
                            "n_pairs": between_pairs,
                            "result_dir": str(result_dir),
                        },
                    ]
                )

        except Exception as e:
            print(f"Skipping split Tanimoto for {dataset_name} / {feature_name}: {e}")

    if split_rows:
        split_tanimoto_df = pd.DataFrame(split_rows)
        split_tanimoto_df["dataset_repeat"] = (
            split_tanimoto_df["dataset"] + " " + split_tanimoto_df["repeat"]
        )
        split_tanimoto_df.to_csv(
            tanimoto_dir / "train_validation_split_tanimoto_summary.csv",
            index=False,
        )

        comparison_order = [
            "train_train",
            "validation_validation",
            "train_validation",
        ]
        titles = {
            "train_train": "Training molecules",
            "validation_validation": "Validation molecules",
            "train_validation": "Training vs validation molecules",
        }

        fig, axes = plt.subplots(1, 3, figsize=(22, 6), sharey=True)
        for ax, comparison in zip(axes, comparison_order):
            plot_df = split_tanimoto_df[
                split_tanimoto_df["comparison"] == comparison
            ].copy()
            sns.barplot(
                data=plot_df,
                x="dataset",
                y="avg_tanimoto",
                hue="repeat",
                ax=ax,
            )
            ax.set_ylim(0, 1)
            ax.set_title(titles[comparison])
            ax.set_xlabel("Dataset")
            ax.set_ylabel("Average Tanimoto" if ax is axes[0] else "")
            ax.tick_params(axis="x", labelrotation=45)

        handles, labels = axes[-1].get_legend_handles_labels()
        for ax in axes:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        fig.legend(
            handles,
            labels,
            title="Repeat",
            bbox_to_anchor=(1.01, 0.98),
            loc="upper left",
        )
        fig.suptitle("Average Tanimoto Similarity Across Train/Validation Splits")
        fig.tight_layout()
        fig.savefig(
            tanimoto_dir / "train_validation_split_tanimoto_3panel.png",
            dpi=400,
            bbox_inches="tight",
        )
        plt.close(fig)

    print(summary_df.to_string(index=False))
    print(f"Saved Tanimoto outputs to: {tanimoto_dir}")


run_36 = False
if run_36:
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    results_root = Path("/users/yhb18174/TL_project/results")
    cfp_root = results_root / "cross_feature_predictions"

    base_exp = "pred_mordred_tr_molformer-c3-1b"
    ft_exp = "pred_mordred_tr_ft-molformer-c3-1b"

    targets = {
        "bp": {
            "rf_dir": "BP_predictions_rf",
            "target_col": "Boiling_Point",
        },
        "logd": {
            "rf_dir": "LOGD_predictions_rf",
            "target_col": "LogD",
        },
        "pka": {
            "rf_dir": "PKA_predictions_rf",
            "target_col": "pKa",
        },
        "hole_re": {
            "rf_dir": "HOLE_RE_predictions_rf",
            "target_col": "Hole_Reorganisation_Energy",
        },
        "elec_re": {
            "rf_dir": "ELEC_RE_predictions_rf",
            "target_col": "Electron_Reorganisation_Energy",
        },
        "aq_sol": {
            "rf_dir": "AQ_SOL_predictions_rf",
            "target_col": "Solubility",
        },
        "egfr_pic50": {
            "rf_dir": "EGFR_PIC50_predictions_rf",
            "target_col": "pIC50",
        },
    }

    tasks = [
        {
            "task": "regression",
            "metric": "r2",
            "label": "Delta R2",
        },
        {
            "task": "binary_classification",
            "metric": "Balanced_Accuracy",
            "label": "Delta Balanced Accuracy",
        },
        {
            "task": "multiclass_classification",
            "metric": "Balanced_Accuracy",
            "label": "Delta Balanced Accuracy",
        },
    ]

    def load_avg_importance(rf_root, target_col):
        files = sorted(
            rf_root.glob(f"repeats/repeat_*/{target_col}_feature_importance.csv")
        )

        if not files:
            print(f"No feature importance files found in: {rf_root}")
            return None

        series = []

        for f in files:
            df = pd.read_csv(f)

            feature_col = "Feature" if "Feature" in df.columns else df.columns[0]
            importance_cols = [
                c for c in df.columns if c.lower().startswith("importance")
            ]

            if not importance_cols:
                print(f"Skipping {f}: no importance column")
                continue

            importance_col = importance_cols[0]

            s = pd.Series(
                pd.to_numeric(df[importance_col], errors="coerce").values,
                index=df[feature_col].astype(str),
            )
            series.append(s)

        if not series:
            return None

        return pd.concat(series, axis=1).mean(axis=1).sort_values(ascending=False)

    for target, cfg in targets.items():
        rf_root = results_root / cfg["rf_dir"] / "mordred"
        avg_imp = load_avg_importance(rf_root, cfg["target_col"])

        if avg_imp is None or avg_imp.empty:
            print(f"Skipping {target}: no valid feature importance")
            continue

        top_features = avg_imp.head(25).index.astype(str).tolist()

        base_dir = cfp_root / target / base_exp
        ft_dir = cfp_root / target / ft_exp

        for p in tasks:
            base_csv = base_dir / f"{base_exp}.csv"
            ft_csv = ft_dir / f"{ft_exp}.csv"

            if not base_csv.exists() or not ft_csv.exists():
                print(f"Skipping {target} / {p['task']}: missing CFP csv")
                continue

            base_df = pd.read_csv(base_csv, index_col=0)
            ft_df = pd.read_csv(ft_csv, index_col=0)

            if "task_type" in base_df.columns:
                base_df = base_df.loc[base_df["task_type"] == p["task"]]

            if "task_type" in ft_df.columns:
                ft_df = ft_df.loc[ft_df["task_type"] == p["task"]]

            if p["metric"] not in base_df.columns or p["metric"] not in ft_df.columns:
                print(f"Skipping {target} / {p['task']}: missing {p['metric']}")
                continue

            common = [
                feat
                for feat in top_features
                if feat in base_df.index and feat in ft_df.index
            ]

            if not common:
                print(f"Skipping {target} / {p['task']}: no top features found in CFP")
                continue

            base = pd.to_numeric(base_df.loc[common, p["metric"]], errors="coerce")
            ft = pd.to_numeric(ft_df.loc[common, p["metric"]], errors="coerce")

            delta = (ft - base).dropna().sort_values()

            if delta.empty:
                print(f"Skipping {target} / {p['task']}: no valid deltas")
                continue

            plt.figure(figsize=(8, max(5, 0.35 * len(delta))))
            plt.barh(
                delta.index,
                delta.values,
                color=["#c44e52" if x < 0 else "#55a868" for x in delta],
            )
            plt.axvline(0, color="black", linewidth=1)

            plt.xlabel(f"{p['label']} (fine-tuned - base)")
            plt.ylabel("Top RF-important Mordred feature")
            plt.title(
                f"{target}: {p['label']} over top 25 RF-important Mordred features"
            )
            plt.tight_layout()

            out = (
                ft_dir
                / f"{target}_top25_rf_importance_delta_{p['metric']}_{p['task']}.png"
            )
            plt.savefig(out, dpi=300)
            plt.close()

            print(f"Saved: {out}")

run_37 = True
if run_37:

    # targ = "/users/yhb18174/TL_project/datasets/boiling_point/cleaned_boiling_point.csv"
    # o = "/users/yhb18174/TL_project/results/BP_predictions_rf/molformer-c3-1b/last_20pct_pred.csv.gz"
    # ft = "/users/yhb18174/TL_project/results/BP_predictions_rf/ft-scaffold-molformer-c3-1b/last_20pct_pred.csv.gz"

    targ = "/users/yhb18174/TL_project/datasets/pka/cleaned_pka_paper1_basic.csv"
    o = "/users/yhb18174/TL_project/results/PKA_P1B_predictions_rf/chembertasey/last_20pct_pred.csv.gz"
    ft = "/users/yhb18174/TL_project/results/PKA_P1B_predictions_rf/ft-scaffold-chembertasey/last_20pct_pred.csv.gz"

    # targ = "/users/yhb18174/TL_project/datasets/pka/cleaned_pka_paper1_acidic.csv"
    # o = "/users/yhb18174/TL_project/results/PKA_P1A_predictions_rf/chembertasey/last_20pct_pred.csv.gz"
    # ft = "/users/yhb18174/TL_project/results/PKA_P1A_predictions_rf/ft-scaffold-chembertasey/last_20pct_pred.csv.gz"

    target_col = "pKa"

    targ_df = pd.read_csv(targ, index_col="ID")[[target_col]]
    o_df = pd.read_csv(o, index_col="ID")
    ft_df = pd.read_csv(ft, index_col="ID")

    o_pred = o_df[target_col].rename("base_pred")
    ft_pred = ft_df[target_col].rename("ft_pred")

    plot_df = targ_df.join(o_pred, how="inner").join(ft_pred, how="inner")
    plot_df = plot_df.apply(pd.to_numeric, errors="coerce").dropna()

    q1 = plot_df[target_col].quantile(0.25)
    q3 = plot_df[target_col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (3 * iqr)
    upper = q3 + (3 * iqr)

    plot_df_3iqr = plot_df.loc[plot_df[target_col].between(lower, upper)].copy()

    lim_min = plot_df_3iqr[[target_col, "base_pred", "ft_pred"]].min().min()
    lim_max = plot_df_3iqr[[target_col, "base_pred", "ft_pred"]].max().max()

    # plot_df_3iqr_no_high_pred = plot_df_3iqr.loc[
    #     (plot_df_3iqr["base_pred"] <= 1000) & (plot_df_3iqr["ft_pred"] <= 1000)
    # ].copy()

    print(f"3xIQR bounds: {lower:.3f} to {upper:.3f}")
    print(f"Rows before 3xIQR trim: {len(plot_df)}")
    print(f"Rows after 3xIQR trim: {len(plot_df_3iqr)}")
    # print(f"Rows after removing predictions > 1000: {len(plot_df_3iqr_no_high_pred)}")

    def calc_metrics(df, pred_col):
        y_true = df[target_col]
        y_pred = df[pred_col]

        ss_res = ((y_true - y_pred) ** 2).sum()
        ss_tot = ((y_true - y_true.mean()) ** 2).sum()

        r2 = 1 - (ss_res / ss_tot)
        pearson_r = y_true.corr(y_pred, method="pearson")
        rmse = ((y_true - y_pred) ** 2).mean() ** 0.5

        return r2, pearson_r, rmse

    def make_plot(df, save_name, title_suffix):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

        for ax, pred_col, title in [
            (axes[0], "base_pred", "Original"),
            (axes[1], "ft_pred", "Fine-tuned"),
        ]:
            r2, pearson_r, rmse = calc_metrics(df, pred_col)

            ax.scatter(
                df[target_col],
                df[pred_col],
                s=18,
                alpha=0.65,
            )
            ax.plot([lim_min, lim_max], [lim_min, lim_max], color="black", linewidth=1)

            ax.text(
                0.97,
                0.97,
                f"R2 = {r2:.3f}\nPearson r = {pearson_r:.3f}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.8),
            )

            ax.set_title(f"{title}, {title_suffix}")
            ax.set_xlabel("True pKa")
            ax.set_ylabel("Predicted pKa")
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)

        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()

    make_plot(
        plot_df_3iqr,
        "o_vs_ft_scatterplot_3xIQR_with_high_preds_pka_b.png",
        "3xIQR trimmed",
    )

    # make_plot(
    #     plot_df_3iqr_no_high_pred,
    #     "o_vs_ft_scatterplot_3xIQR_no_preds_above_1000.png",
    #     "3xIQR trimmed, preds <= 1000",
    # )
