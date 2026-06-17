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
    csv_paths = ["pred_mordred_tr_maccs/pred_mordred_tr_maccs.csv", "pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv"]

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
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_rdkit/pred_mordred_tr_rdkit.csv"
    )
    save_dir = csv_path.parent

    r2_cutoff = 0.6
    avg_r2_line = 0.8315

    df = pd.read_csv(csv_path, index_col=0)

    autocorr_features = getGroups("mordred")["Autocorrelation"]

    autocorr_df = df.loc[
        df.index.isin(autocorr_features) & (df["task_type"] == "regression")
    ].copy()

    autocorr_df["r2"] = pd.to_numeric(autocorr_df["r2"], errors="coerce")
    autocorr_df = autocorr_df.dropna(subset=["r2"])

    def clean_name(name):
        return str(name).replace("_mordred", "").replace("_rdkit", "")

    def get_lag(descriptor):
        match = re.search(r"\d+", clean_name(descriptor))
        return int(match.group()) if match else None

    def get_family(descriptor):
        match = re.match(r"([A-Za-z]+)\d+", clean_name(descriptor))
        return match.group(1) if match else "Unknown"

    # 1. Bar plot of low-r2 Autocorrelation descriptors
    low_r2_df = autocorr_df.loc[autocorr_df["r2"] < r2_cutoff].sort_values(
        "r2",
        ascending=True,
    )

    print(f"\nAutocorrelation descriptors with r2 < {r2_cutoff}:")
    for descriptor, row in low_r2_df.iterrows():
        print(f"{descriptor}: {row['r2']:.4f}")

    labels = [clean_name(idx) for idx in low_r2_df.index]

    plt.figure(figsize=(max(10, 0.45 * len(low_r2_df)), 6))
    plt.bar(labels, low_r2_df["r2"], edgecolor="black")
    plt.axhline(r2_cutoff, color="red", linestyle="--", linewidth=1)
    plt.ylabel("r2")
    plt.xlabel("Autocorrelation feature")
    plt.title(f"Autocorrelation Features Below {r2_cutoff} r2")
    plt.xticks(rotation=75, ha="right", fontsize=8)
    plt.tight_layout()

    save_path = (
        save_dir / f"autocorrelation_below_{str(r2_cutoff).replace('.', 'p')}_r2.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")
    print(f"Features plotted: {len(low_r2_df)}")

    # 2. Box plot by lag / distance
    lag_df = autocorr_df.copy()
    lag_df["lag"] = [get_lag(idx) for idx in lag_df.index]
    lag_df = lag_df.dropna(subset=["lag"])
    lag_df["lag"] = lag_df["lag"].astype(int)

    print("\nAutocorrelation r2 by lag:")
    print(
        lag_df.groupby("lag")["r2"]
        .agg(["count", "mean", "median", "min", "max"])
        .sort_index()
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=lag_df, x="lag", y="r2", color="steelblue")
    plt.axhline(avg_r2_line, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Autocorrelation lag / distance")
    plt.ylabel("r2")
    plt.title("Autocorrelation Descriptor r2 by Lag")
    plt.tight_layout()

    save_path = save_dir / "autocorrelation_r2_by_lag_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")

    # 3. Box plot by Autocorrelation descriptor family
    family_df = autocorr_df.copy()
    family_df["family"] = [get_family(idx) for idx in family_df.index]

    print("\nAutocorrelation r2 by descriptor family:")
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
    plt.axhline(avg_r2_line, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Autocorrelation descriptor family")
    plt.ylabel("r2")
    plt.title("Autocorrelation Descriptor r2 by Family")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = save_dir / "autocorrelation_r2_by_descriptor_family_boxplot.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved plot to: {save_path}")


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

# run_17 = True

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
        c for c in mordred_df.columns
        if str(c).endswith("_mordred") and "ATS" in str(c)
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
        y_labels = [
            re.sub(r"_rdkit$", "", str(c))
            for c in heatmap_matrix.index
        ]
        x_labels = [
            re.sub(r"_mordred$", "", str(c))
            for c in heatmap_matrix.columns
        ]

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

run_20=True
if run_20:
    path = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv"
    df = pd.read_csv(path)
    
    task_type_means = (
    df
    .groupby("task_type", dropna=False)
    .mean(numeric_only=True)
    .reset_index()
)

    print(task_type_means.to_string(index=False))
    
    
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

run_21 = False

if run_21:
    path = Path("/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/")

    rdkit_df = pd.read_csv(path / "pred_mordred_tr_rdkit/regression_group_perf.csv", index_col=0)
    maccs_df = pd.read_csv(path / "pred_mordred_tr_maccs/regression_group_perf.csv", index_col=0)
    morgan_df = pd.read_csv(path / "pred_mordred_tr_morgan/regression_group_perf.csv", index_col=0)

    # Make sure all three dfs are aligned by the same index
    common_index = rdkit_df.index.intersection(maccs_df.index).intersection(morgan_df.index)

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

    rdkit_minus_maccs_long = (
        rdkit_minus_maccs
        .reset_index()
        .melt(
            id_vars=group_col,
            var_name="metric",
            value_name="normalised_difference"
        )
    )
    rdkit_minus_maccs_long["comparison"] = "RDKit - MACCS"

    rdkit_minus_morgan_long = (
        rdkit_minus_morgan
        .reset_index()
        .melt(
            id_vars=group_col,
            var_name="metric",
            value_name="normalised_difference"
        )
    )
    rdkit_minus_morgan_long["comparison"] = "RDKit - Morgan"

    plot_df = pd.concat(
        [rdkit_minus_maccs_long, rdkit_minus_morgan_long],
        ignore_index=True
    )
    
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=plot_df,
        x=group_col,
        y="normalised_difference",
        hue="comparison",
        style="metric",
        markers=True,
        dashes=False
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
        bbox_inches="tight"
    )
    plt.show()
        