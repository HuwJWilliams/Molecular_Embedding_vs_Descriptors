"""
Script to assess the similarity spaces between different molecular representation types
"""

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, RobustScaler
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import KernelPCA
from rdkit import Chem
from rdkit.Chem import Draw

FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR / "src" / "pathing"))
from get_paths import getPaths
paths=getPaths()


sys.path.insert(0, str(SCRIPTS_DIR / "src" / "misc"))
from misc_fns import molid2Smiles

feature_sets=[
    "rdkit", 
    "mordred", 
    "maccs", 
    "morgan", 
    "chemberta", "chembertasey", "molformer", "molformer-c3-1b", "selformer"
    ]
feature_paths = paths['full_features']['fit_lipinski']

def save_smiles_grid(smiles_list, legend_list, out_path="molecule_grid.png", mols_per_row=4, sub_img_size=(320, 260)):
    if len(smiles_list) != len(legend_list):
        raise ValueError("smiles_list and legend_list must be the same length.")

    mols, legends = [], []
    for smi, leg in zip(smiles_list, legend_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Skipping invalid SMILES: {smi}")
            continue
        mols.append(mol)
        legends.append(str(leg))

    if not mols:
        raise ValueError("No valid SMILES to draw.")

    img = Draw.MolsToGridImage(
        mols,
        legends=legends,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        useSVG=False
    )
    img.save(out_path)


def getRBFSimilarity(df):
    df_scaled = StandardScaler().fit_transform(df)
    sq_distance=pairwise_distances(df_scaled, metric="sqeuclidean")
    gamma = 1/np.median(sq_distance[sq_distance > 0])
    sim=np.exp(-gamma * sq_distance)

    # store as dataframe
    rbf_sim_df = pd.DataFrame(
        sim,
        index=df.index,
        columns=df.index
    )
    return rbf_sim_df

def getContinousJaccard(df):

    # scale features
    df_robust = RobustScaler().fit_transform(df)

    df_scaled = df_robust - df_robust.min(axis=0)

    n = df_scaled.shape[0]
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            numerator = np.sum(np.minimum(df_scaled[i], df_scaled[j]))
            denominator = np.sum(np.maximum(df_scaled[i], df_scaled[j]))

            if denominator == 0:
                sim[i, j] = 1.0

            else: 
                sim[i, j] = numerator / denominator

    jacc_sim_df = pd.DataFrame(
        sim,
        index=df.index,
        columns=df.index
    )
    return jacc_sim_df

all_similarity_matrices = {
    "rbf": {},
    "jacc": {}
}

molids = list(pd.read_csv(feature_paths["rdkit"], index_col=0).sample(100).index)

sim_dict = {
    "feat_sim": {
    }
}
# Top-k similar pairs per feature/metric
top_k = 3

for feat in feature_sets:
    feat_df = pd.read_csv(feature_paths[feat], index_col=0).loc[molids]

    all_similarity_matrices["rbf"][feat] = getRBFSimilarity(feat_df)
    all_similarity_matrices["jacc"][feat] = getContinousJaccard(feat_df)
    ids = list(feat_df.index)

    for metric_key in ["rbf", "jacc"]:
        S = all_similarity_matrices[metric_key][feat].to_numpy(dtype=float)
        if S.shape[0] < 2:
            continue

        iu = np.triu_indices(S.shape[0], k=1)
        pair_vals = S[iu]
        if pair_vals.size == 0:
            continue

        # indices of top-k similarities (descending)
        k = min(top_k, pair_vals.size)
        top_idx = np.argsort(pair_vals)[-k:][::-1]

        top_pairs = []
        for rank, idx_flat in enumerate(top_idx, start=1):
            i = int(iu[0][idx_flat])
            j = int(iu[1][idx_flat])
            score = float(pair_vals[idx_flat])

            id1, id2 = ids[i], ids[j]
            smi1, smi2 = molid2Smiles(id1), molid2Smiles(id2)

            print(f"[TOP-{rank} {metric_key.upper()}] {feat}: ({id1}, {id2}) -> {score:.6f}")

            top_pairs.append({
                "rank": rank,
                "score": score,
                "id_pair": (id1, id2),
                "smi_pair": (smi1, smi2),
            })

        sim_dict["feat_sim"][f"{feat}_{metric_key}"] = top_pairs


from pathlib import Path

save_dir = Path("/users/yhb18174/TL_project/scripts/visualisation")
save_dir.mkdir(parents=True, exist_ok=True)

for key, val in sim_dict.get("feat_sim", {}).items():
    print(key)
    print(val)

    # val is now: [ {rank, score, id_pair, smi_pair}, ... ]
    if not key or not isinstance(val, list):
        continue

    label = key.replace("_", " ").upper()
    smiles_list = []
    legend_list = []

    for rec in val:
        if not isinstance(rec, dict):
            continue
        if "id_pair" not in rec or "smi_pair" not in rec:
            continue

        id1, id2 = rec["id_pair"]
        smi1, smi2 = rec["smi_pair"]
        rank = rec.get("rank", "?")
        score = rec.get("score", None)

        smiles_list.extend([smi1, smi2])
        if score is None:
            legend_list.extend([
                f"{id1}\n{label}\nTOP-{rank}",
                f"{id2}\n{label}\nTOP-{rank}",
            ])
        else:
            legend_list.extend([
                f"{id1}\n{label}\nTOP-{rank} ({score:.3f})",
                f"{id2}\n{label}\nTOP-{rank} ({score:.3f})",
            ])

    if not smiles_list:
        continue

    out_path = save_dir / f"{key}_top3.png"
    print(out_path)

    save_smiles_grid(
        smiles_list,
        legend_list,
        out_path=out_path,   # no need to cast to str
        mols_per_row=2
    )
    print(f"Saved: {out_path}")



# assume exactly 2 feature sets
feat_a, feat_b = feature_sets[0], feature_sets[1]

# choose metric to plot
for metric_key, cbar_label in [("rbf", "RBF Similarity"), ("jacc", "Continuous Jaccard")]:
    sim_a = all_similarity_matrices[metric_key][feat_a].to_numpy()
    sim_b = all_similarity_matrices[metric_key][feat_b].to_numpy()

    n = sim_a.shape[0]
    combined = np.zeros((n, n), dtype=float)

    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((n, n), dtype=bool), k=-1)

    combined[upper_mask] = sim_a[upper_mask]   # first feature set -> upper
    combined[lower_mask] = sim_b[lower_mask]   # second feature set -> lower
    combined[np.diag_indices(n)] = 1.0

    idx = all_similarity_matrices[metric_key][feat_a].index
    combined_df = pd.DataFrame(combined, index=idx, columns=idx)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        combined_df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.2,
        cbar_kws={"label": cbar_label},
    )
    plt.title(f"{metric_key.upper()} | Upper: {feat_a} | Lower: {feat_b}")
    plt.xlabel("Molecules")
    plt.ylabel("Molecules")
    plt.tight_layout()
    plt.savefig(f"./test_combined_triangles_{metric_key}.png", dpi=300)
    plt.close()

metric_key = "rbf"

# Keep a common molecule order across all feature sets
common_ids = None
for feat in feature_sets:
    idx = all_similarity_matrices[metric_key][feat].index
    common_ids = idx if common_ids is None else common_ids.intersection(idx)

if common_ids is None or len(common_ids) < 3:
    raise ValueError("Not enough shared molecules across feature sets for KPCA.")

plot_frames = []
for feat in feature_sets:
    S_df = all_similarity_matrices[metric_key][feat].loc[common_ids, common_ids]
    S = S_df.to_numpy()

    coords = KernelPCA(n_components=2, kernel="precomputed").fit_transform(S)
    feat_df = pd.DataFrame(coords, columns=["KPCA1", "KPCA2"], index=common_ids)
    feat_df["Source"] = feat
    plot_frames.append(feat_df.reset_index(names="MolID"))

plot_df = pd.concat(plot_frames, ignore_index=True)

# Fit one KPCA object (same settings) to get eigenvalues for variance-style labels
# Note: for KernelPCA this is kernel-space variance, not classical PCA variance.
kpca_ref = KernelPCA(n_components=2, kernel="precomputed")
kpca_ref.fit(all_similarity_matrices[metric_key][feature_sets[0]].loc[common_ids, common_ids].to_numpy())
eigvals = np.asarray(kpca_ref.eigenvalues_, dtype=float)
var_ratio = eigvals / eigvals.sum() if eigvals.sum() > 0 else np.array([np.nan, np.nan])

xlab = f"KPCA1 ({var_ratio[0]*100:.1f}% var)"
ylab = f"KPCA2 ({var_ratio[1]*100:.1f}% var)"

fig, axes = plt.subplots(
    2, 2, figsize=(11, 8.5), gridspec_kw={"wspace": 0.18, "hspace": 0.18}
)

# (0,0): KDE of KPCA1
sns.kdeplot(
    data=plot_df, x="KPCA1", hue="Source", fill=True, common_norm=False,
    alpha=0.25, ax=axes[0, 0], legend=False
)
axes[0, 0].set_xlabel(xlab)
axes[0, 0].set_ylabel("Density")

# (1,0): scatter KPCA1 vs KPCA2
sns.scatterplot(
    data=plot_df, x="KPCA1", y="KPCA2", hue="Source",
    s=28, alpha=0.75, edgecolor=None, ax=axes[1, 0]
)
axes[1, 0].set_xlabel(xlab)
axes[1, 0].set_ylabel(ylab)

# (1,1): KDE of KPCA2
sns.kdeplot(
    data=plot_df, x="KPCA2", hue="Source", fill=True, common_norm=False,
    alpha=0.25, ax=axes[1, 1], legend=False
)
axes[1, 1].set_xlabel(ylab)
axes[1, 1].set_ylabel("Density")

axes[0, 1].axis("off")

# capture legend handles from scatter and remove axis-level legend
handles, labels = axes[1, 0].get_legend_handles_labels()
leg = axes[1, 0].get_legend()
if leg is not None:
    leg.remove()

# stats for KDE variables (KPCA1, KPCA2)
stats_rows = []
stats_numeric = []
for src, g in plot_df.groupby("Source"):
    kpca1 = g["KPCA1"].dropna().to_numpy()
    kpca2 = g["KPCA2"].dropna().to_numpy()

    q1_1, q3_1 = np.percentile(kpca1, [25, 75]) if len(kpca1) else (np.nan, np.nan)
    q1_2, q3_2 = np.percentile(kpca2, [25, 75]) if len(kpca2) else (np.nan, np.nan)

    std1 = float(np.std(kpca1, ddof=1)) if len(kpca1) > 1 else np.nan
    iqr1 = float(q3_1 - q1_1) if len(kpca1) else np.nan
    std2 = float(np.std(kpca2, ddof=1)) if len(kpca2) > 1 else np.nan
    iqr2 = float(q3_2 - q1_2) if len(kpca2) else np.nan
    stats_numeric.append([src, std1, iqr1, std2, iqr2])
    stats_rows.append([
        src,
        f"{std1:.3f}" if np.isfinite(std1) else "nan",
        f"{iqr1:.3f}" if np.isfinite(iqr1) else "nan",
        f"{std2:.3f}" if np.isfinite(std2) else "nan",
        f"{iqr2:.3f}" if np.isfinite(iqr2) else "nan",
    ])

# optional: sort rows by source name
sort_idx = sorted(range(len(stats_rows)), key=lambda i: str(stats_rows[i][0]))
stats_rows = [stats_rows[i] for i in sort_idx]
stats_numeric = [stats_numeric[i] for i in sort_idx]

col_labels = ["Source", "STD1", "IQR1", "STD2", "IQR2"]
tbl = axes[0, 1].table(
    cellText=stats_rows,
    colLabels=col_labels,
    colWidths=[0.36, 0.16, 0.16, 0.16, 0.16],
    cellLoc="center",
    colLoc="center",
    loc="lower center",
    bbox=[0.02, 0.00, 0.96, 0.78],  # [left, bottom, width, height]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.0, 1.15)

# Highlight best (maximum) value per metric column: STD1, IQR1, STD2, IQR2
metric_vals = np.array([row[1:] for row in stats_numeric], dtype=float)
if metric_vals.size > 0:
    best_vals = np.nanmax(metric_vals, axis=0)
    highlight_colour = "#fff3b0"
    for r_idx in range(metric_vals.shape[0]):
        for c_idx in range(metric_vals.shape[1]):
            val = metric_vals[r_idx, c_idx]
            if np.isfinite(val) and np.isclose(val, best_vals[c_idx], rtol=1e-9, atol=1e-12):
                cell = tbl[(r_idx + 1, c_idx + 1)]  # +1 skips header, +1 skips Source col
                cell.set_facecolor(highlight_colour)
                cell.get_text().set_weight("bold")

axes[0, 1].text(
    0.5, 0.82,
    "KDE Stats\n(STD/IQR for KPCA1 and KPCA2)",
    ha="center", va="bottom", fontsize=9, transform=axes[0, 1].transAxes
)

fig.suptitle(f"KPCA across all feature sets ({metric_key.upper()})", y=0.985, fontsize=14)
if handles:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(len(labels), 5),
        frameon=False,
        title="Source",
        fontsize=9,
        title_fontsize=10,
    )

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.86])
plt.savefig(f"./kpca_{metric_key}_all_feature_sets_panel.png", dpi=300, bbox_inches="tight")
plt.close()
