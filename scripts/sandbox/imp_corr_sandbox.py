from pathlib import Path
import pandas as pd
import sys
import numpy as np

def featImportanceCorrelationTripartite(
        importance_df,
        tr_df,
        te_df,
        top_imp_feats: int=10,
        corr_threshold: float=0.7,
        save_path: str="/users/yhb18174/TL_project/results/tripartite_importance_corr.png",
):
    
    imp_wide = pd.read_csv(importance_df)
    tr_df = pd.read_csv(tr_df, index_col=0)
    te_df = pd.read_csv(te_df, index_col=0)

    common_ids = tr_df.index.intersection(te_df.index)
    X = tr_df.loc[common_ids].select_dtypes("number")
    Y = te_df.loc[common_ids].select_dtypes("number")

    X = X.loc[:, X.var(axis=0) > 0]
    Y = Y.loc[:, Y.var(axis=0) > 0]

    cross_corr = pd.DataFrame(
        np.corrcoef(X.to_numpy().T, Y.to_numpy().T)[:X.shape[1], X.shape[1]:],
        index=X.columns,
        columns=Y.columns
    )

    top_n_feats = pd.DataFrame()
    top_n_feats.index = te_df.index

    imp_cols = [
        c for c in imp_wide.columns if c.startswith("Importance_")
    ]

    importance_matrix = (
        imp_wide
        .rename(columns={"Feature": "TrainedFeature"})
        .set_index("TrainedFeature")
    )

    # remove prefix so cols match descriptor names
    importance_matrix.columns = [
        c.replace("Importance_", "") if c.startswith("Importance_") else c
        for c in importance_matrix.columns
    ]

    importance_matrix = importance_matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return importance_matrix, cross_corr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot_tripartite_rdkit_maccs(
    cross_corr,              # rows=maccs, cols=rdkit (corr values)
    importance_matrix,       # rows=maccs, cols=rdkit (importance values)
    top_n_importance=10,
    corr_threshold=0.6,
    label_fontsize=9,
    connection_width=1.0,
    save_path="/users/yhb18174/TL_project/results/tripartite_rdkit_maccs.png",
):
    # Ensure same shape/index/columns
    cross_corr = cross_corr.copy()
    importance_matrix = importance_matrix.reindex(
        index=cross_corr.index, columns=cross_corr.columns
    ).fillna(0.0)

    mid_nodes = list(cross_corr.index)      # maccs
    rdkit_nodes = list(cross_corr.columns)  # rdkit

    # --- Build edges: middle -> right (correlation)
    corr_edges = []
    for m in mid_nodes:
        for r in rdkit_nodes:
            c = cross_corr.loc[m, r]
            if np.isfinite(c) and abs(c) >= corr_threshold:
                corr_edges.append((m, r, float(c)))

    # --- Build edges: middle -> left (top-N importance per left descriptor)
    imp_edges = []
    for r in rdkit_nodes:
        s = importance_matrix[r].dropna()
        if s.empty:
            continue
        top_feats = s.sort_values(ascending=False).head(top_n_importance)
        for m, w in top_feats.items():
            if w > 0:
                imp_edges.append((m, r, float(w)))

    # Left and right are both rdkit descriptors
    left_nodes = rdkit_nodes
    right_nodes = rdkit_nodes

    # Positions
    xL, xM, xR = 0.0, 1.0, 2.0
    yL = np.linspace(1.6, -0.6, max(1, len(left_nodes)))
    yM = np.linspace(1.6, -0.6, max(1, len(mid_nodes)))
    yR = np.linspace(1.6, -0.6, max(1, len(right_nodes)))

    left_pos = {n: (xL, yL[i]) for i, n in enumerate(left_nodes)}
    mid_pos = {n: (xM, yM[i]) for i, n in enumerate(mid_nodes)}
    right_pos = {n: (xR, yR[i]) for i, n in enumerate(right_nodes)}

    # Degrees for labels
    left_deg = {n: 0 for n in left_nodes}
    mid_deg_imp = {n: 0 for n in mid_nodes}
    mid_deg_corr = {n: 0 for n in mid_nodes}
    right_deg = {n: 0 for n in right_nodes}

    for m, l, _ in imp_edges:
        if m in mid_deg_imp and l in left_deg:
            mid_deg_imp[m] += 1
            left_deg[l] += 1

    for m, r, _ in corr_edges:
        if m in mid_deg_corr and r in right_deg:
            mid_deg_corr[m] += 1
            right_deg[r] += 1

    # Draw
    fig, ax = plt.subplots(figsize=(30, 22))

    # Nodes
    ax.scatter([xL]*len(left_nodes), [left_pos[n][1] for n in left_nodes], c="black", s=20, zorder=3)
    ax.scatter([xM]*len(mid_nodes), [mid_pos[n][1] for n in mid_nodes], c="black", s=20, zorder=3)
    ax.scatter([xR]*len(right_nodes), [right_pos[n][1] for n in right_nodes], c="black", s=20, zorder=3)

    # Importance edges (middle -> left), orange
    if imp_edges:
        max_imp = max(w for _, _, w in imp_edges) if imp_edges else 1.0
        segs, widths = [], []
        for m, l, w in imp_edges:
            segs.append([mid_pos[m], left_pos[l]])
            widths.append(connection_width * (0.2 + 1.0 * (w / max_imp if max_imp > 0 else 0.0)))
        ax.add_collection(LineCollection(segs, colors="tab:orange", linewidths=widths, alpha=0.45, zorder=1))

    # Correlation edges (middle -> right), red/blue
    if corr_edges:
        segs, cols, widths = [], [], []
        for m, r, c in corr_edges:
            segs.append([mid_pos[m], right_pos[r]])
            cols.append("tab:red" if c > 0 else "tab:blue")
            widths.append(connection_width * (0.15 + 0.9 * abs(c)))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=widths, alpha=0.28, zorder=1))

    # Labels
    for n in left_nodes:
        ax.text(xL - 0.03, left_pos[n][1], f"{n} ({left_deg[n]})", ha="right", va="center", fontsize=label_fontsize)

    for n in mid_nodes:
        ax.text(xM, mid_pos[n][1], f"{n} ({mid_deg_imp[n]}|{mid_deg_corr[n]})",
                ha="center", va="bottom", fontsize=label_fontsize)

    for n in right_nodes:
        ax.text(xR + 0.03, right_pos[n][1], f"{n} ({right_deg[n]})", ha="left", va="center", fontsize=label_fontsize)

    ax.set_xlim(-0.8, 4.0)
    ax.set_ylim(-0.9, 1.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=600)
    plt.close(fig)

    return save_path

def plot_corr_vs_importance_scatter(
    cross_corr,
    importance_matrix,
    save_path="/users/yhb18174/TL_project/results/tripartite_rdkit_maccs.png",
    alpha=0.25,
    point_size=10,
):
    """
    Plot correlation (x) vs feature importance (y) for aligned train-test feature pairs.
    Saves output to the same directory as the tripartite plot.
    """
    cross_corr = cross_corr.copy()
    importance_matrix = importance_matrix.reindex(
        index=cross_corr.index, columns=cross_corr.columns
    )

    pairs = (
        importance_matrix.stack()
        .rename("importance")
        .to_frame()
        .join(cross_corr.stack().rename("corr"), how="inner")
        .reset_index()
        .rename(columns={"level_0": "trained_feature", "level_1": "tested_descriptor"})
    )

    pairs["importance"] = pd.to_numeric(pairs["importance"], errors="coerce")
    pairs["corr"] = pd.to_numeric(pairs["corr"], errors="coerce")
    pairs = pairs.dropna(subset=["importance", "corr"])

    out_path = Path(save_path)
    scatter_path = out_path.with_name(f"{out_path.stem}_corr_vs_importance{out_path.suffix}")

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        pairs["corr"],
        pairs["importance"],
        alpha=alpha,
        s=point_size,
        c="tab:blue",
        edgecolors="none",
    )
    ax.axvline(0, color="grey", lw=1.0, alpha=0.6)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature Importance")
    ax.set_title("Correlation vs Feature Importance")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=400)
    plt.close(fig)

    return str(scatter_path)

    


i, c = featImportanceCorrelationTripartite(
    importance_df="/users/yhb18174/TL_project/results/embeddings_and_descriptor_predictions/pred_rdkit_tr_maccs/all_feature_importance.csv",
    tr_df="/users/yhb18174/TL_project/datasets/all/all_maccs.csv",
    te_df="/users/yhb18174/TL_project/datasets/all/all_rdkit.csv",
)

plot_tripartite_rdkit_maccs(
    cross_corr=c,                 # maccs x rdkit corr matrix
    importance_matrix=i,   # maccs x rdkit importance matrix
    top_n_importance=5,
    corr_threshold=0.8,
)

plot_corr_vs_importance_scatter(
    cross_corr=c,
    importance_matrix=i,
    save_path="/users/yhb18174/TL_project/results/tripartite_rdkit_maccs.png",
)
