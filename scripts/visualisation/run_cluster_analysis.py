"""
Code to run clustering using descriptors, embeddings and fingerprints.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


FILE_PATH = Path(__file__).resolve()
SRC_DIR = FILE_PATH.parents[1] / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths
PATHS=getPaths()

feature_sets = PATHS["full_features"]["fit_lipinski"]
feats = ["rdkit", "molformer", "maccs"]
n_clusters = 10
plot_data_path = FILE_PATH.parent / "kmeans_per_feature_plot_data.csv"
plot_path = FILE_PATH.parent / "kmeans_per_feature_test.png"

if plot_data_path.exists():
    plot_df = pd.read_csv(plot_data_path, index_col=0)
else:
    feat_df_ls = [pd.read_csv(feature_sets[ft], index_col=0) for ft in feats]
    common_ids = feat_df_ls[0].index
    for df in feat_df_ls[1:]:
        common_ids = common_ids.intersection(df.index)

    feat_df_ls = [df.loc[common_ids] for df in feat_df_ls]

    plot_dfs = []
    for ft, feat_df in zip(feats, feat_df_ls):
        X_ft = feat_df.select_dtypes(include="number").dropna(axis=1)
        X_ft_scaled = StandardScaler().fit_transform(X_ft)
        pca_ft = PCA(n_components=0.95, random_state=42)
        X_ft_pca = pca_ft.fit_transform(X_ft_scaled)

        pca_df = pd.DataFrame(
            X_ft_pca,
            index=feat_df.index,
            columns=[f"PC{i+1}" for i in range(X_ft_pca.shape[1])]
        )

        kmeans = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=10,
            random_state=42
        )
        cluster_labels = kmeans.fit_predict(pca_df)

        ft_pca_df = pd.DataFrame(
            X_ft_pca[:, :2],
            index=feat_df.index,
            columns=["PC1", "PC2"]
        )
        ft_pca_df["feature_set"] = ft
        ft_pca_df["cluster"] = cluster_labels
        ft_pca_df["n_pca_components"] = X_ft_pca.shape[1]
        ft_pca_df["explained_variance"] = pca_ft.explained_variance_ratio_.sum()
        plot_dfs.append(ft_pca_df)

    plot_df = pd.concat(plot_dfs)
    plot_df.to_csv(plot_data_path)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

pc1_min, pc1_max = plot_df["PC1"].min(), plot_df["PC1"].max()
pc2_min, pc2_max = plot_df["PC2"].min(), plot_df["PC2"].max()
pc1_pad = (pc1_max - pc1_min) * 0.05
pc2_pad = (pc2_max - pc2_min) * 0.05
xlim = (pc1_min - pc1_pad, pc1_max + pc1_pad)
ylim = (pc2_min - pc2_pad, pc2_max + pc2_pad)

scatter = None
for ax, ft in zip(axes, feats):
    ft_pca_df = plot_df[plot_df["feature_set"] == ft]

    scatter = ax.scatter(
        ft_pca_df["PC1"],
        ft_pca_df["PC2"],
        c=ft_pca_df["cluster"],
        cmap="tab20",
        s=22,
        alpha=0.85
    )

    ax.set_title(ft)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    single_fig, single_ax = plt.subplots(figsize=(7, 5))
    single_scatter = single_ax.scatter(
        ft_pca_df["PC1"],
        ft_pca_df["PC2"],
        c=ft_pca_df["cluster"],
        cmap="tab20",
        s=22,
        alpha=0.85
    )

    single_ax.set_title(ft)
    single_ax.set_xlabel("PC1")
    single_ax.set_ylabel("PC2")
    single_ax.set_xlim(xlim)
    single_ax.set_ylim(ylim)

    single_fig.colorbar(single_scatter, ax=single_ax, label="Cluster")

    single_fig.tight_layout()
    single_fig.savefig(FILE_PATH.parent / f"kmeans_per_feature_{ft}.png")
    plt.close(single_fig)

if scatter is not None:
    fig.colorbar(scatter, ax=axes, label="Cluster")
plt.tight_layout()
plt.savefig(plot_path)
