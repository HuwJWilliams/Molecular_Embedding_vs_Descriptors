"""
Code to run clustering using descriptors, embeddings and fingerprints.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt


FILE_PATH = Path(__file__).resolve()
SRC_DIR = FILE_PATH.parents[1] / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths
PATHS=getPaths()

feature_sets = PATHS["full_features"]["fit_lipinski"]
feats = ["rdkit", "molformer", "maccs"]

feat_df_ls = [pd.read_csv(feature_sets[ft], index_col=0) for ft in feats]
common_ids = feat_df_ls[0].index
for df in feat_df_ls[1:]:
    common_ids = common_ids.intersection(df.index)

feat_df_ls = [df.loc[common_ids] for df in feat_df_ls]
combined_df = pd.concat(feat_df_ls, axis=1)

# numeric only + drop bad cols
X = combined_df.select_dtypes(include="number")
X = X.dropna(axis=1)

# scale then PCA
X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    index=common_ids,
    columns=[f"PC{i+1}" for i in range(X_pca.shape[1])]
)

hbd = HDBSCAN(copy=True, min_cluster_size=100)
cluster_labels = hbd.fit_predict(pca_df)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False, sharey=False)

for ax, ft, feat_df in zip(axes, feats, feat_df_ls):
    X_ft = feat_df.select_dtypes(include="number").dropna(axis=1)

    X_ft_scaled = StandardScaler().fit_transform(X_ft)

    X_ft_pca = PCA(n_components=2, random_state=42).fit_transform(X_ft_scaled)

    ft_pca_df = pd.DataFrame(
        X_ft_pca,
        index=feat_df.index,
        columns=["PC1", "PC2"]
    )

    ft_pca_df["cluster"] = cluster_labels

    noise = ft_pca_df["cluster"] == -1

    ax.scatter(
        ft_pca_df.loc[noise, "PC1"],
        ft_pca_df.loc[noise, "PC2"],
        c="lightgrey",
        s=18,
        alpha=0.5,
        label="Noise"
    )

    clustered = ft_pca_df[~noise]

    scatter = ax.scatter(
        clustered["PC1"],
        clustered["PC2"],
        c=clustered["cluster"],
        cmap="tab20",
        s=22,
        alpha=0.85
    )

    ax.set_title(ft)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

fig.colorbar(scatter, ax=axes, label="Cluster")
plt.tight_layout()
plt.save_fig("/users/yhb18174/TL_project/scripts/visualisation/hdbscan_test.png")