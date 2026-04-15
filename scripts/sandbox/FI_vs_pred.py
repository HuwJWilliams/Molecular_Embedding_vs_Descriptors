# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import joblib
from glob import glob

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
# from group_descriptors import getGroups
# from analyse_datasets import plotDescriptorAnalysis

paths=getPaths()

#endregion

paths=getPaths()
RD = paths['imp_dirs']['results_dir']
CFD = paths["prediction_output_dirs"]['lipinski_cross_feature_predictions']

pctr = CFD["pred_chemberta_tr_rdkit"]
prtc = CFD["pred_rdkit_tr_chemberta"]

"""
Notes for experiment-
What I want to show is to plot the feature importance each embedding had vs the prediction when training on rdkit.
Showing
'How well was an embedding estimated and how important was it'
What I need is the prediction of each chemberta embedding when trained on RDKit and I need the feature importance of the 


Which chemberta embeddings are important for predicting RDKit, and how well can RDKit predict them?”
"""

pctr_preds = pd.read_csv(pctr / "pred_chemberta_tr_rdkit.csv")
prtc_feat = pd.read_csv(prtc / "all_feature_importance.csv")
import pandas as pd
import matplotlib.pyplot as plt

# Load with first column as index (feature/model name)
pctr_preds = pd.read_csv(pctr / "pred_chemberta_tr_rdkit.csv", index_col=0)
prtc_feat = pd.read_csv(prtc / "all_feature_importance.csv")

# 1) Predictability: Pearson_r for each chemberta embedding predicted from rdkit
predability = pctr_preds[["Pearson_r"]].copy()
predability.index = predability.index.astype(str).str.strip()

# 2) Importance: mean feature importance across all RDKit targets for each embedding row
importance_cols = [c for c in prtc_feat.columns if c.startswith("Importance_")]
imp = prtc_feat[["Feature"] + importance_cols].copy()
imp["Feature"] = imp["Feature"].astype(str).str.strip()
imp["importance_mean"] = imp[importance_cols].mean(axis=1)
imp = imp.set_index("Feature")[["importance_mean"]]

# 3) Merge on embedding name
plot_df = predability.join(imp, how="inner").dropna()
print(plot_df.shape)
print(plot_df.head())

# 4) Plot
plt.figure(figsize=(7, 6))
plt.scatter(plot_df["Pearson_r"], plot_df["importance_mean"], alpha=0.7, s=18)
plt.xlabel("Predictability from RDKit (Pearson_r)")
plt.ylabel("Mean importance for predicting RDKit descriptors")
plt.title("ChemBERTa Embeddings: Predictability vs Importance")
plt.grid(alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("/users/yhb18174/TL_project/scripts/sandbox/test.png")
