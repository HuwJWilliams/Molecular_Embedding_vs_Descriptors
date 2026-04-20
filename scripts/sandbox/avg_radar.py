# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib
from glob import glob
from datetime import datetime

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
from group_descriptors import getGroups
# from analyse_datasets import plotDescriptorAnalysis

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
from vis import Visualise


paths=getPaths()
v = Visualise()


#endregion
results_dir = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
avg_results_df_paths = [
    results_dir["pred_rdkit_tr_chemberta"] / "pred_rdkit_tr_chemberta.csv",
    results_dir["pred_rdkit_tr_chembertasey"] / "pred_rdkit_tr_chembertasey.csv",
    results_dir["pred_rdkit_tr_molformer"] / "pred_rdkit_tr_molformer.csv",
    results_dir["pred_rdkit_tr_molformer-c3-1b"] / "pred_rdkit_tr_molformer-c3-1b.csv",
    results_dir["pred_rdkit_tr_selformer"] / "pred_rdkit_tr_selformer.csv",
]

other_results_df_paths = [
    results_dir["pred_rdkit_tr_maccs"] / "pred_rdkit_tr_maccs.csv",
    results_dir["pred_rdkit_tr_morgan"] / "pred_rdkit_tr_morgan.csv",
    results_dir["pred_rdkit_tr_mordred"] / "pred_rdkit_tr_mordred.csv",

]

exp_keys = [
    "pred_rdkit_tr_chemberta",
    "pred_rdkit_tr_chembertasey",
    "pred_rdkit_tr_molformer",
    "pred_rdkit_tr_molformer-c3-1b",
    "pred_rdkit_tr_selformer",
    "pred_rdkit_tr_maccs",
    "pred_rdkit_tr_morgan",
    "pred_rdkit_tr_mordred"
]

all_paths = avg_results_df_paths + other_results_df_paths

loaded = []
for p in all_paths:
    # infer label from filename: pred_rdkit_tr_<train>.csv
    train_name = p.stem.replace("pred_rdkit_tr_", "")
    df = pd.read_csv(p, index_col=0)[["Pearson_r"]].rename(columns={"Pearson_r": train_name})
    loaded.append(df)

merged = pd.concat(loaded, axis=1)

# Only embedding columns for the average
embedding_cols = [
    "chemberta",
    "chembertasey",
    "molformer",
    "molformer-c3-1b",
    "selformer",
]

merged["Avg_Pearson_R_Embeddings"] = merged[embedding_cols].mean(axis=1)


group_map = getGroups("rdkit")

# metrics = merged.columns.tolist()
metrics = ["Avg_Pearson_R_Embeddings"]

group_performance_df = v.computeGroupPerf(
    data=merged[["Avg_Pearson_R_Embeddings"]],
    descriptor_groups=group_map,
    metrics=metrics,
    exclude=[]
)


gr_title=f"RDKit Prediction (Average Embedding Performance): Pearson R"
timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

gr_fname_suffix = "excl_low_var" 

# merged.to_csv(paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions" / "pred_rdkit_tr_avg_emb" / "avg_embedding_group_radar.csv")

# v.plotAveragePerformance(
#     exp_keys=exp_keys,
#     results_dir="lipinski_cross_feature_predictions",
#     pred_set="rdkit",
#     save_path=paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions" / "pred_rdkit_tr_avg_emb",
#     save_fname="avg_embedding_grouped_bar_pearson_r",
# )

v.plotGroupRadar(
    group_performance_df,
    title=gr_title,
    save_plot=True,
    save_path=paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions" / "pred_rdkit_tr_avg_emb",
    save_fname=f"avg_emb_group_radar",
    metadata={
        "Title": gr_title,
    }
)