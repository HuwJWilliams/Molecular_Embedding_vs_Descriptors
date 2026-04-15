# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
# import shap
# import matplotlib.pyplot as plt
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

exp_keys = ["pred_rdkit_tr_chemberta", 
            "pred_rdkit_tr_chembertasey", 
            "pred_rdkit_tr_molformer", 
            "pred_rdkit_tr_molformer-c3-1b",
            "pred_rdkit_tr_selformer"]

loaded_df_ls = [pd.read_csv(p, index_col=0)[["Pearson_r"]] for p in avg_results_df_paths]

merged = pd.concat(loaded_df_ls, axis=1)
merged.columns = [f"Pearson_r_{i+1}" for i in range(len(loaded_df_ls))]
merged["Pearson_r_avg"] = merged.mean(axis=1)

group_map = getGroups("rdkit")

group_performance_df = v.computeGroupPerf(
    data=merged,
    descriptor_groups=group_map,
    metrics=["Pearson_r_avg"],
    exclude=[]
)

gr_title=f"RDKit Prediction (Average Embedding Performance): Pearson R"
timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")

gr_fname_suffix = "excl_low_var" 

# v.plotGroupRadar(
#     group_performance_df,
#     title=gr_title,
#     save_plot=True,
#     save_path=paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions",
#     save_fname=f"avg_embedding_group_radar",
#     metadata={
#         "Title": gr_title,
#     }
# )

merged.to_csv(paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions" / "avg_embedding_group_radar.csv")

# for group_name, group_members in group_map.items():
#     present_members = [m for m in group_members if m in merged.index]
#     if not present_members:
#         print(f"Skipping group '{group_name}': no members found in performance index.")
#         continue

#     mb_title = "Performance for RDKit (Average Embedding Performance): Pearson R"

#     v.plotMemberBar(
#         perf_df=merged,
#         group_map=group_map,
#         group_name=group_name,
#         value_col="Pearson_r_avg",
#         save_plot=True,
#         save_path=paths["imp_dirs"]["results_dir"] / "lipinski_embeddings_and_descriptor_predictions",
#         save_fname=f"avg_embedding_{group_name}_pred_rdkit_bar",
#         metadata={"Title": mb_title},
#     )


v.plotAveragePerformance(
    exp_keys=exp_keys
)