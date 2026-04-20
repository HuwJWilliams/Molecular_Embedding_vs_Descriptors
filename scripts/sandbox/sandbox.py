# =============================================================================
# ChemBERTa Importance Analysis
# Goal: Identify which ChemBERTa embeddings are unused/redundant via SHAP/RF
# =============================================================================

# region Imports
from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.lines import Line2D

import sys
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/misc/")


# from get_paths import getPaths
# from group_descriptors import getGroups
# from vis import Visualise
# from misc_fns import getMostImportantFeatures

# v=Visualise(save_all=False)
# SANDBOX = Path("/users/yhb18174/TL_project/scripts/sandbox")
# endregion



# # region Main
# def main():
#     paths=getPaths()
#     exp_paths = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]

#     # --- Load data ---
#     pred_rdkit_tr_maccs_df = pd.read_csv(exp_paths["pred_rdkit_tr_maccs"] / "pred_rdkit_tr_maccs.csv", index_col=0)
#     pred_rdkit_tr_chemberta_df = pd.read_csv(exp_paths["pred_rdkit_tr_chemberta"]  / "pred_rdkit_tr_chemberta.csv" , index_col=0)
#     pred_maccs_tr_chemberta_df = pd.read_csv(exp_paths["pred_maccs_tr_chemberta"] / "pred_maccs_tr_chemberta.csv", index_col=0)

#     shap_bundle = joblib.load(exp_paths["pred_rdkit_tr_maccs"] / "shap" / "full_shap_analysis.joblib.gz")
#     rf_fi_df = pd.read_csv(exp_paths["pred_rdkit_tr_maccs"] / "all_feature_importance.csv", index_col=0)

#     # --- Descriptor group mapping ---
#     group_map = getGroups("rdkit")
#     desc_to_group = {desc: group for group, members in group_map.items() for desc in members}

#     # --- SHAP importance ---
#     shap_importance_source = {
#         "shap_by_desc": shap_bundle["shap_by_desc"],
#         "feature_names": shap_bundle["feat_explain"].columns.tolist(),
#     }

#     shap_avg_imp, shap_cum_imp = getMostImportantFeatures(shap_importance_source, mode="shap")
#     rf_avg_imp, rf_cum_imp =getMostImportantFeatures(rf_fi_df, mode="rf")

#     shap_avg_df = v.plotDescPredictionVsFeatPrediction(
#         importance_map=shap_avg_imp,
#         pred_tr_perf_df=pred_maccs_tr_chemberta_df,
#         pred_on_target_df=pred_rdkit_tr_chemberta_df,
#         desc_to_group=desc_to_group,
#         importance_col="avg_imp_top25_maccs",
#         left_title="SHAP avg importance: Colored by RDKit group",
#         right_title="SHAP avg importance: Colored by importance",
#         save_path=SANDBOX / "chemberta_vs_shap_imp_maccs_by_group_avg.png",
#         mode="avg",
#         r_vmax=1,
#         r_vmin=0,
#         r_high=5,
#         imp_type="SHAP"
#     )

#     # --- RF importance ---
#     shap_avg_df = v.plotDescPredictionVsFeatPrediction(
#         importance_map=rf_avg_imp,
#         pred_tr_perf_df=pred_maccs_tr_chemberta_df,
#         pred_on_target_df=pred_rdkit_tr_chemberta_df,
#         desc_to_group=desc_to_group,
#         importance_col="avg_imp_top25_maccs",
#         left_title="RF avg importance: Colored by RDKit group",
#         right_title="RF avg importance: Colored by importance",
#         save_path=SANDBOX / "chemberta_vs_rf_imp_maccs_by_group_avg.png",
#         mode="avg",
#         r_vmax=1,
#         r_vmin=0,
#         r_high=5,
#         imp_type="RF"
#     )
 

#     # align on shared descriptor index
#     plot_df = pred_rdkit_tr_maccs_df[["Pearson_r"]].rename(columns={"Pearson_r": "maccs_on_rdkit"}).join(
#         pred_rdkit_tr_chemberta_df[["Pearson_r"]].rename(columns={"Pearson_r": "chemberta_on_rdkit"}),
#         how="inner"
#     ).dropna()

#     plt.figure(figsize=(7, 7))
#     plt.scatter(
#         plot_df["maccs_on_rdkit"],      # x
#         plot_df["chemberta_on_rdkit"],  # y
#         s=25,
#         alpha=0.7,
#         edgecolor="none",
#     )

#     plt.plot([0, 1], [0, 1], "k--", lw=1)
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.xlabel("MACCS -> RDKit Pearson_r")
#     plt.ylabel("ChemBERTa -> RDKit Pearson_r")
#     plt.title("Descriptor-level performance: MACCS vs ChemBERTa")
#     plt.tight_layout()
#     plt.savefig("/users/yhb18174/TL_project/scripts/sandbox/maccs_vs_chemberta_rdkit_scatter.png", dpi=300)
#     plt.close()

# if __name__ == "__main__":
#     main()
# endregion

from pathlib import Path
import joblib

p = Path("/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_rdkit_tr_molformer-c3-1b/shap/full_shap_analysis.joblib.gz")

bundle = joblib.load(p)
shap_by_desc = bundle["shap_by_desc"]

bundle["shap_by_desc"] = {
    k.removesuffix("_model"): v
    for k, v in shap_by_desc.items()
}

# optional backup
p_backup = p.with_suffix(p.suffix + ".bak")
joblib.dump(bundle, p_backup, compress=3)

# overwrite original
joblib.dump(bundle, p, compress=3)
print(f"Updated keys and saved: {p}")
