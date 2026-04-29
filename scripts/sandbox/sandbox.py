# =============================================================================
# ChemBERTa Importance Analysis
# Goal: Identify which ChemBERTa embeddings are unused/redundant via SHAP/RF
# =============================================================================

# region Imports
from pathlib import Path

# import joblib
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# from matplotlib.lines import Line2D

import sys
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/misc/")


# from get_paths import getPaths
from group_descriptors import getGroups
# from vis import Visualise
# from misc_fns import getMostImportantFeatures

# v=Visualise(save_all=False)
# SANDBOX = Path("/users/yhb18174/TL_project/scripts/sandbox")
# endregion


 # region Main
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

paths = getPaths()
ft_path_dic = paths["full_features"]["fit_lipinski"]

pred_feat = "rdkit"
tr_feat = "rdkit"

ft_df = pd.read_csv(ft_path_dic[pred_feat], index_col=0)
non_continuous_feats = [col for col in ft_df.columns if ft_df[col].nunique(dropna=True) <= 6]

exp = f"pred_{pred_feat}_tr_{tr_feat}"
result_path = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"][exp]
perf_csv = result_path / f"{exp}.csv"

perf_df = pd.read_csv(perf_csv, index_col=0)
perf_df = perf_df.loc[~perf_df.index.isin(non_continuous_feats)]
perf_df.to_csv(perf_csv, index_label="Features")
print(f"Updated: {perf_csv} -> {perf_df.shape}")



# from rdkit import Chem
# from rdkit.Chem import Draw


# def save_smiles_grid(smiles_list, legend_list, out_path="molecule_grid.png", mols_per_row=4, sub_img_size=(320, 260)):
#     if len(smiles_list) != len(legend_list):
#         raise ValueError("smiles_list and legend_list must be the same length.")

#     mols, legends = [], []
#     for smi, leg in zip(smiles_list, legend_list):
#         mol = Chem.MolFromSmiles(smi)
#         if mol is None:
#             print(f"Skipping invalid SMILES: {smi}")
#             continue
#         mols.append(mol)
#         legends.append(str(leg))

#     if not mols:
#         raise ValueError("No valid SMILES to draw.")

#     img = Draw.MolsToGridImage(
#         mols,
#         legends=legends,
#         molsPerRow=mols_per_row,
#         subImgSize=sub_img_size,
#         useSVG=False
#     )
#     img.save(out_path)





# Full flattened lists (2 molecules per TOP result, in order)

# sim_dict={'feat_sim': 
#  {'rdkit_rbf': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'rdkit_jacc': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'mordred_rbf': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'mordred_jacc': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'maccs_rbf': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'maccs_jacc': {'id_pair': ('bp_4967', 'bp_9379'), 'smi_pair': ('CCCCC(C)CC', 'CCCCC[C@@H](C)CCC')}, 
#   'morgan_rbf': {'id_pair': ('bp_7030', 'bp_807'), 'smi_pair': ('CCCC(=O)OCC(C)C', 'CCOCC(C)C')}, 
#   'morgan_jacc': {'id_pair': ('pic50_1135', 'pic50_1506'), 'smi_pair': ('Cc1ccc(F)cc1-c1ccc2cc(NC(=O)CCN3CCCC3)c3nnc(C)n3c2c1', 'CNC(=O)Cc1ccc(-c2ccc3cc(NC(=O)CCN4CCCC4)c4nnc(C)n4c3c2)cc1')}, ''
#   'chemberta_rbf': {'id_pair': ('bp_165', 'bp_7030'), 'smi_pair': ('CCCC(=O)C(C)C', 'CCCC(=O)OCC(C)C')}, 
#   'chemberta_jacc': {'id_pair': ('bp_165', 'bp_7030'), 'smi_pair': ('CCCC(=O)C(C)C', 'CCCC(=O)OCC(C)C')},
#     'chembertasey_rbf': {'id_pair': ('bp_807', 'bp_3604'), 'smi_pair': ('CCOCC(C)C', 'CSCC(C)C')}, 
#     'chembertasey_jacc': {'id_pair': ('bp_807', 'bp_3604'), 'smi_pair': ('CCOCC(C)C', 'CSCC(C)C')}, 
#     'molformer_rbf': {'id_pair': ('pic50_1135', 'pic50_1506'), 'smi_pair': ('Cc1ccc(F)cc1-c1ccc2cc(NC(=O)CCN3CCCC3)c3nnc(C)n3c2c1', 'CNC(=O)Cc1ccc(-c2ccc3cc(NC(=O)CCN4CCCC4)c4nnc(C)n4c3c2)cc1')}, 
#     'molformer_jacc': {'id_pair': ('pic50_1135', 'pic50_1506'), 'smi_pair': ('Cc1ccc(F)cc1-c1ccc2cc(NC(=O)CCN3CCCC3)c3nnc(C)n3c2c1', 'CNC(=O)Cc1ccc(-c2ccc3cc(NC(=O)CCN4CCCC4)c4nnc(C)n4c3c2)cc1')}, 
#     'molformer-c3-1b_rbf': {'id_pair': ('bp_6062', 'bp_3951'), 'smi_pair': ('CC(O)c1c(F)c(F)c(F)c(F)c1F', 'Fc1c(F)c(F)c(CCl)c(F)c1F')}, 
#     'molformer-c3-1b_jacc': {'id_pair': ('bp_6062', 'bp_3951'), 'smi_pair': ('CC(O)c1c(F)c(F)c(F)c(F)c1F', 'Fc1c(F)c(F)c(CCl)c(F)c1F')}, 
#     'selformer_rbf': {'id_pair': ('logd_3528', 'logd_1594'), 'smi_pair': ('CN[C@H]1CCN(C(=O)c2ccc(Nc3nccc(-c4cnc(C)n4C(C)C)n3)cc2)C1', 'CNC(=O)c1ccc(Nc2nccc(-c3cnc(C)n3C(C)C)n2)cc1')}, 
#     'selformer_jacc': {'id_pair': ('logd_3528', 'logd_1594'), 'smi_pair': ('CN[C@H]1CCN(C(=O)c2ccc(Nc3nccc(-c4cnc(C)n4C(C)C)n3)cc2)C1', 'CNC(=O)c1ccc(Nc2nccc(-c3cnc(C)n3C(C)C)n2)cc1')}}
#     }

# # Example:
# # Build smiles + legend lists directly from sim_dict
# smiles_list = []
# legend_list = []

# for key, val in sim_dict.get("feat_sim", {}).items():
#     # skip malformed/empty keys
#     if not key or not isinstance(val, dict):
#         continue
#     if "id_pair" not in val or "smi_pair" not in val:
#         continue

#     id1, id2 = val["id_pair"]
#     smi1, smi2 = val["smi_pair"]

#     # key like "rdkit_rbf" -> "RDKIT RBF"
#     label = key.replace("_", " ").upper()

#     smiles_list.extend([smi1, smi2])
#     legend_list.extend([
#         f"{id1}\n{label}",
#         f"{id2}\n{label}",
#     ])

# # Draw
# save_smiles_grid(
#     smiles_list,
#     legend_list,
#     out_path="grid.png",
#     mols_per_row=2
# )
