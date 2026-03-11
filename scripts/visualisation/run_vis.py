# %% --- Script setup
import sys
from pathlib import Path
import pandas as pd
from vis import Visualise

# ------------------ Pathing ------------------
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = PROJ_DIR / "results"

EMB_AND_DESC_PREDS = "embeddings_and_descriptor_predictions"
LD50_PREDS_RF = "LD50_predictions_rf"

sys.path.insert(0, str(PROJ_DIR / "scripts" / "path"))
from get_paths_new import getPaths, addNewDatasetPaths, addFeatureSetPaths

sys.path.insert(0, str(PROJ_DIR / "scripts" / 'datasets'))
from group_descriptors import getGroups

paths = getPaths()
addNewDatasetPaths("solubility", "boiling_point.csv", "SOLUBILITY", "solubility")
addFeatureSetPaths("MCCV", "fingerprints")

# ------------------- Setup -------------------
vis = Visualise(save_all=False)
print("Visualise module loaded")



# %% ------------------- Multi-task Performance
# experiment = "pred_rdkit_tr_molformer"
# mt_perf_path = paths["prediction_output_dirs"]["embedding_and_descriptor_cross_predictions"][experiment]
# multitask_performance_df = pd.read_csv(mt_perf_path / f"{experiment}.csv", index_col=0)
# multitask_performance_df.index.name = "Feature"
# vis.plotMultiTaskPerformance(multitask_performance_df, x_col="Pearson_r", y_col="Feature")

# %% ------------------- Computing Group Performances
# experiment = "pred_rdkit_tr_chemberta"
# mt_perf_path = paths["prediction_output_dirs"]["embedding_and_descriptor_cross_predictions"][experiment]
# multitask_performance_df = pd.read_csv(mt_perf_path / f"{experiment}.csv", index_col=0)
# group_performance_df = vis.computeGroupPerf(
#     data=multitask_performance_df,
#     descriptor_groups=getGroups("rdkit"),
#     metrics=["Pearson_r", "r2", "RMSE", "Bias"],
# )


# vis.plotGroupRadar(group_performance_df,
#                 title=f"RDKit Prediction (ChemBERTa trained)",
#                    save_plot=True,
#                    save_path=mt_perf_path,
#                    save_fname=f"{experiment}_radar")

# # %% ------------------- Comparing Group Performances
# vis.plotGroupBar(group_performance_df, labels=["chemberta"], save_plot=True, 
#                  save_path=paths["prediction_output_dirs"]["embedding_and_descriptor_cross_predictions"][experiment],
#                  save_fname="pred_rdkit_tr_molformer_bar")

# %% -------------------  for cross-embedding predictions
# emb_desc_keys = paths["prediction_output_dirs"]["embedding_and_descriptor_predictions"].keys()

# emb_desc_dfs = []
# for k in emb_desc_keys:
#     df_ = pd.read_csv(paths["prediction_output_dirs"]["embedding_and_descriptor_predictions"][k], index_col=0)
#     df_.index.name = "Feature"
#     emb_desc_dfs.append(df_)

# vis.plotBoxPlots(
#     *emb_desc_dfs,
#     trained_labels=["rdkit", "mordred", "rdkit", "mordred", "chemberta", "molformer", "chemberta", "molformer"],
#     predicted_labels=["molformer", "molformer", "chemberta", "chemberta", "mordred", "mordred", "rdkit", "rdkit"],
#     save_plot=False,
#     save_path=RESULTS_DIR / EMB_AND_DESC_PREDS
# )



# %% ------------------- Model performance barplots 

# path_files = paths["prediction_output_dirs"]["rf"]

# prop_ls = ["bp", "logd", "pka", "ld50", "pic50"]
# name_ls = ["Boiling_Point", "LogD", "pKa", "LD50", "pIC50"]


# for prop, name in zip(prop_ls, name_ls):
#     perf_files = {
#         "rdkit": path_files[prop]["rdkit"] / f"{name}_internal_performance_dict.json",
#         "mordred": path_files[prop]["mordred"] / f"{name}_internal_performance_dict.json",
#         "chemberta": path_files[prop]["chemberta"] / f"{name}_internal_performance_dict.json",
#         "molformer": path_files[prop]["molformer"] / f"{name}_internal_performance_dict.json",
#     }


#     vis.plotModelPerformanceBars(
#         base_path=RESULTS_DIR,
#         model_jsons=perf_files,
#         model_labels=list(perf_files.keys()),
#         metrics=["r2", "Pearson_r", "RMSE", "Bias"],
#         save_plot=True,
#         save_path=RESULTS_DIR / "performance_plots",
#         save_fname=f"{prop}_internal_performance"
#     )




# %% ------------------- Feature Importance
# prop_ls = ["bp", "logd", "pka", "ld50", "pic50"]
# name_ls = ["Boiling_Point", "LogD", "pKa", "LD50", "pIC50"]

# desc_ls = ["rdkit", "mordred"]

# for desc in desc_ls:
#     for prop, name in zip(prop_ls, name_ls):
#         feat_imp_df = paths["prediction_output_dirs"]["rf"][prop][desc] / f"{name}_feature_importance.csv"

#         feature_importance_df = pd.read_csv(
#             feat_imp_df, 
#             )
#         print(feature_importance_df.columns)

#         vis.plotFeatureImportance(
#             feature_importance_df, 
#             top_n=20,
#             save_plot=True,
#             save_path=RESULTS_DIR / "feature_importance_plots",
#             save_fname=f"top_20_{prop}_{desc}"
#                                   )



# %% ------------------- PCA of embeddings
# data_dict = {
#     "chemberta" : all_data["raw_features"]["chemberta"]
# }


# fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCA(
#     data_dict=data_dict,
#     n_components=5,
#     plot_area=False,
#     save_plot=False,
#     save_path=RESULTS_DIR,
# )

# %% ------------------- Unique counts
feature_set = "rdkit"
# vis.plotNumUniqueDescValues(data=all_data["raw_features"][feature_set],
#                             save_plot=True,
#                             save_path=PROJ_DIR / "datasets" / "descriptors",
#                             save_fname="rdkit_unique_value_count",
#                             show_x_ticks=True,
#                             x_label="Descriptors",
#                             y_label="Unique Count",
#                             tick_fontsize=8
#                             )

# %%------------------- Perf vs uniqueness
feature_set = "rdkit"
emb_desc_key = "pred_rdkit_tr_chemberta"

# vis.plotNumUniqueDescVsPerf(
#     train_data=all_data["raw_features"][feature_set],
#     perf_data=all_data["embedding_and_descriptor_predictions"][emb_desc_key],
#     desc_set=feature_set,
#     save_plot=True,
#     save_path=RESULTS_DIR / EMB_AND_DESC_PREDS / emb_desc_key,
#     save_fname="nunique_vs_performance_tr_chemberta"
# )

#
# %%

