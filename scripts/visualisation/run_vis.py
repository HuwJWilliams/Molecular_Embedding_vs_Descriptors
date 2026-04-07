# %% --- Script setup
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------ Pathing ------------------
FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
RESULTS_DIR = PROJ_DIR / "results"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "pathing")) 
from get_paths import getPaths, addNewDatasetPaths, addFeatureSetPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups
from analyse_datasets import getLowVarianceColumns, plotLowVarianceColumns, getOutlierSummary

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import getFeatures

paths = getPaths()

# ------------------- Setup -------------------
vis = Visualise(save_all=False)
print("Visualise module loaded")



# %% ------------------- Multi-task Performance

do_multitask_performance = False
if do_multitask_performance:
    # experiment = "pred_mordred_tr_molformer"
    # mt_perf_path = paths["prediction_output_dirs"]["cross_feature_predictions"][experiment]
    # multitask_performance_df = pd.read_csv(mt_perf_path / f"{experiment}.csv", index_col=0)
    # multitask_performance_df.index.name = "Feature"
    # vis.plotMultiTaskPerformance(multitask_performance_df, x_col="Pearson_r", y_col="Feature")

#%% ------------------- Computing Group Performances
    pred = "rdkit"
    tr = "morgan"
    threshold = 0.7
    experiment = f"pred_{pred}_tr_{tr}"
    mt_perf_path = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"][experiment]
    low_variance_columns = getLowVarianceColumns(paths["full_features"]["all"][pred], threshold=threshold)
    print(low_variance_columns)

   
    plotLowVarianceColumns(
        input_df=paths["full_features"]["all"][pred], 
        threshold=threshold,
        output_path="/users/yhb18174/TL_project/datasets/all/descriptor_analysis/",
        save_name=f"low_variance_features_{pred}")
    
    multitask_performance_df = pd.read_csv(mt_perf_path / f"{experiment}.csv", index_col=0)


    # getOutlierSummary(pd.read_csv(paths["full_features"]["all"][pred], index_col=0))

    group_performance_df = vis.computeGroupPerf(
        data=multitask_performance_df,
        descriptor_groups=getGroups(pred),
        metrics=["Pearson_r", "r2", "RMSE", "Bias"],
        exclude=low_variance_columns
    )



    vis.plotGroupRadar(group_performance_df,
                       title=f"{pred.capitalize()} Prediction ({tr.capitalize()} trained)",
                       save_plot=True,
                       save_path=mt_perf_path,
                       save_fname=f"{experiment}_radar_excl_low_var")


    for group_name in getGroups(pred).keys():
        try:
            vis.plotMemberBar(
                perf_df=multitask_performance_df,
                group_map=getGroups(pred),
                group_name=group_name,
                value_col="Pearson_r",
                save_plot=True,
                save_path=mt_perf_path,
                save_fname=f"{experiment}_{group_name}_{pred}_bar")
        except Exception as e:
            print(e)   
            continue

        vis.plotPoorPredictionFeatureDistribution(
                perf_df=multitask_performance_df,
                full_features=paths["full_features"]["all"][pred],
                group_map=getGroups(pred),
                group_name=group_name,
                value_col="Pearson_r",
                save_plot=True,
                save_path=mt_perf_path,            
        )



# # %% ------------------- Comparing Group Performances
# vis.plotGroupBar(group_performance_df, labels=["chemberta"], save_plot=True, 
#                  save_path=paths["prediction_output_dirs"]["cross_feature_predictions"][experiment],
#                  save_fname="pred_rdkit_tr_molformer_bar")

# %% -------------------  for cross-embedding predictions
# emb_desc_keys = paths["prediction_output_dirs"]["cross_feature_predictions"].keys()

# emb_desc_dfs = []
# for k in emb_desc_keys:
#     df_ = pd.read_csv(paths["prediction_output_dirs"]["cross_feature_predictions"][k], index_col=0)
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
do_pca = True 
if do_pca:
    def center_rows(df):
        return df.sub(df.mean(axis=1), axis=0)

    def scale_rows(df):
        denom = df.abs().max(axis=1).replace(0, 1)
        return df.div(denom, axis=0)

    def center_columns(df):
        return df.sub(df.mean(axis=0), axis=1)

    def scale_columns(df):
        denom = df.abs().max(axis=0).replace(0, 1)
        return df.div(denom, axis=1)


    def filter_molecules_by_mw(
        df: pd.DataFrame,
        min_mw: float | None = None,
        max_mw: float | None = None,
        mw_column_candidates: tuple[str, ...] = ("MolWt_rdkit", "MW_mordred", "MolWt"),
    ) -> pd.DataFrame:
        """Filter a feature dataframe by an existing molecular-weight descriptor column."""

        if min_mw is None and max_mw is None:
            return df

        mw_column = next((col for col in mw_column_candidates if col in df.columns), None)
        if mw_column is None:
            raise ValueError(
                "No molecular-weight column found. "
                f"Tried: {list(mw_column_candidates)}"
            )

        filtered_df = df.copy()

        if min_mw is not None:
            filtered_df = filtered_df[filtered_df[mw_column] >= min_mw]

        if max_mw is not None:
            filtered_df = filtered_df[filtered_df[mw_column] <= max_mw]

        return filtered_df


    def get_ids_in_mw_range(
        df: pd.DataFrame,
        min_mw: float | None = None,
        max_mw: float | None = None,
        mw_column_candidates: tuple[str, ...] = ("MolWt_rdkit", "MW_mordred", "MolWt"),
    ) -> pd.Index:
        """Get the IDs that fall within a molecular-weight range."""

        filtered_df = filter_molecules_by_mw(
            df=df,
            min_mw=min_mw,
            max_mw=max_mw,
            mw_column_candidates=mw_column_candidates,
        )
        return filtered_df.index



    feature_names = ["chemberta", "molformer", "mordred", "rdkit"]
    property_names = ["bp", "logd", "pka", "ld50", "pic50"]

    # Keep transposed workflow available (features as rows, IDs as columns)
    do_pca_transposed = False
    do_full_pca_transposed = False
    if do_pca_transposed:
        experiments = {
            "center_scale_rows_then_center_scale_columns": lambda df: scale_columns(center_columns(scale_rows(center_rows(df)))),
        }
        MIN_MW = None
        MAX_MW = None
        sampled_ids = None

        for experiment_name, transform in experiments.items():
            final_df_t = pd.DataFrame()
            rdkit_df = pd.read_csv(paths["full_features"]["all"]["rdkit"], index_col=0)
            selected_ids = get_ids_in_mw_range(rdkit_df, min_mw=MIN_MW, max_mw=MAX_MW)
            if sampled_ids is None:
                n_ids = min(2000, len(selected_ids))
                sampled_ids = (
                    pd.Index(selected_ids)
                    .to_series()
                    .sample(n=n_ids, random_state=42, replace=False)
                    .index
                )

            for feat in feature_names:
                temp_df = pd.read_csv(paths["full_features"]["all"][feat], index_col=0)
                temp_df = temp_df.loc[temp_df.index.intersection(sampled_ids)].copy()
                temp_df = temp_df.T
                temp_scaled = transform(temp_df)
                temp_scaled["Source"] = feat
                final_df_t = pd.concat([final_df_t, temp_scaled], axis=0)

            print(f"{experiment_name} (with .T) before dropna:", final_df_t.shape)
            final_df_t = final_df_t.dropna(axis=1)
            print(f"{experiment_name} (with .T) after dropna:", final_df_t.shape)

            fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCABiplot(
                data_dict={"Data": final_df_t},
                pc_x=1,
                pc_y=2,
                n_components=5,
                top_n_loadings=20,
                plot_area=False,
                save_plot=True,
                save_path=RESULTS_DIR,
                save_fname=f"inverted_feature_biplot_{experiment_name}",
                axis_fontsize=14,
                label_fontsize=10,
                legend_fontsize=10,
            )

            if do_full_pca_transposed:
                fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCA(
                    data_dict={"Data": final_df_t},
                    n_components=5,
                    plot_area=False,
                    save_plot=True,
                    save_path=RESULTS_DIR,
                    save_fname=f"inverted_feature_pca_{experiment_name}",
                    axis_fontsize=14,
                )

    # Property-wise biplots (no transpose)
    do_property_full_pca = False
    for prop in property_names:
        for feat in feature_names:
            train_path = paths["imp_dirs"]["datasets_dir"] / "training_data" / f"{prop}_model_training.csv"
            val_path = paths["imp_dirs"]["datasets_dir"] / "training_data" / f"{prop}_model_validation.csv"

            train_df = getFeatures(train_path, feature_name=feat)
            val_df = getFeatures(val_path, feature_name=feat)

            # Keep common descriptor columns for train/val before PCA
            common_cols = train_df.columns.intersection(val_df.columns)
            if len(common_cols) == 0:
                print(f"Skipping {prop}/{feat}: no common descriptor columns between train and val.")
                continue

            train_df = train_df[common_cols].copy()
            val_df = val_df[common_cols].copy()

            print(f"{prop} / {feat}: train={train_df.shape}, val={val_df.shape}")

            fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCABiplot(
                data_dict={"train": train_df, "val": val_df},
                pc_x=1,
                pc_y=2,
                n_components=5,
                top_n_loadings=25,
                plot_area=False,
                remove_outliers=False,
                scale=True,
                save_plot=True,
                save_path=RESULTS_DIR / "property_biplots" / prop,
                save_fname=f"{prop}_{feat}_train_val_biplot",
                axis_fontsize=14,
                label_fontsize=10,
                legend_fontsize=10,
            )

            if do_property_full_pca:
                fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCA(
                    data_dict={"train": train_df, "val": val_df},
                    n_components=5,
                    plot_area=False,
                    save_plot=True,
                    save_path=RESULTS_DIR / "property_biplots" / prop,
                    save_fname=f"{prop}_{feat}_train_val_pca",
                    axis_fontsize=14,
                )




# for feat in feature_names:
#     print(feat)
#     for p in prop:
#         print(p)
#         df = pd.read_csv(paths["full_features"]['all'][feat], index_col=0)
#         train = getFeatures(paths["imp_dirs"]["datasets_dir"] / "training_data" / f"{p}_model_training.csv", feature_name=feat)
#         val = getFeatures(paths["imp_dirs"]["datasets_dir"] / "training_data" / f"{p}_model_validation.csv", feature_name=feat)

#         data4 = {
#             "train" : train,
#             "val" : val
#         }

#         for data in [data4]:
#             dataset_name = next(iter(data))
#             fig, pca_df, loadings_df, abs_loadings_df = vis.plotPCA(
#                 data_dict=data,
#                 n_components=5,
#                 plot_area=False,
#                 save_plot=True,
#                 save_path=RESULTS_DIR,
#                 save_fname=f"{p}_tr_val_pca",
#             )

# %% ------------------- Unique counts
# feature_set = "rdkit"
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
# feature_set = "rdkit"
# emb_desc_key = "pred_rdkit_tr_chemberta"

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
