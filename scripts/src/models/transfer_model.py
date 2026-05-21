#%%
from pathlib import Path
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import joblib
import json
import torch
import os


FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
DATASET_DIR = PROJ_DIR / "datasets"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
PALMERCHEM_SOFTWARE_ANALYSIS = Path.home() / "PalmerChem_Software" / "src" / "analysis"
LOG_DIR = SRC_DIR / "models" / "logs"

sys.path.insert(0, str(SRC_DIR / "models"))
from RF_models import RFRegressor, RFClassifier, RFMultiClassifier

sys.path.insert(0, str(PALMERCHEM_SOFTWARE_ANALYSIS))
from performance_calculation import calculatePerformance

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData

sys.path.insert(0, str(SRC_DIR / "models"))
#from mlp_model import MLPRegressorTrainer, RegressionMLP

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import trimRowsByPercentile

# ========== Constants ========== #
BATCH_SIZE      = 64
SEED            = 42
LOG_LEVEL       = logging.DEBUG

# ============ Class ============ #
class TL():
    def __init__(
            self,
            unembedded_df: pd.DataFrame=None,
            embedded_df: pd.DataFrame=None,
            seed: int=None,
            log_name: str = "TLModel",
            log_to_file: bool=False,
            log_dir: Path = LOG_DIR,
            log_level= LOG_LEVEL,
            log_identifier: str = ""
            ):
        
        """
        Initialise the transfer learning class

        Parameters
        ----------
        unembedded_df:  pd.DataFrame

        embedded_df:    pd.DataFrame

        seed:   int

        log_name:   str (optional)

        log_to_file:    bool (optional)

        log_dir:    Path, str (optional)

        log_level:

        log_identifier:     str (optional)


        """
        
        self.unembedded_df = unembedded_df
        self.embedded_df = embedded_df
        self.train_df = None
        self.test_df = None
        self.seed=seed
        self.instantiated_model = None

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        #===== Logger =====#
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(funcName)s | %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler
            if log_to_file:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{log_name}_{log_identifier}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self.logger.info("Logger initialised.")

    
# --- Model Training
# --- --- Random Forests
    def trainMultiTargetRFModels(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        rf_regressor_class=None,
        hyper_params: dict =  {
                "n_estimators": [200, 400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
        output_csv: str = "total_performance.csv",
        existing_performance_csv: str = None,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int | None = None,
        skip_existing: bool = True,
        save_models: bool = True,
        save_path: str = "./",
        log_level=logging.DEBUG,
        trim_by_percentile: bool=True,
        percentile: float=0.99,
        save_feat_imp:bool=False,
        min_training_samples: int = 2500,
    ) -> pd.DataFrame:
        """
        Train Random Forest models for multiple target columns using logging.
        """

        # Set up empty dataframe to save performances
        total_performance_df = pd.DataFrame()
        regressor_cls = rf_regressor_class if rf_regressor_class is not None else RFRegressor

        completed_targets =  set()

        self.logger.debug(f"Existing performance CSV path: {existing_performance_csv}")

        # Load existing performance CSV
        if existing_performance_csv and Path(existing_performance_csv).exists():
            try:
                # Reading in existing performance CSV
                total_performance_df = loadData(existing_performance_csv, index_col=0)
                completed_targets = {str(c).strip() for c in total_performance_df.index}                
                self.logger.info(f"Loaded existing performance data from {existing_performance_csv}")
            except Exception as e:
                self.logger.warning(f"Could not load existing performance CSV: {e}")

        # Align indices
        common_indices = features_df.index.intersection(targets_df.index)
        if len(common_indices) == 0:
            self.logger.error("No common indices found between features_df and targets_df")
            raise ValueError("No common indices found between features_df and targets_df")

        features_df = features_df.loc[common_indices]
        self.logger.debug(f"Features DF: {features_df}")

        targets_df = targets_df.loc[common_indices]
        self.logger.debug(f"Targets DF: {features_df}")

        self.logger.info(f"Training models for {len(targets_df.columns)} target columns")
        self.logger.info(f"Using {len(common_indices)} samples with {len(features_df.columns)} features")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"

        # Loop through each target
        for i, target_column in enumerate(targets_df.columns):
            self.logger.debug(f"Predicting target_column...")
            target_column = str(target_column).strip()

            if target_column.upper() == "SMILES":
                self.logger.info(f"Skipping {target_column}: SMILES column detected")
                continue

            self.logger.info(f"Processing target: {target_column} ({i+1}/{len(targets_df.columns)})")
            current_target = targets_df[target_column]

            target_df = current_target.to_frame(name=target_column)

            if trim_by_percentile:
                trimmed_target_df, removed_rows_df = trimRowsByPercentile(
                    input_df=target_df,
                    columns=[target_column],
                    percentile=percentile,
                    tail="upper",
                    return_removed_rows=True,
                )

                common_idx = features_df.index.intersection(trimmed_target_df.index)

                combined_data = pd.concat(
                    [features_df.loc[common_idx], trimmed_target_df.loc[common_idx, target_column]],
                    axis=1,
                )

            else:
                combined_data = pd.concat([features_df, current_target], axis=1, join="inner")

            # Pull out the target as a Series
            y = combined_data[target_column]

            # Drop rows where target is missing
            combined_data = combined_data.loc[y.notna()].copy()
            if len(combined_data) < min_training_samples:
                self.logger.warning(
                    f"Skipping {target_column}: only {len(combined_data)} samples "
                    f"after filtering; minimum is {min_training_samples}"
                )
                continue
            
            target_series = combined_data[target_column]
            n_unique = int(target_series.nunique(dropna=True))
            non_na_target = target_series.dropna()

            self.logger.info(f"  After removing non-finite values: {len(combined_data)} samples")
            self.logger.info(f"  Target unique values: {n_unique}")

            if n_unique < 2:
                self.logger.warning(f"  Skipping {target_column}: fewer than 2 unique target values")
                continue

            # Class-like targets:
            # - object/category/bool labels
            # - numeric but all values are integer-like
            is_class_like = False
            if (
                pd.api.types.is_object_dtype(non_na_target)
                or pd.api.types.is_categorical_dtype(non_na_target)
                or pd.api.types.is_bool_dtype(non_na_target)
            ):
                is_class_like = True
            elif pd.api.types.is_numeric_dtype(non_na_target):
                arr = non_na_target.to_numpy(dtype=float)
                is_class_like = bool(np.all(np.isclose(arr, np.round(arr), atol=1e-9)))

            # Ensure sklearn classifiers see integer class labels for numeric class-like targets.
            if is_class_like and pd.api.types.is_numeric_dtype(non_na_target):
                combined_data[target_column] = (
                    pd.to_numeric(combined_data[target_column], errors="coerce")
                    .round()
                    .astype("Int64")
                )
                combined_data = combined_data.loc[combined_data[target_column].notna()].copy()
                combined_data[target_column] = combined_data[target_column].astype(int)

            if is_class_like and n_unique == 2:
                expected_task_type = "binary_classification"
            elif is_class_like and n_unique <= 6:
                expected_task_type = "multiclass_classification"
            else:
                expected_task_type = "regression"

            if skip_existing and target_column in completed_targets:
                previous_task_type = None
                if "task_type" in total_performance_df.columns:
                    previous_task_type = total_performance_df.loc[target_column, "task_type"]

                if previous_task_type == expected_task_type:
                    self.logger.info(
                        f"Skipping {target_column} ({i+1}/{len(targets_df.columns)})... "
                        f"already processed as {previous_task_type}"
                    )
                    continue

                self.logger.info(
                    f"Re-running {target_column} ({i+1}/{len(targets_df.columns)}): "
                    f"previous task_type={previous_task_type}, expected task_type={expected_task_type}"
                )

            print(target_column)
            print("n_unique:", n_unique)
            print("unique values:", sorted(target_series.dropna().unique()))
            print("is_class_like:", is_class_like)

            # If binary use classifier
            if is_class_like and n_unique == 2:
                print("Number of unique values = 2. Using RFClassifier")
                try:
                    rf_model = RFClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "roc_auc"},                
                        log_level=log_level,
                        random_seed=random_seed
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFClassifier(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        hyperparameters=hyper_params,
                        test_size=test_size,
                        save_interval_models=False,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        batch_size=batch_size,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model"
                        )
                    performance_dict["task_type"] = "binary_classification"
                    
                except Exception as e:
                    self.logger.error(f"  Error training model for {target_column}: {str(e)}")
                    continue

            # If discrete use multiclass
            elif is_class_like and n_unique <= 6:
                print("Number of unique values 2 < n <=6 Using RFMultiClassifier")

                try:
                    rf_model = RFMultiClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "f1_macro"},                
                        log_level=log_level,
                        random_seed=random_seed
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFMultiClassifier(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        hyperparameters=hyper_params,
                        test_size=test_size,
                        save_interval_models=False,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        batch_size=batch_size,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model"
                        )
                    performance_dict["task_type"] = "multiclass_classification"
                    
                except Exception as e:
                    self.logger.error(f"  Error training model for {target_column}: {str(e)}")
                    continue

            else:
                print("Number of unique values > 6 Using RFRegressor")

                try:
                    rf_model = regressor_cls(
                        cv_function=KFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
                        log_level=log_level,
                        random_seed=random_seed
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFRegressor(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        hyperparameters=hyper_params,
                        test_size=test_size,
                        save_interval_models=False,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        batch_size=batch_size,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model"
                        )
                    performance_dict["task_type"] = "regression"

                except Exception as e:
                    self.logger.error(f"  Error training model for {target_column}: {str(e)}")
                    continue

            # Maintain one cumulative feature-importance CSV:
            # Feature, Importance_<target1>, Importance_<target2>, ...
            if save_feat_imp and feat_importance_df is not None and not feat_importance_df.empty:
                safe_target_column = str(target_column).replace("/", "_").replace("\\", "_")
                fi_df = feat_importance_df.copy()
                if "Feature" not in fi_df.columns:
                    fi_df = fi_df.reset_index().rename(columns={"index": "Feature"})

                if "Importance" in fi_df.columns:
                    imp_col = f"Importance_{safe_target_column}"
                    fi_df = fi_df[["Feature", "Importance"]].rename(columns={"Importance": imp_col})
                    fi_df = fi_df.drop_duplicates(subset=["Feature"], keep="first")

                    if combined_feat_importance_csv.exists():
                        combined_df = pd.read_csv(combined_feat_importance_csv)
                        if "Feature" not in combined_df.columns:
                            combined_df = combined_df.reset_index().rename(columns={"index": "Feature"})
                        combined_df = combined_df.drop_duplicates(subset=["Feature"], keep="first")
                        combined_df = combined_df.merge(fi_df, on="Feature", how="outer")
                    else:
                        combined_df = fi_df

                    combined_df.to_csv(combined_feat_importance_csv, index=False)
                else:
                    self.logger.warning(
                        f"  Feature importance for {target_column} has no 'Importance' column; "
                        "skipping cumulative feature-importance update."
                    )
            elif save_feat_imp:
                self.logger.warning(f"  No feature importance dataframe returned for {target_column}")

            # Convert dict → DataFrame
            perf_df = pd.DataFrame([performance_dict], index=[target_column])

            # Replace existing row for this target (if any), keep mixed-schema columns
            total_performance_df = total_performance_df.drop(index=[target_column], errors="ignore")
            total_performance_df = pd.concat([total_performance_df, perf_df], axis=0, sort=False)

            # Write full table so mixed regression/classification columns are preserved consistently
            total_performance_df.to_csv(output_csv)

            self.logger.info(
                f"  Completed {target_column} ({performance_dict.get('task_type', 'unknown')}) → saved to {output_csv}"
            )

        self.logger.info(f"Completed training for {len(total_performance_df)} targets")
        self.logger.info(f"Results saved to: {output_csv}")

        return total_performance_df

#region Hide code
    def trainWithinFeatureSetRFModels(
        self,
        data_df: pd.DataFrame,
        rf_regressor_class,
        output_csv: str = "within_set_performance.csv",
        hyper_params: dict =  {
            "n_estimators": [200, 400, 500],
            "max_features": ["sqrt"],
            "max_depth": [25, 50, 75, 100],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2, 4, 8],
        },
        save_path: str = "./",
        skip_existing: bool = True,
        save_models: bool = True,
        save_feat_imp: bool = False,
        min_training_samples: int = 2500,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        random_seed: int | None = None,
        trim_by_percentile: bool = True,
        percentile: float = 0.99,
        exclude_same_group: bool = False,
        group_map: dict | None = None,
    ):
        total_performance_df = pd.DataFrame()
        completed_targets = set()

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"

        if output_csv.exists():
            try:
                prev = pd.read_csv(output_csv, index_col=0)
                completed_targets = {str(i).strip() for i in prev.index}
                total_performance_df = prev.copy()
            except Exception:
                pass

        data_df = data_df.copy().apply(pd.to_numeric, errors="coerce")

        # Optional group map
        desc_to_group = {}
        if exclude_same_group:
            if group_map is None:
                raise ValueError(
                    "exclude_same_group=True requires a valid group_map dictionary."
                )
            desc_to_group = {d: g for g, members in group_map.items() for d in members}

        for i, target_column in enumerate(data_df.columns):
            target_column = str(target_column).strip()

            if skip_existing and target_column in completed_targets:
                self.logger.info(f"Skipping {target_column} ({i+1}/{len(data_df.columns)})... already processed")
                continue

            y = data_df[target_column]

            # Start from all other descriptors
            feature_cols = [c for c in data_df.columns if c != target_column]

            # Remove same-group descriptors if requested
            if exclude_same_group:
                tgt_group = desc_to_group.get(target_column)
                if tgt_group is not None:
                    feature_cols = [
                        c for c in feature_cols
                        if desc_to_group.get(c) != tgt_group
                    ]

            if len(feature_cols) == 0:
                self.logger.warning(f"Skipping {target_column}: no predictors left after group exclusion")
                continue

            X = data_df[feature_cols]
            target_df = y.to_frame(name=target_column)

            if trim_by_percentile:
                trimmed_target_df, _removed_rows_df = trimRowsByPercentile(
                    input_df=target_df,
                    columns=[target_column],
                    percentile=percentile,
                    tail="upper",
                    return_removed_rows=True,
                )

                common_idx = X.index.intersection(trimmed_target_df.index)
                combined_data = pd.concat(
                    [X.loc[common_idx], trimmed_target_df.loc[common_idx, target_column]],
                    axis=1,
                )
            else:
                combined_data = pd.concat([X, y], axis=1, join="inner")

            # Pull out the target as a Series
            y = combined_data[target_column]
            # Drop rows where target is missing
            combined_data = combined_data.loc[combined_data[target_column].notna()].copy()
            target_series = combined_data[target_column]
            n_unique = int(target_series.nunique(dropna=True))
            non_na_target = target_series.dropna()
            self.logger.info(f"  After removing non-finite values: {len(combined_data)} samples")
            self.logger.info(f"  Target unique values: {n_unique}")

            if n_unique < 2:
                self.logger.warning(f"  Skipping {target_column}: fewer than 2 unique target values")
                continue

            # Class-like targets:
            # - object/category/bool labels
            # - numeric but all values are integer-like
            is_class_like = False
            if (
                pd.api.types.is_object_dtype(non_na_target)
                or pd.api.types.is_categorical_dtype(non_na_target)
                or pd.api.types.is_bool_dtype(non_na_target)
            ):
                is_class_like = True
            elif pd.api.types.is_numeric_dtype(non_na_target):
                arr = non_na_target.to_numpy(dtype=float)
                is_class_like = bool(np.all(np.isclose(arr, np.round(arr), atol=1e-9)))

            # Ensure sklearn classifiers see integer class labels for numeric class-like targets.
            if is_class_like and pd.api.types.is_numeric_dtype(non_na_target):
                combined_data[target_column] = (
                    pd.to_numeric(combined_data[target_column], errors="coerce")
                    .round()
                    .astype("Int64")
                )
                combined_data = combined_data.loc[combined_data[target_column].notna()].copy()
                combined_data[target_column] = combined_data[target_column].astype(int)

            try:
                if is_class_like and n_unique == 2:
                    rf_model = RFClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "roc_auc"},
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFClassifier(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        test_size=test_size,
                        save_interval_models=False,
                        hyperparameters=hyper_params,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model",
                    )
                    performance_dict["task_type"] = "binary_classification"

                elif is_class_like and n_unique <= 6:
                    rf_model = RFMultiClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "f1_macro"},
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFMultiClassifier(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        test_size=test_size,
                        save_interval_models=False,
                        hyperparameters=hyper_params,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model",
                    )
                    performance_dict["task_type"] = "multiclass_classification"

                else:
                    rf_model = rf_regressor_class(
                        cv_function=KFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
                        hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = rf_model.trainRFRegressor(
                        n_resamples=n_resamples,
                        data=combined_data,
                        target_column=target_column,
                        test_size=test_size,
                        save_interval_models=False,
                        hyperparameters=hyper_params,
                        save_path=save_path,
                        save_final_model=save_models,
                        plot_feat_importance=False,
                        n_jobs=os.cpu_count(),
                        final_rf_seed=None,
                        final_model_name=f"{target_column}_model",
                    )
                    performance_dict["task_type"] = "regression"

                perf_df = pd.DataFrame([performance_dict], index=[target_column])
                total_performance_df = total_performance_df.drop(index=[target_column], errors="ignore")
                total_performance_df = pd.concat([total_performance_df, perf_df], axis=0, sort=False)

                # Maintain one cumulative feature-importance CSV:
                # Feature, Importance_<target1>, Importance_<target2>, ...
                if save_feat_imp and feat_importance_df is not None and not feat_importance_df.empty:
                    safe_target_column = str(target_column).replace("/", "_").replace("\\", "_")
                    fi_df = feat_importance_df.copy()
                    if "Feature" not in fi_df.columns:
                        fi_df = fi_df.reset_index().rename(columns={"index": "Feature"})

                    if "Importance" in fi_df.columns:
                        imp_col = f"Importance_{safe_target_column}"
                        fi_df = fi_df[["Feature", "Importance"]].rename(columns={"Importance": imp_col})
                        fi_df = fi_df.drop_duplicates(subset=["Feature"], keep="first")

                        if combined_feat_importance_csv.exists():
                            combined_df = pd.read_csv(combined_feat_importance_csv)
                            if "Feature" not in combined_df.columns:
                                combined_df = combined_df.reset_index().rename(columns={"index": "Feature"})
                            combined_df = combined_df.drop_duplicates(subset=["Feature"], keep="first")
                            combined_df = combined_df.merge(fi_df, on="Feature", how="outer")
                        else:
                            combined_df = fi_df

                        combined_df.to_csv(combined_feat_importance_csv, index=False)
                    else:
                        self.logger.warning(
                            f"  Feature importance for {target_column} has no 'Importance' column; "
                            "skipping cumulative feature-importance update."
                        )
                elif save_feat_imp:
                    self.logger.warning(f"  No feature importance dataframe returned for {target_column}")

                total_performance_df.to_csv(output_csv)
                self.logger.info(
                    f"Completed {target_column} ({performance_dict.get('task_type', 'unknown')})"
                )

            except Exception as e:
                self.logger.error(f"Error training model for {target_column}: {e}")
                continue

        return total_performance_df
            
    def trainSingleTargetRFModel(
        self,
        data: pd.DataFrame,
        target_column: str,
        hyper_params: dict =  {
                "n_estimators": [400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 50, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int | None=None,
        save_models: bool = False,
        save_path: str = "./",
        log_level=logging.DEBUG,
        n_jobs:int=os.cpu_count()
    ) :
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._rf_setup(
            random_seed=random_seed,
            cv_splits=cv_splits,
            log_level=log_level,
        )

        final_model, best_params, performance_dict, feat_importance_df = self.instantiated_model.trainRFRegressor(
            n_resamples=n_resamples,
            data=data,
            target_column=target_column,
            hyperparameters=hyper_params,
            test_size=test_size,
            save_interval_models=False,
            save_path=save_path,
            save_final_model=save_models,
            plot_feat_importance=False,
            batch_size=batch_size,
            n_jobs=os.cpu_count(),
            final_rf_seed=None
        )

        joblib.dump(final_model, Path(save_path / f"{target_column}_RF_model.pkl.gz"), compress=("gzip", 3))

        with open(Path(save_path / f"{target_column}_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        with open(Path(save_path / f"{target_column}_internal_performance_dict.json"), "w") as f:
            json.dump(performance_dict, f, indent=4)

        feat_importance_df.to_csv(Path(save_path / f"{target_column}_feature_importance.csv"))

        return final_model, best_params, performance_dict, feat_importance_df

# --- Making Predictions
# --- --- Random Forests
    def predictSingleTargetRF(
            self,
            model,
            data: pd.DataFrame | str | Path,
            target_column: str = "LD50",
            feature_cols: list[str] | None = None,
            calc_perf: bool = False,
            save_preds: bool = False,
            save_path: str | Path = Path(FILE_DIR),
            preds_filename: str = "preds",
            perf_filename: str = "performance"
    ):
        data = loadData(data, index_col="ID") if isinstance(data, (str, Path)) else data.copy()

        if feature_cols is not None:
            feature_data = data[feature_cols].copy()
        else:
            feature_data = data.drop(columns=[target_column, "SMILES"], errors="ignore").copy()

        targets_true = data[[target_column]].copy() if target_column in data.columns else None

        if isinstance(model, (str, Path)):
            model = joblib.load(model)

        if not self.instantiated_model:
            self.instantiated_model = RFRegressor(
                cv_function=None,
                hp_search_function=None,
                cv_kwargs={},
                hp_search_kwargs={}
            )

        targets_pred = self.instantiated_model.predictRFRegressor(
            feature_data=feature_data,
            prediction_col=target_column,
            final_rf=model,
            save_preds=save_preds,
            save_path=save_path,
            filename=preds_filename
        )

        targ_pred, targ_true, perf_dict = None, None, None

        if calc_perf:
            if targets_true is None:
                raise ValueError(
                    f"calc_perf=True but '{target_column}' is not present in the provided data."
                )

            targ_pred = (
                targets_pred[target_column]
                if isinstance(targets_pred, pd.DataFrame) and target_column in targets_pred.columns
                else targets_pred.squeeze()
            )
            targ_true = targets_true[target_column]

            perf_dict = self._calculate_performance(
                pred_targs=targ_pred,
                true_targs=targ_true
            )

            if save_preds:
                with open(Path(save_path) / f"{perf_filename}.json", "w") as f:
                    json.dump(perf_dict, f, indent=4)

        return targ_pred, targ_true, perf_dict

# --- Miscellaneous Functions
    def rng(self) -> int:
        if self.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = self.seed
        
        self.logger.info(f"Random seed: {seed}")
        return seed

    def _rf_setup(
            self,
            random_seed=None,
            cv_splits: int = 5,
            log_level=logging.DEBUG,
        ):
        
        self.instantiated_model = RFRegressor(
            cv_function=KFold,
            hp_search_function=GridSearchCV,
            cv_kwargs={"n_splits": cv_splits, "shuffle": True, "random_state": random_seed},
            hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
            log_level=log_level,
            random_seed=random_seed
        )

        return self.instantiated_model
        
    def _calculate_performance(
        self, 
        pred_targs, 
        true_targs, 
    ):
        """
        Returns
        -------
        Dictionary of performance metrics in this order-
        1. Bias
        2. Standard Error of Potential
        3. Mean Squared Error
        4. Root Mean Squared Error (computed from SDEP and Bias)
        5. Pearson R coefficient
        6. Spearman R coefficient
        7. r2 score
        """

        # Calculate Errors

        true_targs = np.asarray(true_targs, dtype=float)
        pred_targs = np.asarray(pred_targs, dtype=float)

        if true_targs.shape != pred_targs.shape:
            raise ValueError(f"Shape mismatch: true {true_targs.shape} vs pred {pred_targs.shape}")

        errors = true_targs - pred_targs

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (np.mean((true_targs - pred_targs - (np.mean(true_targs - pred_targs))) ** 2)) ** 0.5
        mse = mean_squared_error(true_targs, pred_targs)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_targs, pred_targs)

        # Pearson & Spearman correlations
        try:
            r_pearson, p_pearson = pearsonr(true_targs, pred_targs)
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}\n")
            r_pearson, p_pearson = None, None

        try:
            r_spearman, _ = spearmanr(true_targs, pred_targs)
        except Exception as e:
            self.logger.warning(f"Spearman correlation failed: {e}\n")
            r_spearman = None

        return {
                "bias": bias,
                "sdep": sdep,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "r_pearson": r_pearson,
                "p_pearson": p_pearson,
                "r_spearman": r_spearman,
        }
    
    def _feature_reduction(
            self,
            feature_data,
            index_col: str="ID",
            wildcard:str="*",
            by_feature_correlation: bool=False,
            corr_threshold: float=0.7
    ) -> list[str]:
        
        feature_data = loadData(
            feature_data, index_col=index_col, wildcard=wildcard
            )
        
        if by_feature_correlation:
            # Get absolute correlation matric
            corr_matrix = feature_data.corr().abs()

            upper = corr_matrix.where(np.triu(
                np.ones(corr_matrix.shape), k=1
            ).astype(bool))

            feats_to_drop = [
                column for column in upper.columns if any(upper[column] > corr_threshold)
                ]
        
        return feats_to_drop
    
    def _df_2_tensor(
            self, df: pd.DataFrame, dtype = torch.float32
    ):
        return torch.tensor(df.to_numpy(), dtype=dtype)

#endregion
