from __future__ import annotations

"""Script to train a single Random Forest Regressor and save outputs"""

# region Script Functionality
# region Imports
import argparse
import json
import pandas as pd
from pathlib import Path
import sys
# endregion


# region Path Setup
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SRC_DIR / "models"))
sys.path.insert(0, str(SRC_DIR / "misc"))
sys.path.insert(0, str(SCRIPTS_DIR / "config"))
sys.path.insert(0, str(SRC_DIR / "datasets"))

from get_paths import getPaths
from transfer_model import TL
from misc_fns import loadData
from pipeline_config import SUPPORTED_FEATURE_SETS, resolve_target_column
from feature_cleaning import clean_feature_df

data_paths = getPaths()
#endregion

# region Argument Parsing
p = argparse.ArgumentParser()
p.add_argument("--predict-on", required=True, choices=list(data_paths["targets"].keys()))
p.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
p.add_argument("--lipinski", action="store_true", help="Flag to ensure mols fit 'relaxed' lipinski criteria")
p.add_argument("--repeats", type=int, default=1, help="Number of independent RF repeats to run")
p.add_argument("--base-seed", type=int, default=42, help="Base random seed used to seed each repeat")
p.add_argument("--n-resamples", type=int, default=50, help="Number of internal resamples per repeat")
# endregion

# region Running Script
if __name__ == "__main__":
    args = p.parse_args()
    task = args.predict_on
    feat = args.feature_set
    target_col = resolve_target_column(task, args.target_column)
    identifier = args.identifier or f"{task}_{feat}"

    in_features = data_paths["full_features"][task][feat]
    in_targets  = data_paths["targets"][task]
    out_dir     = data_paths["prediction_output_dirs"]['rf'][task][feat]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[RF] task={task}  feature={feat}")
    print(f"Features: {in_features}")
    print(f"Targets : {in_targets}")
    print(f"Output  : {out_dir}")
    print(f"Target column: {target_col}")

    X = loadData(in_features, index_col="ID", wildcard="*")

    if args.lipinski:
        lipinski_ids = set(
            pd.read_csv(
                data_paths["full_features"]["fit_lipinski"]["rdkit"],
                index_col=0,
            ).index
        )

        X = X.loc[X.index.intersection(lipinski_ids)]

    # Drop metadata and any descriptor-generation error strings before RF fitting.
    X, clean_report = clean_feature_df(
        X,
        max_nan_fraction=0.10,
        drop_constant_cols=True,
        median_impute=True,
        correlation_threshold=None,
    )

    print("Feature cleaning report:")
    print(f"  Dropped metadata cols: {len(clean_report['dropped_metadata'])}")
    print(f"  Dropped >10% NaN cols: {len(clean_report.get('dropped_high_nan_cols', []))}")
    print(f"  Dropped constant cols: {len(clean_report['dropped_constant_cols'])}")
    print(f"  Median-imputed cols: {len(clean_report['median_imputed_cols'])}")
    print(f"  Dropped correlated cols: {len(clean_report['dropped_correlated_cols'])}")

    y_df = pd.read_csv(in_targets, index_col="ID")
    if target_col not in y_df.columns:
        raise SystemExit(f"Target column '{target_col}' not found in {in_targets}. "
                        f"Available: {list(y_df.columns)}")

    y = y_df[[target_col]]

    # Align indexes
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    print(f"Aligned shapes → X: {X.shape}, y: {y.shape}")
    #endregion

    #region --- Train single-target RF via TL wrapper
    data = X.join(y)
    print(f"Full modelling data: {data.shape}")

    repeat_perf_records = []
    repeat_internal_records = []
    repeat_pred_series = []

    for repeat_i in range(args.repeats):
        repeat_n = repeat_i + 1
        repeat_seed = args.base_seed + repeat_i
        repeat_dir = out_dir / "repeats" / f"repeat_{repeat_n:03d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nRepeat {repeat_n}/{args.repeats} | seed={repeat_seed}")

        shuffled_data = data.sample(frac=1.0, random_state=repeat_seed)
        cut = int(0.8 * len(shuffled_data))
        train_data = shuffled_data.iloc[:cut].copy()
        test_data = shuffled_data.iloc[cut:].copy()

        split_dir = repeat_dir / "training_data"
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": train_data.index}).to_csv(split_dir / "split_train_ids.csv", index=False)
        pd.DataFrame({"ID": test_data.index}).to_csv(split_dir / "split_validation_ids.csv", index=False)

        print(f"Train: {train_data.shape}  Test: {test_data.shape}")

        model = TL(log_identifier=f"{identifier}_repeat_{repeat_n:03d}")
        final_model, _, internal_perf, _ = model.trainSingleTargetRFModel(
            data=train_data,
            target_column=target_col,
            hyper_params={
                "n_estimators": [200, 400, 500],
                "max_features": ["sqrt"],
                "max_depth": [25, 5, 75, 100],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 4, 8],
            },
            n_resamples=args.n_resamples,
            test_size=0.3,
            random_seed=repeat_seed,
            save_models=True,
            save_path=repeat_dir,
        )

        _, _, external_perf = model.predictSingleTargetRF(
            model=final_model,
            data=test_data,
            target_column=target_col,
            calc_perf=True,
            save_preds=True,
            save_path=repeat_dir,
            preds_filename="last_20pct_pred",
            perf_filename="last_20_pct_perf",
        )

        repeat_perf = {
            "repeat": repeat_n,
            "seed": repeat_seed,
            **external_perf,
        }
        repeat_perf_records.append(repeat_perf)

        repeat_internal_records.append({
            "repeat": repeat_n,
            "seed": repeat_seed,
            **internal_perf,
        })

        pred_name = f"repeat_{repeat_n:03d}"
        saved_pred_df = pd.read_csv(repeat_dir / "last_20pct_pred.csv.gz", index_col="ID")
        repeat_pred_series.append(saved_pred_df[target_col].rename(pred_name))

    perf_df = pd.DataFrame(repeat_perf_records).set_index("repeat")
    perf_df.to_csv(out_dir / "repeat_external_performance.csv")

    internal_perf_df = pd.DataFrame(repeat_internal_records).set_index("repeat")
    internal_perf_df.to_csv(out_dir / "repeat_internal_performance.csv")

    numeric_perf = perf_df.select_dtypes(include="number").drop(columns=["seed"], errors="ignore")
    perf_mean = numeric_perf.mean(axis=0).to_dict()
    perf_std = numeric_perf.std(axis=0).to_dict()

    average_perf = {
        metric: round(float(value), 6)
        for metric, value in perf_mean.items()
    }
    performance_summary = {
        "n_repeats": args.repeats,
        "base_seed": args.base_seed,
        "mean": average_perf,
        "std": {
            metric: round(float(value), 6)
            for metric, value in perf_std.items()
        },
    }

    with open(out_dir / "repeat_external_performance_summary.json", "w") as f:
        json.dump(performance_summary, f, indent=4)

    with open(out_dir / "last_20_pct_perf.json", "w") as f:
        json.dump(average_perf, f, indent=4)

    if repeat_pred_series:
        pred_df = pd.concat(repeat_pred_series, axis=1)
        pred_df[target_col] = pred_df.mean(axis=1)
        pred_df["n_repeats_predicted"] = pred_df[[
            f"repeat_{repeat_i + 1:03d}" for repeat_i in range(args.repeats)
        ]].notna().sum(axis=1)
        pred_df.to_csv(out_dir / "last_20pct_pred_all_repeats.csv.gz", index_label="ID", compression="gzip")
        pred_df[[target_col, "n_repeats_predicted"]].to_csv(
            out_dir / "last_20pct_pred.csv.gz",
            index_label="ID",
            compression="gzip",
        )
