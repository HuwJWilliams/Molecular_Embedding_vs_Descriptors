"""
Script to train, run, and save predictions using Random Forest Regressor Models
"""

# %% ===== Python Imports =====
import argparse
import json
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# %% ===== Project Imports & Pathing Setup =====
from config import SCRIPTS_DIR, SRC_DIR, PATHING_JSON_PATH, SUPPORTED_FEATURE_SETS

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData

sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_cleaning import cleanFeatureDF

# %% ===== Argument Parsing =====
p = argparse.ArgumentParser()
p.add_argument(
    "--predict-on", required=True, choices=list(FULL_PATHING["targets"].keys())
)
p.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
p.add_argument(
    "--repeats", type=int, default=1, help="Number of independent RF repeats to run"
)
p.add_argument(
    "--base-seed",
    type=int,
    default=42,
    help="Base random seed used to seed each repeat",
)
p.add_argument(
    "--n-resamples",
    type=int,
    default=50,
    help="Number of internal resamples per repeat",
)
p.add_argument(
    "--max_nan_frac",
    type=float,
    default=0,
    help="Fraction of NaN rows to drop a column",
)
p.add_argument(
    "--corr-threshold",
    type=float,
    default=0.9,
    help="Correlation threshold for training features",
)
args = p.parse_args()
task = args.predict_on
feat = args.feature_set
target_col = args.target_column
identifier = args.identifier or f"{task}_{feat}"
max_nan_frac = args.max_nan_frac
corr_threshold = args.corr_threshold
in_features = FULL_PATHING["full_features"][task][feat]
in_targets = FULL_PATHING["targets"][task]
out_dir = FULL_PATHING["prediction_output_dirs"]["rf"][task][feat]
out_dir.mkdir(parents=True, exist_ok=True)

print(f"[RF] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Output  : {out_dir}")
print(f"Target column: {target_col}")

# %% ===== Setting Up Data =====

X = loadData(in_features, index_col="ID", wildcard="*")

X, clean_report = cleanFeatureDF(
    X,
    max_nan_fraction=max_nan_frac,
    drop_constant_cols=True,
    median_impute=False,
    correlation_threshold=corr_threshold,
)

print("Feature cleaning report:")
print(f"  Dropped metadata cols: {len(clean_report['dropped_metadata'])}")
print(
    f"  Dropped >{max_nan_frac * 100}% NaN cols: {len(clean_report.get('dropped_high_nan_cols', []))}"
)
print(f"  Dropped constant cols: {len(clean_report['dropped_constant_cols'])}")
print(f"  Median-imputed cols: {len(clean_report['median_imputed_cols'])}")
print(f"  Dropped correlated cols: {len(clean_report['dropped_correlated_cols'])}")

y_df = pd.read_csv(in_targets, index_col="ID")
if target_col not in y_df.columns:
    raise SystemExit(
        f"Target column '{target_col}' not found in {in_targets}. "
        f"Available: {list(y_df.columns)}"
    )

y = y_df[[target_col]]

# Align Indexes
common = X.indes.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

print(f"Aligned shapes → X: {X.shape}, y: {y.shape}")

data = X.join(y)
print(f"Full modelling data: {data.shape}")

# %% ===== Setting up training resamples =====

repeat_perf_records = []
repeat_internal_records = []
repeat_pred_series = []

for repeat_i in range(args.repeats):
    repeat_n = repeat_i + 1
    repeat_seed = args.base_seed + np.randint(2**32)
    repeat_dir = out_dir / "repeats" / f"repeat_{repeat_n:03d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRepeat {repeat_n}/{args.repeats} | seed={repeat_seed}")

    # Shuffling Data & making TR/TE split
    shuffled_data = data.sample(frac=1.0, random_state=repeat_seed)
    cut = int(0.8 * len(shuffled_data))
    train_data = shuffled_data.iloc[:cut].copy()
    test_data = shuffled_data.iloc[cut:].copy
    print(f"Train: {train_data.shape}  Test: {test_data.shape}")

    # Saving TR/TE Data
    split_dir = repeat_dir / "training_data"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train_data.index}).to_csv(
        split_dir / "split_train_ids.csv", index=False
    )
    pd.DataFrame({"ID": test_data.index}).to_csv(
        split_dir / "split_validation_ids.csv", index=False
    )

    # %% ===== Training Models =====
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
        preds_filename="external_preds",
        perf_filename="external_perf",
    )

    # %% ===== Recordiong and Averaging resample performances =====
    repeat_perf = {"repeat": repeat_n, "seed": repeat_seed, **external_perf}
    repeat_perf_records.append(repeat_perf)

    repeat_internal_records.append(
        {
            "repeat": repeat_n,
            "seed": repeat_seed,
            **internal_perf,
        }
    )

    (repeat_dir / "external_perf.json").unlink(missing_ok=True)
    (repeat_dir / f"{target_col}_internal_performance_dict.json").unlink(
        missing_ok=True
    )
    (repeat_dir / "training_data" / "performance_stats.json").unlink(missing_ok=True)

    pred_name = f"repeat_{repeat_n:03d}"
    saved_pred_df = pd.read_csv(repeat_dir / "external_preds.csv.gz", index_col="ID")
    repeat_pred_series.append(saved_pred_df[target_col].rename(pred_name))

    def summarise_performance(records: list[dict]) -> dict:
        perf_df = pd.DataFrame(records).set_index("repeat")
        numeric_perf = perf_df.select_dtypes(include="number").drop(
            columns=["seed"], errors="ignore"
        )

        return {
            "individual": perf_df.reset_index().to_dict(orient="records"),
            "mean": {
                metric: round(float(value), 6)
                for metric, value in numeric_perf.mean(axis=0).items()
            },
            "std": {
                metric: round(float(value), 6)
                for metric, value in numeric_perf.std(axis=0).items()
            },
        }

    rf_performance = {
        "n_repeats": args.repeats,
        "base_seed": args.base_seed,
        "internal": summarise_performance(repeat_internal_records),
        "external": summarise_performance(repeat_perf_records),
    }

    with open(out_dir / "rf_performance.json", "w") as f:
        json.dump(rf_performance, f, indent=4)

    if repeat_pred_series:
        pred_df = pd.concat(repeat_pred_series, axis=1)
        pred_df[target_col] = pred_df.mean(axis=1)
        pred_df["n_repeats_predicted"] = (
            pred_df[[f"repeat_{repeat_i + 1:03d}" for repeat_i in range(args.repeats)]]
            .notna()
            .sum(axis=1)
        )
        pred_df.to_csv(
            out_dir / "last_20pct_pred_all_repeats.csv.gz",
            index_label="ID",
            compression="gzip",
        )
        pred_df[[target_col, "n_repeats_predicted"]].to_csv(
            out_dir / "last_20pct_pred.csv.gz",
            index_label="ID",
            compression="gzip",
        )
