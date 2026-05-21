"""
Script to generate features for datasets and saving them to the pathing
"""
# region Script Functionality
# region Imports
import argparse
import pandas as pd
from pathlib import Path
import shutil
import sys
# endregion

# region Path Setup
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "datasets"))
sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SCRIPTS_DIR / "config"))
sys.path.insert(0, str(SRC_DIR / "misc"))

from feature_generator import FeatureGenerator
from pipeline_config import (
    SUPPORTED_FEATURE_SETS,
    TRANSFORMER_FEATURE_SPECS,
    resolve_target_column,
)
from get_paths import getPaths
from misc_fns import splitByMurckoScaffold

# endregion

# region Argument Setup
parser = argparse.ArgumentParser(
    description="Generate descriptors, fingerprints, or embeddings for a target dataset."
)

parser.add_argument(
    "--task",
    required=True,
    help="Dataset key to generate features for, e.g. bp, pka, ld50.",
)

parser.add_argument(
    "--feature-set",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Feature set to generate.",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=100_000,
    help="Number of molecules to process per batch.",
)

parser.add_argument(
    "--paths-json",
    default=None,
    help="Optional pathing JSON. Defaults to the canonical paths.json.",
)

parser.add_argument(
    "--fine-tune",
    action="store_true",
    help="Fine-tune transformer feature generators on the task target before generating features.",
)

parser.add_argument(
    "--target-col",
    default=None,
    help="Target column to use for fine-tuning. Defaults to the configured target for --task.",
)

parser.add_argument(
    "--fine-tune-test-frac",
    type=float,
    default=0.2,
    help="Validation fraction used during transformer fine-tuning.",
)

parser.add_argument(
    "--fine-tune-epochs",
    type=float,
    default=20,
    help="Number of epochs for transformer fine-tuning.",
)

parser.add_argument(
    "--fine-tune-lr",
    type=float,
    default=2e-5,
    help="Learning rate for transformer fine-tuning.",
)

parser.add_argument(
    "--fine-tune-batch-size",
    type=int,
    default=32,
    help="Per-device batch size used during transformer fine-tuning.",
)

parser.add_argument(
    "--max-token-len",
    type=int,
    default=0,
    help="Maximum transformer token length for fine-tuning and embedding generation.",
)

parser.add_argument(
    "--pooling",
    choices=("mean", "cls"),
    default="mean",
    help="Pooling strategy for generated transformer embeddings.",
)

parser.add_argument(
    "--fine-tune-early-stopping-patience",
    type=int,
    default=5,
    help="Stop fine-tuning after this many evals without improvement, use 0 to disable."
)

parser.add_argument(
    "--split-by",
    default="scaffold",
    choices=["scaffold", "random"],
    help="How to split the dataset for fine-tuning"
)

parser.add_argument(
    "--n-mols-fine-tune",
    type=int,
    default=2000,
    help="Number of molecules to fine-tune on."
)

# endregion


# region Resolve Feature Paths Helper
def resolve_feature_paths(paths: dict, task: str, feature_set: str):
    if task not in paths["targets"]:
        available = list(paths["targets"].keys())
        raise KeyError(
            f"Task '{task}' is not available in paths['targets']. "
            f"Available tasks: {available}"
        )

    if task not in paths["full_features"]:
        available = list(paths["full_features"].keys())
        raise KeyError(
            f"Task '{task}' is not available in paths['full_features']. "
            f"Available tasks: {available}"
        )

    if feature_set not in paths["full_features"][task]:
        available = list(paths["full_features"][task].keys())
        raise KeyError(
            f"Feature set '{feature_set}' missing for task '{task}'. "
            f"Available feature sets: {available}"
        )
    return paths["targets"][task], paths["full_features"][task][feature_set]
# endregion
# endregion

# region Running Script
if __name__ == "__main__":
    args = parser.parse_args()

    paths = getPaths(args.paths_json) if args.paths_json else getPaths()

    task = args.task.lower()
    feature_set = args.feature_set.lower()

    if args.max_token_len == 0:
        max_token_len = 202 if "molformer" in feature_set else 512

    in_path, out_path = resolve_feature_paths(
        paths=paths,
        task=task,
        feature_set=feature_set,
    )

    in_df = pd.read_csv(in_path, index_col="ID")

    if "SMILES" not in in_df.columns:
        raise KeyError(f"Input file must contain a 'SMILES' column: {in_path}")

    generator = FeatureGenerator(
        feature_set=feature_set,
        log_name=f"FG_{task}",
    )
    fine_tune_output_dir = None

    if args.fine_tune:
        pathing_key = f"ft-{feature_set}"
        out_path = paths["full_features"][task][pathing_key]
        fine_tune_output_dir = Path(str(out_path).replace("*", "fine_tuned_model")).with_suffix("")


        if feature_set not in TRANSFORMER_FEATURE_SPECS:
            raise ValueError("--fine-tune is only supported for transformer feature sets.")

        target_col = resolve_target_column(task, args.target_col)
        if target_col not in in_df.columns:
            raise KeyError(f"Fine-tuning target column '{target_col}' not found in: {in_path}")

        fine_tune_df = in_df[["SMILES", target_col]].dropna().copy()

        if args.split_by == "random":
            fine_tune_df = fine_tune_df.sample(
                n=min(2000, len(fine_tune_df)),
            )
        else:
            fine_tune_df = splitByMurckoScaffold(fine_tune_df, n_cmpds=args.n_mols_fine_tune, max_scaff_diversity_pass=5)

        spec = TRANSFORMER_FEATURE_SPECS[feature_set]
        result = generator.fineTuneTransformer(
            smiles_ls=fine_tune_df["SMILES"].to_list(),
            ids=fine_tune_df.index.to_list(),
            targets=fine_tune_df[target_col].astype(float).to_list(),
            model_label=spec["model_label"],
            batch_size=args.fine_tune_batch_size,
            max_token_len=max_token_len,
            pooling=args.pooling,
            output_dir=fine_tune_output_dir,
            test_frac=args.fine_tune_test_frac,
            early_stopping_patience=args.fine_tune_early_stopping_patience,
            training_kwargs={
                "num_train_epochs": args.fine_tune_epochs,
                "learning_rate": args.fine_tune_lr,
                "weight_decay": 0.01,
                "eval_strategy": "epoch",
                "save_strategy": "epoch",
                "logging_strategy": "steps",
                "logging_steps": 25,
                "load_best_model_at_end": True,
                "metric_for_best_model": "mse",
                "greater_is_better": False,
                "report_to": "none",
                "save_total_limit": 2
            },
        )

        in_df = in_df.loc[~in_df.index.isin(fine_tune_df.index)]


    generator.calcBatchFeatures(
        smiles_ls=in_df["SMILES"].to_list(),
        id_ls=in_df.index.to_list(),
        fpath=out_path,
        batch_size=args.batch_size,
        max_token_len=max_token_len,
        pooling=args.pooling,
    )

    if fine_tune_output_dir is not None:
        for checkpoint_dir in fine_tune_output_dir.glob("checkpoint-*"):
            if checkpoint_dir.is_dir():
                shutil.rmtree(checkpoint_dir)
# endregion
