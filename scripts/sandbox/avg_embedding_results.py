import argparse
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#region Imports & Pathing
FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent.parent
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import getLowVarianceColumns
from group_descriptors import getGroups

paths = getPaths()
#endregion

#region Set Values
TASK_METRICS = {
    "regression": ["Pearson_r", "r2", "RMSE", "Bias", "MSE", "SDEP"],
    "binary_classification": [
        "Accuracy",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "AUC",
        "MCC",
        "Balanced_Accuracy",
    ],
    "multiclass_classification": [
        "Accuracy",
        "Balanced_Accuracy",
        "F1_macro",
        "AUC_OVR",
        "MCC",
    ],
}

PLOT_METRICS = {
    "regression": "r2",
    "binary_classification": "Balanced_Accuracy",
    "multiclass_classification": "Balanced_Accuracy"
    }
#endregion

#region Function Definitions
def removeDescriptorSuffix(desc: str)-> str:
    desc = str(desc)
    for suffix in ("_mordred", "_rdkit"):
        desc = desc.replace(suffix, "")
    return(desc)

def loadAllModelsResults(pred: str, model_ls: list[str], result_dir: dict | Path):
    loaded = {}
    
    for model in model_ls:
        try:
            exp = f"pred_{pred}_tr_{model}"
            csv_path = Path(result_dir[exp]) / f"{exp}.csv"
            
            loaded[model] = pd.read_csv(csv_path, index_col=0)
        except KeyError:
            print(f"Model results not found under {Path(result_dir[exp]) / f"{exp}.csv"}")
            continue
    
    if not loaded:
        raise ValueError("No model result CSVs were loaded")
        
    return loaded

def averageModelResults(model_results: dict[str:pd.DataFrame]) -> pd.DataFrame:
    metric_cols = [
        "Bias", "SDEP", "MSE", "RMSE", "r2",
        "Pearson_r", "Pearson_p",
        "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV",
        "AUC", "MCC",
        "Balanced_Accuracy", "F1_macro", "AUC_OVR",
    ]

    string_cols = ["task_type"]
    
    combined = pd.concat(
        model_results,
        names=["model", "descriptor"],
        axis=0,
    )
    
    average_results = combined.groupby(level=["model", "descriptor"]).agg(
        {
            **{col: "mean" for col in metric_cols if col in combined.columns},
            **{col: lambda x: ", ".join(sorted(set(x.dropna().astype(str)))) for col in string_cols if col in combined.columns},
        }
    )
    
    return average_results.reset_index()

def separateByTask(df:pd.DataFrame):
    df_dict = {}
    for task_name, metrics in TASK_METRICS.items():
        task_df = df.loc[df["task_type"] == task_name].copy()
        task_df = task_df[metrics]
        task_df = pd.to_numeric(task_df, errors="coerce")
        df_dict[task_name] = task_df
    return df_dict
    
def groupDescriptors(
        df:pd.DataFrame,
        group_map: dict[str, list[str]],
        exclude_cols: list[str] | None = None,
        ) -> pd.DataFrame:
    
    exclude_cols = set(exclude_cols or [])
    rows = []

    for group_name, members in group_map.items():
        present_members = [
            m for m in members
            if m in df.index and m not in exclude_cols
            ]
        if not present_members:
            continue
        

#endregion

#region Setting up Arguments
def main() -> None:
    parser = argparse.ArgumentParser(
        description=""
        )
    
    parser.add_argument(
        "--pred",
        help="")
    
    parser.add_argument(
        "--results-dir",
        default="lipinski_cross_feature_predictions",
        help=""
        )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "chemberta",
            "molformer",
            "molformer-c3-1b",
            "selformer",
            "chembertasey",
        ],
        help="Transformer encoder model names to average."
        )
    
    parser.add_argument(
        "--out-dir",
        default=str(paths["lipinski_cross_feature_predictions"]["pred_mordred_tr_avg"]),
        help=""
        )
    
    parser.add_argument(
        "--exclude-low-var",
        action="store_true",
        help="Exclude low-variance descriptors from group averages and plots.",
    )
    parser.add_argument(
        "--var-threshold",
        type=float,
        default=0.8,
        help="Low-variance threshold passed to getLowVarianceColumns.",
    )
    
    args = parser.parse_args()
    
    results_dir = paths["prediction_output_dirs"][args.results_dir]
   
    exclude_cols = []
    if args.exclude_low_var:
        pred_ft_df = Path(paths["full_features"]["all"][args.pred])
        exclude_cols = getLowVarianceColumns(
            pred_ft_df,
            threshold=args.var_threshold,
        )
   
    
    
#region Running the Code

























