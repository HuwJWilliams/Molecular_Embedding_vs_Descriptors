import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPTS_DIR))
from path.get_paths import getPaths

FILE_DIR = Path(__file__).resolve().parent

DEFAULT_TARGET = {
    "bp":   "Boiling_Point",
    "logd": "LogD",
    "pka":  "pKa",
    "ld50": "LD50",
}

paths = getPaths()

true_paths = paths["targets"]

rf_pred_dirs = paths["prediction_output_dirs"]["rf"]

bp_pred_df_ls = []
logd_pred_df_ls = []
pka_pred_df_ls = []
ld50_pred_df_ls = []


def load_rf_prediction_column(pred_dir, task, feat, default_target_map):
    """
    Load a single RF prediction CSV (.csv.gz) from pred_dir,
    keep ID + target column, rename column with feature suffix.

    Returns
    -------
    pd.DataFrame or None
    """
    files = glob(str(pred_dir) + "/*.csv.gz")
    if not files:
        print(f"No prediction files found in {pred_dir}")
        return None

    targ_column = default_target_map[task]

    df = pd.read_csv(
        files[0],
        usecols=["ID", targ_column],
        index_col="ID"
    )

    df = df.rename(columns={targ_column: f"{targ_column}_{feat}"})
    return df

for task in rf_pred_dirs.keys():
    for feat, pred_dir in rf_pred_dirs[task].items():
        if task == "bp":
            df = load_rf_prediction_column(
                pred_dir=pred_dir,
                task=task,
                feat=feat,
                default_target_map=DEFAULT_TARGET
            )
            bp_pred_df_ls.append(df)


        elif task == "logd":
            df = load_rf_prediction_column(
                pred_dir=pred_dir,
                task=task,
                feat=feat,
                default_target_map=DEFAULT_TARGET
            )
            logd_pred_df_ls.append(df)
        
        elif task == "pka":
            df = load_rf_prediction_column(
                pred_dir=pred_dir,
                task=task,
                feat=feat,
                default_target_map=DEFAULT_TARGET
            )
            pka_pred_df_ls.append(df)

        elif task == "ld50":
            df = load_rf_prediction_column(
                pred_dir=pred_dir,
                task=task,
                feat=feat,
                default_target_map=DEFAULT_TARGET
            )
            ld50_pred_df_ls.append(df)

# Combine predictions (keep only common IDs across features)
full_bp_df   = pd.concat(bp_pred_df_ls, axis=1, join="inner")
full_logd_df = pd.concat(logd_pred_df_ls, axis=1, join="inner")
full_pka_df  = pd.concat(pka_pred_df_ls, axis=1, join="inner")
full_ld50_df = pd.concat(ld50_pred_df_ls, axis=1, join="inner")

# Load true BP
true_bp = (
    pd.read_csv(true_paths["bp"], usecols=["ID", "Boiling_Point"])
      .rename(columns={"Boiling_Point": "True_Boiling_Point"})
      .set_index("ID")
)

true_logd = (
    pd.read_csv(true_paths["logd"], usecols=["ID", "LogD"])
    .rename(columns={"LogD": "True_LogD"})
    .set_index("ID")
)

true_pka = (
    pd.read_csv(true_paths["pka"], usecols=["ID", "pKa"])
    .rename(columns={"pKa": "True_pKa"})
    .set_index("ID")
)

true_ld50 = (
    pd.read_csv(true_paths["ld50"], usecols=["ID", "LD50"])
    .rename(columns={"LD50": "True_LD50"})
    .set_index("ID")
)



# Join predictions with true values (again inner join)
full_bp_df = full_bp_df.join(true_bp, how="inner")
full_logd_df = full_logd_df.join(true_logd, how="inner")
full_pka_df = full_pka_df.join(true_pka, how="inner")
full_ld50_df = full_ld50_df.join(true_ld50, how="inner")


import numpy as np
from scipy.stats import pearsonr

DESCRIPTORS = ["rdkit", "mordred", "chemberta", "molformer"]

def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "RMSD": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
        "SDEP": float(np.std(err, ddof=1)),
        "Pearson_r": float(pearsonr(y_true, y_pred)[0])
    }

task_dfs = {
    "BP": full_bp_df,
    "LogD": full_logd_df,
    "pKa": full_pka_df,
    "LD50": full_ld50_df
}

metrics_by_task = {}

for task, df in task_dfs.items():
    true_col = [c for c in df.columns if c.startswith("True_")][0]
    y_true = df[true_col].values

    pred_cols = [c for c in df.columns if not c.startswith("True_")]

    task_metrics = {}

    for d in DESCRIPTORS:
        col = [c for c in pred_cols if c.endswith(f"_{d}")]
        if len(col) != 1:
            continue

        task_metrics[d] = regression_metrics(
            y_true,
            df[col[0]].values
        )

    metrics_by_task[task] = task_metrics


COLORS = {
    "rdkit": "tab:blue",
    "mordred": "tab:orange",
    "chemberta": "tab:green",
    "molformer": "tab:red"
}

from pathlib import Path
import matplotlib.pyplot as plt

def plot_task_metrics_2x2(
    task,
    metrics_dict,
    out_dir="rf_metric_plots",
    show=True,
    dpi=300
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f"{task} – RF Performance by Descriptor", fontsize=14)

    metric_names = ["RMSD", "Bias", "SDEP", "Pearson_r"]

    for ax, metric in zip(axes.ravel(), metric_names):
        values = [metrics_dict[d][metric] for d in DESCRIPTORS]
        colors = [COLORS[d] for d in DESCRIPTORS]

        ax.bar(DESCRIPTORS, values, color=colors)
        ax.set_title(metric)

        if metric == "Bias":
            ax.axhline(0, linestyle="--", linewidth=1)

        if metric == "Pearson_r":
            ax.set_ylim(0, 1)
            ax.axhline(0, linestyle="--", linewidth=1)


    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ---- Save
    fname = out_dir / f"{task}_rf_descriptor_metrics_2x2.png"
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {fname}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# for task, m in metrics_by_task.items():
#     plot_task_metrics_2x2(task, m, out_dir=FILE_DIR / "visualisation" / "plots")

import json
import pandas as pd
from pathlib import Path

RESULTS_ROOT = Path("/users/yhb18174/TL_project/results")

DESCRIPTORS = ["rdkit", "mordred", "chemberta", "molformer"]
TASKS = ["bp", "logd", "pka", "ld50"]

def detect_descriptor(path_str):
    s = path_str.lower()
    for d in DESCRIPTORS:
        if f"/{d}/" in s:
            return d
    return None

def detect_task(path_str):
    s = path_str.lower()
    for t in TASKS:
        if f"_pred_{t}" in s or f"/{t}/" in s:
            return t
    return None

def detect_kind(fname):
    f = fname.lower()
    if "internal_performance" in f:
        return "internal"
    if "prediction_performance" in f:
        return "prediction"
    return None

rows = []

for json_file in RESULTS_ROOT.rglob("*.json"):
    kind = detect_kind(json_file.name)
    if kind is None:
        continue

    desc = detect_descriptor(str(json_file))
    task = detect_task(str(json_file))

    if desc is None or task is None:
        continue

    with open(json_file) as f:
        data = json.load(f)

    row = {
        "task": task,
        "descriptor": desc,
        "kind": kind,
        "path": str(json_file)
    }

    # Copy all metrics found in JSON
    for k, v in data.items():
        row[k] = v

    rows.append(row)

df = pd.DataFrame(rows)

out_csv = RESULTS_ROOT / "rf_all_performance" / "rf_all_performance_metrics.csv"
df.to_csv(out_csv, index=False)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_ROOT = Path("/users/yhb18174/TL_project/results")
CSV_PATH = out_csv

DESCRIPTORS = ["rdkit", "mordred", "chemberta", "molformer"]
COLORS = {
    "rdkit": "tab:blue",
    "mordred": "tab:orange",
    "chemberta": "tab:green",
    "molformer": "tab:red",
}

# Map task codes in CSV -> plot titles
TASK_LABEL = {"bp": "BP", "logd": "LogD", "pka": "pKa", "ld50": "LD50"}

# Your JSONs use RMSE. We'll plot RMSE (you can rename label to RMSD if you want).
METRICS_2X2 = ["RMSE", "Bias", "SDEP", "Pearson_r"]

def plot_task_metrics_2x2_from_csv(
    df: pd.DataFrame,
    task: str,                  # "bp"/"logd"/"pka"/"ld50"
    kind: str,                  # "internal" or "prediction"
    out_dir: Path,
    show: bool = False,
    dpi: int = 300,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dft = df[(df["task"] == task) & (df["kind"] == kind)].copy()
    if dft.empty:
        print(f"[SKIP] No rows for task={task}, kind={kind}")
        return

    # Ensure we have one row per descriptor
    dft = dft.set_index("descriptor").reindex(DESCRIPTORS)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f"{TASK_LABEL.get(task, task)} – {kind} performance (by descriptor)", fontsize=14)

    for ax, metric in zip(axes.ravel(), METRICS_2X2):
        vals = dft[metric].astype(float).values
        ax.bar(DESCRIPTORS, vals, color=[COLORS[d] for d in DESCRIPTORS])
        ax.set_title(metric)

        if metric in ("Bias", "Pearson_r"):
            ax.axhline(0, linestyle="--", linewidth=1)

        if metric == "Pearson_r":
            ax.set_ylim(0, 1)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = out_dir / f"{TASK_LABEL.get(task, task)}_{kind}_metrics_2x2.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    if show:
        plt.show()


# ---------- RUN ----------
df = pd.read_csv(CSV_PATH)

# Optional sanity check (should be 4 descriptors per task per kind)
print(df.groupby(["task", "kind"]).size())

# Save plots under results/plots_from_csv/<kind>/
for kind in ["internal", "prediction"]:
    for task in ["bp", "logd", "pka", "ld50"]:
        plot_task_metrics_2x2_from_csv(
            df=df,
            task=task,
            kind=kind,
            out_dir=RESULTS_ROOT / "rf_all_performance" / "plots_from_csv" / kind,
            show=False
        )
