# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
# import shap
import matplotlib.pyplot as plt
# import joblib
from glob import glob

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
# from group_descriptors import getGroups
# from analyse_datasets import plotDescriptorAnalysis

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
from vis import Visualise


paths=getPaths()
v = Visualise()


#endregion
"""
Experiment description:
to try and find what embeddings are totally unused from chemberta
"""

paths = getPaths()

CFD = paths["prediction_output_dirs"]["lipinski_cross_feature_predictions"]
prtc = CFD["pred_rdkit_tr_chemberta"]  # rdkit predicted from chemberta

results_dir = Path(prtc) / "training_data"
tr_feat_path = results_dir / "training_features.csv.gz"

# SHAP settings
max_bg = 200
max_exp = 500

# run SHAP for every rdkit descriptor model and aggregate per embedding feature
model_paths = sorted(glob(str(results_dir / "*.pkl")))
if not model_paths:
    raise FileNotFoundError(f"No models found in {results_dir}")

all_rows = []

out_csv = Path(prtc) / "shap_feature_scores_for_fi_vs_shap.csv"
if not out_csv.exists():
    for model_path in model_paths:
        pred_feature = Path(model_path).stem  # e.g. MolWt_rdkit

        shap_v, feat_explain, _ = v.shapAnalysis(
            model=model_path,
            features=tr_feat_path,
            pred_feature=pred_feature,
            output_dir=results_dir,  # not used here
            max_bg=max_bg,
            max_explain=max_exp,
            plot=False,
            max_display=20,
        )

        sv_obj = shap_v[0] if isinstance(shap_v, list) else shap_v
        sv = sv_obj.values if hasattr(sv_obj, "values") else np.asarray(sv_obj)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)

        mean_abs = np.abs(sv).mean(axis=0)
        for feat_name, val in zip(feat_explain.columns, mean_abs):
            all_rows.append(
                {
                    "pred_feature": pred_feature,
                    "embedding_feature": feat_name,
                    "mean_abs_shap": float(val),
                }
            )

    shap_df = pd.DataFrame(all_rows)

    # For FI vs SHAP: one SHAP score per embedding across all descriptor targets
    shap_agg = (
        shap_df.groupby("embedding_feature", as_index=False)["mean_abs_shap"]
        .mean()
        .rename(columns={"mean_abs_shap": "shap_score"})
    )

    # save for merge with FI table later
    shap_agg.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

else:
    shap = pd.read_csv(out_csv, index_col=0)
    fi = pd.read_csv(out_csv.parent / "all_feature_importance.csv", index_col=0)
    fi["avg_fi"] = fi.mean(axis=1)

    plt.figure(figsize=(7, 6))
    plt.scatter(shap["shap_score"], fi["avg_fi"], alpha=0.7, s=20)
    plt.xlabel("Average SHAP Score")
    plt.ylabel("Average Feature Importance")
    plt.title("SHAP vs Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.savefig(out_csv.parent / "shap_vs_fi.png", dpi=400)