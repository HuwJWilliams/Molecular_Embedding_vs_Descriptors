# region Imports and Pathing
from pathlib import Path
import pandas as pd
import sys
import numpy as np
# import shap
# import matplotlib.pyplot as plt
# import joblib
from glob import glob
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors


sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/pathing/")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/datasets")
# from group_descriptors import getGroups
# from analyse_datasets import plotDescriptorAnalysis

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation/")
from vis import Visualise


paths=getPaths()

# endregion
path = "/users/yhb18174/TL_project/datasets/descriptors/LD50_descriptors/ld50_rdkit_1.csv"
out_path = "/users/yhb18174/TL_project/scripts/sandbox/checking_descriptor_generation.csv"

df = pd.read_csv(path, index_col=0)
if "SMILES" not in df.columns:
    raise ValueError("Input file must contain 'SMILES' column.")

mols = df["SMILES"].astype(str).apply(Chem.MolFromSmiles)

rows = []
for mol in mols:
    if mol is None:
        rows.append({f"{name}_rdkit": np.nan for name, _ in Descriptors.descList})
        continue

    try:
        d = Descriptors.CalcMolDescriptors(mol)  # full RDKit suite
    except Exception:
        d = {name: np.nan for name, _ in Descriptors.descList}

    rows.append({f"{k}_rdkit": v for k, v in d.items()})

desc_df = pd.DataFrame(rows, index=df.index)
df_out = pd.concat([df[["SMILES"]], desc_df], axis=1)

df_out.to_csv(out_path, index_label="ID")
print(f"Saved: {out_path}")

nan_counts = desc_df.isna().sum().sort_values(ascending=False)
print("\nNaN count per descriptor:")
print(nan_counts)

print("\nDescriptors with any NaN:")
print(nan_counts[nan_counts > 0])
