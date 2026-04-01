"""
Script for standardising SMILES strings for each dataset
"""

# %%
import pandas as pd
from pathlib import Path
import sys
from glob import glob

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "datasets"))
from standardise_dataset import getMoleculeIntersection, cleanAndSaveDataset

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData


# % ========= Constants =========
PATHS = getPaths()
RANDOM_SEED = 42

# =======================
# Calls for each dataset
# =======================
data_to_clean = []

    # === Boiling Point
if "bp" in data_to_clean:
    cleanAndSaveDataset(
        in_path=Path(PROJ_DIR / "datasets" / "boiling_point" / "boiling_point.csv"),
        out_path=Path(PROJ_DIR / "datasets" / "boiling_point" / "cleaned_boiling_point.csv"),
        usecols=["Boiling Point {measured, converted}", "UNIT {Boiling Point}.1", "SMILES"],
        rename={
            "Boiling Point {measured, converted}": "Boiling_Point",
            "UNIT {Boiling Point}.1": "Unit",
        },
        target_col="Boiling_Point",
        unit_col="Unit",
        id_prefix="bp",
        average_ranges=True,
    )

    # === logD
if "logd" in data_to_clean:
    cleanAndSaveDataset(
        in_path=Path(PROJ_DIR / "datasets" / "logD" / "logd.csv"),
        out_path=Path(PROJ_DIR / "datasets" / "logD" / "cleaned_logd.csv"),
        usecols=["LogD", "UNIT {LogD}", "pH", "UNIT {pH}", "Temperature", "SMILES"],
        rename={"UNIT {LogD}": "Unit"},
        target_col="LogD",
        unit_col="Unit",
        id_prefix="logd",
        average_ranges=True,
    )

    # === pKa
if "pka" in data_to_clean:
    cleanAndSaveDataset(
        in_path=Path(PROJ_DIR / "datasets" / "pka" / "pka_molecules.csv"),
        out_path=Path(PROJ_DIR / "datasets" / "pka" / "cleaned_pka.csv"),
        usecols=["pKa", "UNIT {pKa}", "SMILES"],
        rename={"UNIT {pKa}": "Unit"},
        target_col="pKa",
        unit_col="Unit",
        id_prefix="pka",
        average_ranges=True,
    )

    # === LD50
if "ld50" in data_to_clean:
    cleanAndSaveDataset(
        in_path=Path(PROJ_DIR / "datasets" / "LD50" / "ld50_rat-mouse_oral_lt_9500.csv"),
        out_path=Path(PROJ_DIR / "datasets" / "LD50" / "cleaned_ld50.csv"),
        usecols=["LD50 {measured, converted}", "UNIT {LD50}", "SMILES"],
        rename={"LD50 {measured, converted}": "LD50", "UNIT {LD50}": "Unit"},
        target_col="LD50",
        unit_col="Unit",
        id_prefix="ld50",
        average_ranges=True,
    )

    # === pIC50
if "pic50" in data_to_clean:
    cleanAndSaveDataset(
        in_path=Path(PROJ_DIR / "datasets" / "pic50" / "pic50_data.csv"),
        out_path=Path(PROJ_DIR / "datasets" / "pic50" / "cleaned_pic50.csv"),
        usecols=["pIC50", "SMILES"],
        rename=None,
        target_col="pIC50",
        unit_col=None,
        id_prefix="pic50",
        average_ranges=False,
    )


# %% =======================================
paths = getPaths()
feat_paths = paths['full_features']

tasks = [
    'pka', "logd", "ld50", "bp", "pic50"
    ]

for task in tasks:
    print(f"\n{task}")
    desc_paths = feat_paths[task]

    n_dfs = len(glob(str(desc_paths["rdkit"])))
    print("Number of DFs found:", n_dfs)

    for n in range(1, n_dfs+1):
        try:
            getMoleculeIntersection(df_ls = [
                    desc_paths['rdkit'].parent     / f"{task}_rdkit_{n}.csv",
                    desc_paths['mordred'].parent   / f"{task}_mordred_{n}.csv",
                    desc_paths['chemberta'].parent / f"{task}_chemberta_{n}.csv",
                    desc_paths['molformer'].parent / f"{task}_molformer_{n}.csv",
            ],
            index_col = "ID",
            on_id = True,
            save_paths = [
                    aligned_desc_paths['rdkit'].parent     / f"{task}_rdkit_{n}.csv",
                    aligned_desc_paths['mordred'].parent   / f"{task}_mordred_{n}.csv",
                    aligned_desc_paths['chemberta'].parent / f"{task}_chemberta_{n}.csv",
                    aligned_desc_paths['molformer'].parent / f"{task}_molformer_{n}.csv",
            ]
            )
            
        except Exception as e:
            print(f"Exception made: {e}")

# %%
