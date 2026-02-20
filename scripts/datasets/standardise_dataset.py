# %%
"""
Script to keep dataset standardisation functions and classes
"""
import pandas as pd
from pathlib import Path
import sys
import re
from typing import Union
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import numpy as np


FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from misc_fns import loadData


def averageRanges(
        df: str | Path | pd.DataFrame,
        column: str,
        index_col: int | str=0,
        wildcard: str="*",
        ref_column: str="SMILES"
) -> pd.DataFrame:
    """
    Averages ranges which come from datasets. E.g., if a boiling point
    is reported as 100-102 then this function will replace that with 101.
    Similarly, if a value is reported as ± it will take the face value

    Parameters
    ----------
    df : str, Path, pd.DataFrame
                        Path to a CSV file, a wildcard pattern (e.g., "*.csv"), 
                        or an existing DaataFrame.
    column: str
                        Column which you wish to average values for
    index_col : int, str (optional)
                        Column to use as the index when reading CSV files.
                        Default = 0.
    wildcard : str (optional)
                        Character used to denote a wildcard pattern when loading 
                        multiple files.
                        Default = "*".

    Returns
    -------
    pd.DataFrame
                        A copy of the original DataFrame containing averaged values
                        for the specified column
    """

    df = loadData(
        df=df,
        index_col=index_col,
        wildcard=wildcard,
    )


    # Normalize dash/minus variants to ASCII hyphen
    s = (df[column].astype(str).str.strip()
        .str.replace("\u2212", "-", regex=False)   # Unicode minus
        .str.replace("–", "-", regex=False)        # en dash
        .str.replace("—", "-", regex=False))       # em dash

    # Remove uncertainty tails like "± 0.01", "+/- 0.01", or "+- 0.01"
    # also strip wrapping parentheses e.g. "(6.42 ± 0.01)" -> "6.42"
    s = s.str.replace(r'^\((.*?)\)$', r'\1', regex=True)
    s = s.str.replace(
        r'\s*(?:±|\+/-|\+-)\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?',
        '',
        regex=True
    ).str.strip()

    # Split on a true range hyphen (digit on left, optional minus+digit on right)
    parts = s.str.split(r'(?<=\d)\s*-\s*(?=-?\d)', n=1, expand=True)

    # Start with singles coerced to numbers
    out = pd.to_numeric(s, errors="coerce")

    # Compute midpoint for detected ranges
    if parts.shape[1] > 1:
        mask = parts[1].notna()
        a = pd.to_numeric(parts.loc[mask, 0], errors="coerce")
        b = pd.to_numeric(parts.loc[mask, 1], errors="coerce")
        out.loc[mask] = (a + b) / 2

    # Write back
    df[column] = out
    
    df = (
        df.groupby([ref_column, "Unit"], as_index=False)
        .agg({column: "mean"})
    )

    df[column] = df[column].round(2)
    
    return df

def removeFragments(
        df: str | Path | pd.DataFrame,
        smi_column: str,
        index_col: str | int=0,
        wildcard: str="*",
        ) -> pd.DataFrame:
    """
    Function to remove SMILES strings that contain fragments/multiple entities
    """
    df = loadData(
        df=df,
        index_col=index_col,
        wildcard=wildcard,
    ).copy()

    mask = df[smi_column].astype(str).str.contains(r"\.", na=False)
    df = df.loc[~mask]

    return df

def removeMetals(
    df: str | Path | pd.DataFrame,
    smi_column: str,
    index_col: str | int = 0,
    wildcard: str = "*",
    keep_nonmetal_frag: bool = False,

) -> pd.DataFrame:
    """
    Remove rows whose SMILES contain metals (no RDKit).
    If keep_nonmetal_frag=True, keep the largest fragment without metals; drop row if none.
    """
    df = loadData(df=df, index_col=index_col, wildcard=wildcard).copy()

    metals = [
        "Li","Be","Na","Mg","Al","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga",
        "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb",
        "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
        "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi",
        "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    ]
    metals_alt = "|".join(sorted(metals, key=len, reverse=True))
    metal_re = re.compile(rf"\[[^\]]*(?:{metals_alt})[^\]]*\]")

    smi = df[smi_column].astype(str)

    if not keep_nonmetal_frag:
        has_metal = smi.str.contains(metal_re, na=False)
        return df.loc[~has_metal].copy()

    # keep the largest fragment without metals (string heuristic)
    def choose_fragment(s: str) -> str | None:
        frags = s.split(".")
        clean = [f for f in frags if not metal_re.search(f)]
        if not clean:
            return None
        return max(clean, key=len)

    df[smi_column] = smi.map(choose_fragment)
    return df[df[smi_column].notna()].copy()

def getMoleculeIntersection(
    df_ls: list[Union[Path, str, pd.DataFrame]],
    index_col: str | int = 0,
    exclude: list[str] = [],
    wildcard: str = "*",
    smiles_col: str = "SMILES",
    on_id: bool = False,
    on_smiles: bool = False,
    on_columns: bool = False,
    save_paths: list[str | Path] = None,
):
    """
    Get the intersection between multiple datasets based on:
      - ID (index)
      - SMILES column
      - columns
    """

    if save_paths and len(save_paths) != len(df_ls):
        raise ValueError(
            f"Length mismatch:\ndf_ls = {len(df_ls)}\nsave_paths = {len(save_paths)}"
        )

    if not any([on_id, on_smiles, on_columns]):
        raise ValueError("At least one of on_id, on_smiles, on_columns must be True.")

    loaded_dfs = [
        loadData(df=df, index_col=index_col, exclude=exclude, wildcard=wildcard)
        for df in df_ls
    ]

    # Normalise/validate SMILES only if needed
    if on_smiles:
        for i, d in enumerate(loaded_dfs):
            if smiles_col not in d.columns:
                raise ValueError(f"SMILES column '{smiles_col}' not found in dataframe {i}.")
            loaded_dfs[i] = (
                d.dropna(subset=[smiles_col])
                 .assign(**{smiles_col: d[smiles_col].astype(str).str.strip()})
            )

    # Compute intersections
    common_cols = None
    common_idx = None
    common_smiles = None

    if on_columns:
        it = iter(loaded_dfs)
        common_cols = next(it).columns
        for d in it:
            common_cols = common_cols.intersection(d.columns)

    if on_id:
        it = iter(loaded_dfs)
        common_idx = next(it).index
        for d in it:
            common_idx = common_idx.intersection(d.index)

    if on_smiles:
        it = iter(loaded_dfs)
        common_smiles = pd.Index(next(it)[smiles_col])
        for d in it:
            common_smiles = common_smiles.intersection(pd.Index(d[smiles_col]))

    # Apply filters
    result = []
    for df in loaded_dfs:
        if on_smiles:
            df = df[df[smiles_col].isin(common_smiles)]
        if on_id:
            df = df.loc[common_idx]
        if on_columns:
            df = df.loc[:, common_cols]
        result.append(df.copy())

    # Save (keep ID as index)
    if save_paths:
        for path, df in zip(save_paths, result):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=True, index_label=df.index.name or "ID")

    return result

def filterMolWt(
        df: pd.DataFrame | str | Path,
        smiles_col: str="SMILES",
        index_col: str | int=0,
        min_threshold: int=0,
        max_threshold: int=500
):
    
    if isinstance(df, (str, Path)):
        df = pd.read_csv(df, index_col=index_col)

    df['Mol'] = df[smiles_col].apply(lambda s: Chem.MolFromSmiles(s))
    df["MolWt"] = df['Mol'].apply(lambda m: Descriptors.MolWt(m) if m is not None else np.nan)
    
    df = df.copy()
    df = df.drop(columns=["Mol"])

    df = df[
        (df["MolWt"] > min_threshold) &
        (df["MolWt"] < max_threshold)
        ]
    
    return df.copy()

def canonicaliseSMILES(smi):
    
    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None
    
    enumerator = rdMolStandardize.TautomerEnumerator()
    canon_mol = enumerator.Canonicalize(mol)

    if canon_mol is None:
        return None

    # Canonicalise SMILES string
    canon_smi = Chem.MolToSmiles(canon_mol, isomericSmiles=True, canonical=True)
    return canon_smi

# %%



def cleanAndSaveDataset(
    *,
    in_path: Path,
    out_path: Path,
    usecols: list[str],
    rename: dict[str, str] | None = None,
    target_col: str,
    id_prefix: str,
    unit_col: str | None = "Unit",
    smiles_col: str = "SMILES",
    average_ranges: bool = True,
    keep_cols: list[str] | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Load a dataset, standardise SMILES, clean rows, optionally average target ranges,
    shuffle, add sequential IDs, and save.

    Returns the cleaned dataframe.
    """
    df = pd.read_csv(in_path, index_col=False)

    # select + rename
    df = df[usecols].copy()
    if rename:
        df = df.rename(columns=rename)

    # ensure expected columns exist after rename
    required = {target_col, smiles_col}
    if unit_col is not None:
        required.add(unit_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after rename: {missing}")

    # keep only needed cols (default: target, unit (if any), smiles)
    if keep_cols is None:
        keep_cols = [target_col, smiles_col] + ([unit_col] if unit_col is not None else [])
    df = df.loc[:, keep_cols].copy()

    # drop missing/empty SMILES early
    df = df.dropna(subset=[smiles_col])
    df[smiles_col] = df[smiles_col].astype(str).str.strip()
    df = df[df[smiles_col] != ""]

    # canonicalise smiles
    df[smiles_col] = df[smiles_col].apply(canonicaliseSMILES)

    # average ranges
    if average_ranges:
        df = averageRanges(df=df, column=target_col, index_col=None)

    # remove fragments and metals
    df = removeFragments(df=df, smi_column=smiles_col, index_col=None)
    df = removeMetals(df=df, smi_column=smiles_col, index_col=None)

    # coerce target to numeric + final dropna on required fields
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[smiles_col] = df[smiles_col].astype(str).str.strip().replace({"": pd.NA, "None": pd.NA})
    df = df.dropna(subset=[target_col, smiles_col])

    # shuffle + reset index
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # add ID column
    df.insert(0, "ID", [f"{id_prefix}_{n}" for n in range(1, len(df) + 1)])

    # save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    return df
