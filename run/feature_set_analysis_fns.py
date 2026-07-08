"""
Defining and Running analysis for feature data
"""

# %% ===== Python Imports =====
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
import zipfile
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from collections import Counter

# %% ===== Project Imports & Pathing Setup =====
from config import PATHING_JSON_PATH, SRC_DIR

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)


# %% ===== Function Definitions =====
def getFeatureDistributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    zip_all: bool = True,
    save_dir: str | Path = FULL_PATHING["imp_dirs"]["datasets_dir"]
    / "feature_distributions",
):
    """
    Plot the feature distribution of defined columns, if none defined it will use all available numerical columns in
    the provided dataframe

    Returns
    -------
    list[str]: List of save paths
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
    else:
        columns = df[columns].select_dtypes(include="number").columns.tolist()

    df = df[columns]

    save_paths = []

    for col in columns:
        values = df[col].dropna()

        if values.empty:
            print(f"Skipping {col}: no non-null numeric values")
            continue

        mean_val = values.mean()
        min_val = values.min()
        max_val = values.max()
        data_range = max_val - min_val

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_iqr = q1 - 3.0 * iqr
        upper_iqr = q3 + 3.0 * iqr

        outlier_mask = (df[col] < lower_iqr) | (df[col] > upper_iqr)

        safe_col = str(col).replace("/", "_").replace("\\", "_").replace(" ", "_")
        save_name = f"{safe_col}_dist_3xIQR.png"
        save_path = save_dir / save_name

        plt.figure(figsize=(8, 5))

        plt.axvspan(
            lower_iqr,
            upper_iqr,
            alpha=0.12,
            label=f"3x IQR bounds: {lower_iqr:.2f} to {upper_iqr:.2f}",
        )

        sns.histplot(values, kde=True, bins=30)

        plt.axvline(
            mean_val,
            linestyle="-",
            linewidth=2,
            label=f"Mean = {mean_val:.2f}",
        )

        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(
            f"Distribution of {col} values\n"
            f"Range: {min_val:.2f} to {max_val:.2f} | "
            f"Outliers flagged: {outlier_mask.sum()}"
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        save_paths.append(save_path)

        print(f"{save_name}")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  Range: {data_range:.4f}")
        print(f"  Q1: {q1:.4f}")
        print(f"  Q3: {q3:.4f}")
        print(f"  IQR: {iqr:.4f}")
        print(f"  3x IQR bounds: {lower_iqr:.4f} to {upper_iqr:.4f}")
        print(f"  Outliers flagged: {outlier_mask.sum()} / {df[col].notna().sum()}")
        print()

    if zip_all and save_paths:
        zip_path = save_dir / "feature_distributions.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in save_paths:
                zf.write(path, arcname=path.name)

        print(f"Saved zip: {zip_path}")

    return save_paths


def plotBarChart(
    counts: Counter | dict | pd.DataFrame,
    title: str = "Distribution",
    save_path: str | Path | None = None,
    x_col: str = "class",
    y_col: str = "count",
    show: bool = True,
) -> pd.DataFrame:
    """
    Plot a bar chart from a Counter/dict or a summary dataframe.

    Returns a dataframe with:
        class, count, percentage
    """

    if isinstance(counts, pd.DataFrame):
        summary_df = counts.copy()
    else:
        summary_df = pd.DataFrame(
            {
                x_col: list(counts.keys()),
                y_col: list(counts.values()),
            }
        )

    if summary_df.empty:
        raise ValueError(f"No data available for plot: {title}")

    summary_df = summary_df.sort_values(y_col, ascending=False).reset_index(drop=True)
    summary_df["percentage"] = summary_df[y_col] / summary_df[y_col].sum() * 100

    plt.figure(figsize=(10, 6))
    plt.bar(summary_df[x_col], summary_df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)

        if save_path.suffix == "":
            raise ValueError(
                "save_path must include a filename, e.g. " "'.../atom_distribution.png'"
            )

        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        csv_path = save_path.with_suffix(".csv")
        summary_df.to_csv(csv_path, index=False)

        print(f"Saved bar chart to: {save_path}")
        print(f"Saved summary CSV to: {csv_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return summary_df


def checkAtomDistribution(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    save_path: str | Path = FULL_PATHING["imp_dirs"]["datasets_dir"],
):
    """
    Check the distribution of atom types within the provided dataframe.
    """

    atom_counts = Counter()
    invalid_smiles = 0

    for smi in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            invalid_smiles += 1
            continue

        for atom in mol.GetAtoms():
            atom_counts[atom.GetSymbol()] += 1

    total_atoms = sum(atom_counts.values())

    print(f"Invalid SMILES skipped: {invalid_smiles}")
    print(f"Total atoms counted: {total_atoms}")

    summary_df = plotBarChart(
        atom_counts,
        title=f"Atom Counts ({total_atoms})",
        save_path=Path(save_path) / "atom_distribution.png",
    )

    return summary_df, atom_counts


def checkDegreeSubstitutionDistribution(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    elements: tuple[str, ...] = ("C", "N", "O", "S", "Si"),
    save_path: str | Path = FULL_PATHING["imp_dirs"]["datasets_dir"],
):
    """
    Count atom degree/substitution classes.

    Degree is defined as number of heavy-atom neighbours.

    Example classes:
        C_primary
        C_secondary
        N_primary
        O_secondary
        Si_quaternary
    """

    def _degree_label(degree: int) -> str:
        if degree == 0:
            return "zero_degree"
        elif degree == 1:
            return "primary"
        elif degree == 2:
            return "secondary"
        elif degree == 3:
            return "tertiary"
        else:
            return "quaternary"

    def _heavy_atom_degree(atom) -> int:
        return sum(
            1 for neighbour in atom.GetNeighbors() if neighbour.GetAtomicNum() != 1
        )

    degree_counts = Counter()
    invalid_smiles = 0

    for smi in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            invalid_smiles += 1
            continue

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()

            if symbol not in elements:
                continue

            degree = _heavy_atom_degree(atom)
            degree_type = _degree_label(degree)

            atom_degree_class = f"{symbol}_{degree_type}"
            degree_counts[atom_degree_class] += 1

    total_atoms = sum(degree_counts.values())

    print(f"Invalid SMILES skipped: {invalid_smiles}")
    print(f"Total atoms counted: {total_atoms}")

    summary_df = plotBarChart(
        degree_counts,
        title=f"Atomic Degree Counts ({total_atoms})",
        save_path=Path(save_path) / "degree_substitution_distribution.png",
    )

    return summary_df, degree_counts


def checkCarbonHybridisationDistribution(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    save_path: str | Path = FULL_PATHING["imp_dirs"]["datasets_dir"],
    split_aromatic: bool = False,
):
    """
    Check the distribution of carbon hybridisation within the provided dataframe.

    If split_aromatic=True, aromatic carbons are counted as C_aromatic.
    Otherwise, aromatic carbons are usually counted as Csp2 by RDKit.
    """

    def _hybridisation_label(atom) -> str:
        if split_aromatic and atom.GetIsAromatic():
            return "C_aromatic"

        hyb = atom.GetHybridization()

        if hyb == HybridizationType.SP:
            return "Csp1"
        elif hyb == HybridizationType.SP2:
            return "Csp2"
        elif hyb == HybridizationType.SP3:
            return "Csp3"
        elif hyb == HybridizationType.SP3D:
            return "Csp3d"
        elif hyb == HybridizationType.SP3D2:
            return "Csp3d2"
        else:
            return "C_other"

    c_hybridisation_counts = Counter()
    invalid_smiles = 0

    for smi in df[smiles_col].dropna():
        mol = Chem.MolFromSmiles(str(smi))

        if mol is None:
            invalid_smiles += 1
            continue

        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "C":
                continue

            c_hybridisation_counts[_hybridisation_label(atom)] += 1

    total_c = sum(c_hybridisation_counts.values())

    print(f"Invalid SMILES skipped: {invalid_smiles}")
    print(f"Total carbons counted: {total_c}")

    summary_df = plotBarChart(
        c_hybridisation_counts,
        title=f"Carbon Hybridisation Counts ({total_c})",
        save_path=Path(save_path) / "carbon_hybridisation_distribution.png",
    )

    return summary_df, c_hybridisation_counts
