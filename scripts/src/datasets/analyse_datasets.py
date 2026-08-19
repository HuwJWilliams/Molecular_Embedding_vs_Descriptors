import pandas as pd
from pathlib import Path
import numpy as np

# % ========= Constants =========
FILE_DIR = Path(__file__).parent
SRC_DIR = FILE_DIR.parent
PROJ_DIR = SRC_DIR.parent.parent


def getLowVarianceColumns(
    input_df: str | Path,
    threshold: float = 0.95,
    index_col: str | None = "ID",
    exclude_columns: list[str] | None = None,
) -> list[str]:
    """
    Identify columns where the most common value accounts for at least
    ``threshold`` of all rows.

    Parameters
    ----------
    input_df : str | Path
        Path to the CSV file to inspect.
    threshold : float, optional
        Minimum fraction of rows occupied by the most common value for a column
        to be flagged. Default = 0.95.
    index_col : str | None, optional
        Column to use as the index when reading the CSV. Default = "ID".
    exclude_columns : list[str] | None, optional
        Columns to skip, e.g. metadata such as "SMILES". Default = None.

    Returns
    -------
    list[str]
        Column names flagged as near-constant by the threshold rule.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    exclude_columns = set(exclude_columns or [])

    if isinstance(input_df, (str, Path)):
        df = pd.read_csv(input_df, index_col=index_col)
    else:
        df = input_df

    low_variance_cols = []

    for col in df.columns:
        if col in exclude_columns:
            continue

        dominant_fraction = df[col].value_counts(normalize=True, dropna=False).iloc[0]

        if dominant_fraction >= threshold:
            low_variance_cols.append(col)

    return low_variance_cols


def trimRowsByPercentile(
    input_df: str | Path | pd.DataFrame,
    columns: list[str] | None = None,
    percentile: float = 0.99,
    tail: str = "upper",
    index_col: str | None = "ID",
    exclude_columns: list[str] | None = None,
    return_removed_rows: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows whose values fall outside a percentile cutoff.

    This is intended for trimming rows with extreme feature values before model
    training. By default, it removes rows above the 99th percentile for the
    selected numeric columns.
    """

    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1.")

    if tail not in {"upper", "lower", "both"}:
        raise ValueError("tail must be one of: 'upper', 'lower', 'both'.")

    if isinstance(input_df, pd.DataFrame):
        df = input_df.copy()
    else:
        df = pd.read_csv(input_df, index_col=index_col)

    exclude_columns = set(exclude_columns or [])

    if columns is None:
        candidate_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in candidate_columns if col not in exclude_columns]
    else:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"These columns are missing from the input data: {missing}"
            )

    if not columns:
        raise ValueError("No columns available for percentile trimming.")

    keep_mask = pd.Series(True, index=df.index)

    for col in columns:
        s = df[col]

        if not pd.api.types.is_numeric_dtype(s):
            raise TypeError(f"Column '{col}' must be numeric for percentile trimming.")

        # numpy/pandas quantile interpolation can fail on boolean dtype
        # (e.g., "numpy boolean subtract" TypeError). Coerce to float first.
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)

        lower_cutoff = s.quantile(1 - percentile)
        upper_cutoff = s.quantile(percentile)

        if tail == "upper":
            keep_mask &= s.le(upper_cutoff) | s.isna()
        elif tail == "lower":
            keep_mask &= s.ge(lower_cutoff) | s.isna()
        else:
            keep_mask &= s.between(lower_cutoff, upper_cutoff) | s.isna()

    trimmed_df = df.loc[keep_mask].copy()

    if return_removed_rows:
        removed_df = df.loc[~keep_mask].copy()
        return trimmed_df, removed_df

    return trimmed_df


def checkLipinskiCriteria(
    df: str | Path | pd.DataFrame,
    mw: int = 600,
    logp: float = 6,
    n_hbd: int = 6,
    n_hba: int = 11,
    columns: list[str] = [
        "MolWt_rdkit",
        "MolLogP_rdkit",
        "NumHDonors_rdkit",
        "NumHAcceptors_rdkit",
    ],
) -> list[str]:
    """Function to check how many molecules fit the Lipinski Ro5, default to 'relaxed' criteria"""

    if isinstance(df, (str, Path)):
        df = pd.read_csv(df, index_col=0)

    n_orig = len(df)
    print(f"Number of original molecules:\n{n_orig}")

    trimmed_df = df
    criteria = [mw, logp, n_hbd, n_hba]

    for i, desc in enumerate(columns):
        trimmed_df = trimmed_df[trimmed_df[desc] < criteria[i]]
        if trimmed_df.empty:
            raise ValueError("Empty dataframe, make sure that the columns list\
                             is ordered: mw, logp, hb donor, hb acceptor")

    n_pass = len(trimmed_df)

    print(f"Number of molecules fitting Lipinski Criteria:\n \
          {n_pass} ({round((n_pass/n_orig)*100, 2)} %)")

    return trimmed_df.index
