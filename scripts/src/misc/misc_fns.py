import pandas as pd
from pathlib import Path
from glob import glob
import logging
import sys
import numpy as np

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

#  --- CONSTANTS
PATHS = getPaths()

def loadData(
        df: str | pd.DataFrame | Path,
        index_col: int | str=0,
        exclude: list[str]=[],
        wildcard: str="*",
        force_float32: bool=True,  # Changed default to True
) -> pd.DataFrame:
    """
    Flexibly loads data into a pandas DataFrame from various input types.
    This function accepts either a file path, a wildcard pattern (to merge 
    multiple CSV files), or an existing DataFrame. It standardises the input 
    into a clean, indexed DataFrame and optionally removes specified columns.
    Parameters
    ----------
    df : str, Path, pd.DataFrame
                        Path to a CSV file, a wildcard pattern (e.g., "*.csv"), 
                        or an existing DataFrame.
    index_col : int, str (optional)
                        Column to use as the index when reading CSV files.
                        Default = 0.
    exclude : list[str] (optional)
                        List of column names to drop after loading.
                        Default = [].
    wildcard : str (optional)
                        Character used to denote a wildcard pattern when loading 
                        multiple files.
                        Default = "*".
    force_float32 : bool (optional)
                        If True, convert all numeric columns to float32, clipping
                        any values that exceed the float32 range.
                        Default = True.
    Returns
    -------
    pd.DataFrame
                        A single DataFrame containing all loaded and cleaned data.
    Raises
    ------
    TypeError
                        If the input is not a path (str/Path) or a pandas DataFrame.
    """
    if isinstance(df, (str, Path)) and wildcard in str(df):
        loaded_df = pd.concat([pd.read_csv(f, index_col=index_col) for f in glob(str(df))], axis=0)
        try:
            print("Trying to number the index in order")
            loaded_df = loaded_df.sort_index(
                key=lambda idx: idx.str.extract(r'(\d+)$').astype(int)[0]
            )
            print("Succeeded.")
        except Exception as e:
            print("Failed. Returning unsorted DF")
    elif isinstance(df, (str, Path)):
        loaded_df = pd.read_csv(df, index_col=index_col)
    elif isinstance(df, pd.DataFrame):
        loaded_df = df.copy()
    else:
        raise TypeError("Input must be a path (str/Path) or a pandas DataFrame.")
    
    if exclude:
        loaded_df = loaded_df.drop(columns=[c for c in exclude if c in loaded_df.columns])
    
    # FLOAT32 CONVERSION WITH CLIPPING
    if force_float32:
        numeric_cols = loaded_df.select_dtypes(include=[np.number]).columns
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        
        clipped_cols = []
        
        for col in numeric_cols:
            max_val = loaded_df[col].abs().max()
            
            # Check if clipping is needed
            if max_val > float32_max or loaded_df[col].min() < float32_min:
                # Clip values to float32 range
                loaded_df[col] = loaded_df[col].clip(lower=float32_min, upper=float32_max)
                clipped_cols.append((col, max_val))
        
        if clipped_cols:
            print(f"Clipped {len(clipped_cols)} columns to float32 range")
            # Show first 5 clipped columns
            for col, val in clipped_cols[:5]:
                print(f"  - {col}: original max = {val:.2e}")
        
        # Convert all numeric columns to float32
        loaded_df[numeric_cols] = loaded_df[numeric_cols].astype(np.float32)
        print(f"✓ Converted {len(numeric_cols)} numeric columns to float32")
    
    return loaded_df


def setupLogger(
        name: str,
        identifier: str,
        save_logger: bool,
        level: int=logging.DEBUG,     
        log_dir: Path=PATHS['config']["logs"],
        message: str='%(asctime)s | %(funcName)s | %(message)s'
    ):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(message)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        if save_logger:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"{name}_{identifier}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def fixCSVColumns(
        root_dir: str | Path,
        col_to_drop: str=None,
        rename_to: dict={},
        file_fnames: list="*feature_importance.csv",
        file_ls: list=None,
        ):
    
    if not file_ls:
        file_ls = root_dir.rglob(file_fnames)
    
    for csv_file in file_ls:
        try:
            df = pd.read_csv(csv_file)

            # If duplicate Feature columns exist
            if col_to_drop:
                df = df.drop(columns=["Feature"])

            if rename_to:
                df = df.rename(columns=rename_to)

            df.to_csv(csv_file, index=False)
            print(f"Fixed: {csv_file}")

        except Exception as e:
            print(f"Skipped {csv_file}: {e}")


def getFeatures(
        ids: list[str] | Path | pd.DataFrame,
        feature_name: str,
):
    
    if isinstance(ids, Path):
        id_df = pd.read_csv(ids)
        id_ls = id_df["ID"].to_list()

    elif isinstance(ids, pd.DataFrame):
        id_df = ids.reset_index()
        id_ls = id_df["ID"].to_list()
    
    else:
        id_ls = ids

    feature_data = PATHS["full_features"]["all"][feature_name]
    feature_df = pd.read_csv(feature_data, index_col=0)

    rows = feature_df.loc[feature_df.index.intersection(id_ls)]

    return rows

def center_rows(df):
    return df.sub(df.mean(axis=1), axis=0)

def scale_rows(df):
    denom = df.abs().max(axis=1).replace(0, 1)
    return df.div(denom, axis=0)

def center_columns(df):
    return df.sub(df.mean(axis=0), axis=1)

def scale_columns(df):
    denom = df.abs().max(axis=0).replace(0, 1)
    return df.div(denom, axis=1)

def filter_molecules_by_mw(
    df: pd.DataFrame,
    min_mw: float | None = None,
    max_mw: float | None = None,
    mw_column_candidates: tuple[str, ...] = ("MolWt_rdkit", "MW_mordred", "MolWt"),
) -> pd.DataFrame:
    """Filter a feature dataframe by an existing molecular-weight descriptor column."""

    if min_mw is None and max_mw is None:
        return df

    mw_column = next((col for col in mw_column_candidates if col in df.columns), None)
    if mw_column is None:
        raise ValueError(
            "No molecular-weight column found. "
            f"Tried: {list(mw_column_candidates)}"
        )

    filtered_df = df.copy()

    if min_mw is not None:
        filtered_df = filtered_df[filtered_df[mw_column] >= min_mw]

    if max_mw is not None:
        filtered_df = filtered_df[filtered_df[mw_column] <= max_mw]

    return filtered_df

def get_ids_in_mw_range(
    df: pd.DataFrame,
    min_mw: float | None = None,
    max_mw: float | None = None,
    mw_column_candidates: tuple[str, ...] = ("MolWt_rdkit", "MW_mordred", "MolWt"),
) -> pd.Index:
    """Get the IDs that fall within a molecular-weight range."""

    filtered_df = filter_molecules_by_mw(
        df=df,
        min_mw=min_mw,
        max_mw=max_mw,
        mw_column_candidates=mw_column_candidates,
    )
    return filtered_df.index

def check_path_exists(path, expect=None, non_empty=False, name="path"):
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"{name} does not exist: {p}")

    if expect == "file" and not p.is_file():
        raise NotADirectoryError(f"{name} is not a file: {p}")

    if expect == "dir" and not p.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {p}")

    if non_empty and p.is_file() and p.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {p}")

    return p

def getMostImportantFeatures(
        importance_source,
        top_n: int=25,
        mode: str="shap"
) -> tuple[dict, dict, dict]:
    """
    Summarise the top-N most important features for each descriptor.

    Parameters
    ----------
    importance_source : dict | pd.DataFrame
        For mode='shap': dict with keys 'shap_by_desc' and 'feature_names'.
        For mode='rf':   DataFrame with columns named 'Importance_<descriptor>'.
    top_n : int
        Number of top features to retain.
    mode : str
        'shap' or 'rf'.

    Returns
    -------
    avg_importance_dict : dict
        {descriptor: [top_feature_names, mean_top_n_importance]}
    cum_importance_dict : dict
        {descriptor: [top_feature_names, sum_top_n_importance]}
    count_dict : dict
        {feature_name: number_of_descriptors_where_feature_is_in_top_n}
    """
    avg_importance_dict: dict = {}
    cum_importance_dict: dict = {}
    count_dict: dict = {}

    if mode == "shap":
        shap_by_desc = importance_source["shap_by_desc"]
        feature_names = importance_source["feature_names"]

        for desc, shap_values in shap_by_desc.items():
            top = (
                pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names)
                .nlargest(top_n)
            )
            clean_desc = desc.rsplit("_", 1)[0]
            avg_importance_dict[clean_desc] = [top.index.tolist(), float(top.mean())]
            cum_importance_dict[clean_desc] = [top.index.tolist(), float(top.sum())]
            for feat in top.index.tolist():
                count_dict[feat] = count_dict.get(feat, 0) + 1

    elif mode == "rf":
        fi_df = importance_source.copy()
        if "Feature" in fi_df.columns:
            fi_df = fi_df.set_index("Feature")

        for col in fi_df.columns:
            if not str(col).startswith("Importance_"):
                continue
            desc = str(col).replace("Importance_", "")
            top = pd.to_numeric(fi_df[col], errors="coerce").dropna().nlargest(top_n)
            avg_importance_dict[desc] = [top.index.tolist(), float(top.mean())]
            cum_importance_dict[desc] = [top.index.tolist(), float(top.sum())]
            for feat in top.index.tolist():
                count_dict[feat] = count_dict.get(feat, 0) + 1

    else:
        raise ValueError(f"mode must be 'shap' or 'rf', got '{mode}'")

    return avg_importance_dict, cum_importance_dict, count_dict

def molid2Smiles(molid):
    cleaned_all = pd.read_csv(PROJ_DIR / "datasets" / "all" / "cleaned_all.csv")
    return cleaned_all.loc[cleaned_all["ID"] == molid, "SMILES"].iloc[0]
