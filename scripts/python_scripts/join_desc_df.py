import pandas as pd
from glob import glob

from pathlib import Path
SCRIPTS_DIR = Path(__file__).parent.parent

import sys

sys.path.insert(0, str(SCRIPTS_DIR / "path"))
from get_paths import getPaths

paths = getPaths()


# Creating a single csv for all properties 
# which holds SMILES and properties

def makeUniqueSMILES(
        properties: list=["Boiling_Point", "LogD", "LD50", "pKa", "pIC50"],
        target_dict: dict=paths["targets"],
        override_full_csv: bool=True
        ):
    """
    Creates a Data Frame which contains only unique SMILES from across all specified properties
    """

    full_smi_df = pd.DataFrame()

    # Looping over all target data frames
    for prop, path in target_dict.items():
        if prop == "all":
            continue
        try:
            df = pd.read_csv(path, 
                            #usecols=["SMILES", "ID"], 
                            index_col="ID"
                            )
        except FileNotFoundError as e:
            print("Property file not found. Skipping.\n{e}")
            continue
        
        # Concatenate all target data frames and drop the duplicates
        full_smi_df = pd.concat([full_smi_df, df], axis=0)

    full_smi_df = full_smi_df.drop_duplicates(subset="SMILES")

    # Keep all available propery data
    cols_to_keep = ["SMILES"] + properties
    full_smi_df = full_smi_df[cols_to_keep]

    out_path = target_dict["all"]

    # Saving the full csv
    if override_full_csv or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_smi_df.to_csv(out_path, index_label="ID")
        print(f"Saved '{out_path}'")
    else:
        print(f"'{out_path}' already exists. Skipping.")



def _find_bad_ids(df: pd.DataFrame, file: str, n=20):
    idx = df.index

    # If index isn't strings, make it so for regex checks
    idx_s = idx.astype(str)

    extracted = idx_s.str.extract(r"(\d+)")[0]  # digits or NaN

    # NaN index values OR "no digits found"
    bad_mask = extracted.isna() | idx.isna()

    if bad_mask.any():
        bad_vals = idx_s[bad_mask].unique()
        print(f"\n[BAD IDS] file={file}")
        print(f"  bad_rows={bad_mask.sum()} unique_bad={len(bad_vals)} sample={bad_vals[:n]}")
        print("  index dtype:", idx.dtype)
        # show which rows they correspond to
        print(df.iloc[list(pd.Series(bad_mask).to_numpy().nonzero()[0])].head(3))



def combineAllFeats(
    full_smi_df: pd.DataFrame | str | Path = paths["targets"]["all"],
    feat_set_ls: list = [
        # "rdkit", 
        # "mordred", 
        "chemberta", 
        "molformer"],
    feature_paths: dict = paths["full_features"],
    cols_to_drop: list[str] = ["SMILES"],
    properties: list = None,
    save: bool=True
):
    
    if properties is None:
        properties = list(feature_paths.keys())[:-1]

    if isinstance(full_smi_df, (str, Path)):
        full_smi_df = pd.read_csv(full_smi_df, index_col="ID")

    for desc_emb in feat_set_ls:
        out_path = feature_paths["all"][desc_emb]
        print(f"- Processing {desc_emb}")
        print(f"- Output path {out_path}")

        full_feat_df = pd.DataFrame()

        for prop in properties:
            print(f"    - Processing {prop}")
            file_prefix = feature_paths[prop][desc_emb]
            files = glob(str(file_prefix))

            temp_df = pd.DataFrame()
            for file in files:
                df = pd.read_csv(file, index_col="ID", low_memory=False)

                # Drop potentially problematic columns like SMILES
                df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

                # diagnose raw IDs
                _find_bad_ids(df, file)  
                df = df.apply(pd.to_numeric, errors="coerce")
                temp_df = pd.concat([temp_df, df], axis=0)

            # safe sort
            temp_df = temp_df.sort_index(
                key=lambda x: pd.to_numeric(
                    x.astype(str).str.extract(r"(\d+)")[0],
                    errors="coerce"
                ).fillna(10**18)
            )

            full_feat_df = pd.concat([full_feat_df, temp_df], axis=0)
        
        print(f"Length of DF: {len(full_feat_df)}")
        print("Rows with any NaN:", full_feat_df.isna().any(axis=1).sum())

        print("Dropping columns with NaN present")
        before_cols = full_feat_df.columns
        full_feat_df_clean = full_feat_df.dropna(axis=1)
        after_cols = full_feat_df_clean.columns
        dropped_cols = before_cols.difference(after_cols)
        print(f"Dropped {len(dropped_cols)}")

        if save:
            full_feat_df_clean.to_csv(out_path, index_label="ID")
            with open(out_path.parent / f"nan_cols_{desc_emb}.txt", "w") as f:
                for col in dropped_cols:
                    f.write(col + "\n")

    


def getCommonIDs(
        df_path_ls: list[pd.DataFrame] = paths['full_features']['all'].values(),
        output_path_ls: list[Path] = paths['aligned_features']['all'].values()
        ):
    
    common_ids = None
    
    for df_path in df_path_ls:
        print(f"Processing {df_path}")
        id_set = set(pd.read_csv(df_path, low_memory=False)["ID"])

        if common_ids is None:
            common_ids = id_set

        else: 
            common_ids &= id_set
        
        print(f"Current intersection size: {len(common_ids)}")

        if not common_ids:
            break
    
    for df_path, output_path in zip(df_path_ls, output_path_ls):
        print(f"Generating common IDs between all features:")
        print(f"{df_path} -> {output_path}")
        df = pd.read_csv(df_path, index_col="ID", low_memory=False)
        df = df.loc[list(common_ids)]
        df.to_csv(output_path, index_label="ID")



# makeUniqueSMILES()
combineAllFeats(save=True)
# getCommonIDs()