# %% --- Imports
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator
from mordred import Calculator, descriptors
from transformers import AutoTokenizer, AutoModel
import torch

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"


sys.path.insert(0, str(SCRIPTS_DIR / "path"))
from get_paths import getPaths

sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from misc_fns import (setupLogger, loadData)

# --- Constants
TOKENIZERS = {
    "MolFormer": "ibm/MoLFormer-XL-both-10pct",
    "ChemBERTa" : "DeepChem/ChemBERTa-100M-MLM"
}
MODELS = {
    "MolFormer": "ibm/MoLFormer-XL-both-10pct",
    "ChemBERTa" : "DeepChem/ChemBERTa-100M-MLM"
}

LOG_LEVEL = logging.DEBUG
PATHS = getPaths()
MIN_UNIQUE = 2


# %% --- Classes & Functions
class FeatureGenerator():
    """
    Class to hold all of the feature generation functions
    """
    def __init__(
            self,
            feature_set: str,
            log_name: str = "FG",
            save_log: bool=True,
            log_level= LOG_LEVEL,
            log_identifier: str = "FeatureGenerator"
    ):
        """
        Class Initialiser
        
        Parameters
        ----------
        feature_set     (str)      Name of the feature set to generate (currently supports [rdkit, mordred, molformer, chemberta])
        log_name        (str)      Name of the logger
        save_log        (bool)     Flag to save logger
        log_level       (int)      Level of logging to save
        log_identifier  (str)      Identifier for the log
        """
        #===== Logger Setup=====#
        self.logger = setupLogger(
            name=log_name,
            save_logger=save_log,
            level=log_level,
            identifier=log_identifier
        )

        self.feature_set = feature_set.lower()
        self.tokeniser = "Uninitialised"
        self.encoder = "Uninitialised"

        if self.feature_set == "chemberta":
            self._initialise_embedding_models(tokeniser=TOKENIZERS['ChemBERTa'], model=MODELS['ChemBERTa'])

        elif self.feature_set == "molformer":
            self._initialise_embedding_models(tokeniser=TOKENIZERS['MolFormer'], model=MODELS['MolFormer'])

        if self.encoder != "Uninitialised":
            self.encoder.eval()

        return

    # ====== Feature Calculations

    def calcRDKit(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            drop_cols: bool=False,
            min_unique: int=MIN_UNIQUE
    ) -> pd.DataFrame:
        
        """
        Description
        -----------
        Function to calculate RDKit descriptors for a list of smiles

        Parameters
        ----------
        smiles_ls           list[str]   List of SMILES
        id_ls               list[str]   List of IDs corresponding to each SMILE
        drop_cols           bool        Flag to drop columns with low variance
        min_unique          int         Variance threshold for dopping columns       
        """
        
        if len(smiles_ls) != len(id_ls):
            self.logger.error(
                f"Length of SMILES and IDs not the same."
                f"(SMILES = {len(smiles_ls)}, IDs = {len(id_ls)})"
            )
            raise ValueError("len(smiles_ls) != len(id_ls)")

        self.logger.info(f"Creating RDKit descriptors for {len(smiles_ls)} smiles.")

        (parsed_ids, parsed_smiles, parsed_mols), _ = self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)
        
        df = self._setup_molecule_df(
            id_ls=parsed_ids, smiles_ls=parsed_smiles, mol_ls = parsed_mols
            )
        
        self.logger.info(f"Generating descriptors for {len(df)} molecules")
        desc_dicts = df["Mols"].apply(lambda m: Descriptors.CalcMolDescriptors(m))
        desc_df = pd.DataFrame.from_records(desc_dicts.tolist(), index=df.index)
        desc_df = desc_df.add_suffix("_rdkit")
        final_df = pd.concat([df.drop(columns=["Mols"]), desc_df], axis=1)
        final_df.index.name = "ID"

        if drop_cols:
            final_df = self._drop_columns(df=final_df, min_unique=min_unique)
            self.logger.info(f"RDKit data frame created with shape: {final_df.shape}")

        return final_df
    
    def calcMordred(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            ignore_3D: bool=False,
            drop_cols: bool=False,
            min_unique: int=MIN_UNIQUE,
    ) -> pd.DataFrame:

        """
        Description
        -----------
        Function to calculate Mordred descriptors for a list of smiles

        Parameters
        ----------
        smiles_ls           list[str]   List of SMILES
        id_ls               list[str]   List of IDs corresponding to each SMILE
        ignore_3d           bool        Flag to ignore 3D mordred descriptors
        drop_cols           bool        Flag to drop columns with low variance
        min_unique          int         Variance threshold for dopping columns       
        """
    
        self.logger.info(f"Creating Mordred descriptors for {len(smiles_ls)} smiles.")

        (parsed_ids, parsed_smiles, parsed_mols), _ = self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

        df = self._setup_molecule_df(
            id_ls=parsed_ids, smiles_ls=parsed_smiles, mol_ls=parsed_mols
            )

        self.logger.info(f"Generating descriptors for {len(df)} valid molecules")
        calc = Calculator(descriptors, ignore_3D=ignore_3D)
        desc_df = calc.pandas(df["Mols"])
        desc_df = desc_df.add_suffix("_mordred")
        final_df = pd.concat([df.drop(columns=["Mols"]), desc_df], axis=1)
        final_df.index.name="ID"

        if drop_cols:
            final_df = self._drop_columns(df=final_df, min_unique=min_unique)
            self.logger.info(f"Mordred data frame created with shape: {final_df.shape}")

        return final_df
    
        
    # ====== Fingerprint Calculations

    def calcMorganFingerprints(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            radius:int=2,
            fingerprint_size:int=1024
    ):
        
        self.logger.info(f"Creating Morgan Fingerprints descriptors for {len(smiles_ls)} smiles.")
        (parsed_ids, parsed_smiles, parsed_mols), _ = self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

        df = self._setup_molecule_df(
            id_ls=parsed_ids, smiles_ls=parsed_smiles, mol_ls=parsed_mols
            )

        self.logger.info(f"Generating dMorgan Fingerprints for {len(df)} valid molecules")

        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpsize=fingerprint_size
            )

        fps = []

        for mol in df["mol"]:
            fp = fpgen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
        
        fp_df = pd.DataFrame(
            fps,
            index=df.index,
            columns=[f"morgan_{i}" for i in range(fingerprint_size)]
        )
        result_df = pd.concat(
            [df[["ID", "SMILES"]].reset_index(drop=True), fp_df.reset_index(drop=True)],
            axis=1
        )

        return result_df

    
    # ====== Embedding Calculations

    def calcChemBERTa(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            batch_size: int=64,
            max_token_len: int=512,
            drop_cols: bool=False,
            min_unique: int=MIN_UNIQUE,
            pooling: str="mean"
    ) -> pd.DataFrame:
        
        """
        Description
        -----------
        Function to calculate ChemBERTa embeddings for a list of smiles

        Parameters
        ----------
        smiles_ls           list[str]   List of SMILES
        id_ls               list[str]   List of IDs corresponding to each SMILE
        batch_size          int         Size of batch to process
        max_token_len         int         Maximum length of the generated embeddings
        drop_cols           bool        Flag to drop columns with low variance
        min_unique          int         Variance threshold for dopping columns       
        pooling             str         Pooling strategy: "cls" or "mean"
        """

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")
        
        if len(smiles_ls) != len(id_ls):
            self.logger.error(
                f"Length of SMILES and IDs not the same."
                f"(SMILES = {len(smiles_ls)}, IDs = {len(id_ls)})"
            )
            raise ValueError("len(smiles_ls) != len(id_ls)")

        self.logger.info(f"Creating ChemBERTa embeddings for {len(smiles_ls)} smiles.")
        self.logger.debug(f"Tokeniser:\n{TOKENIZERS['ChemBERTa']}\nEncoder:\n{MODELS['ChemBERTa']}")
        self.logger.info(f"Pooling strategy: {pooling}")

        (parsed_ids, parsed_smiles, parsed_mols), _ = self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

        embeddings = []
        device = next(self.encoder.parameters()).device

        n_smiles = len(parsed_smiles)
        total_batches = (n_smiles + batch_size - 1) // batch_size
        self.logger.info(f"Generating embeddings in {total_batches} batches")

        for i in range(0, n_smiles, batch_size):
            batch = [
                s for s in parsed_smiles[i: i + batch_size]
            ]
            current_batch_no = i // batch_size + 1

            self.logger.debug(f"SMILES for batch {current_batch_no}:\n{batch}")

            enc = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=max_token_len,
                return_tensors="pt",
                add_special_tokens=True
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                output = self.encoder(**enc)

                if hasattr(output, "last_hidden_state"):
                    hidden = output.last_hidden_state  # (B, L, H)

                    if pooling == "cls":
                        pooled = hidden[:, 0, :]

                    elif pooling == "mean":
                        mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
                        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

                    else:
                        raise RuntimeError(f"Unexpected pooling strategy: {pooling}")

                elif isinstance(output, torch.Tensor):
                    pooled = output

                elif hasattr(output, "pooler_output"):
                    pooled = output.pooler_output

                else:
                    raise RuntimeError("Unknown output structure from encoder")
            
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")

        try:
            emb_array = np.vstack(embeddings)
        except ValueError as e:
            self.logger.error(f"Could not stack the embedding arrays:\n{e}")
            return pd.DataFrame()

        final_df = pd.DataFrame(
            emb_array,
            index=parsed_ids,
            columns=[f"emb_{i}" for i in range(1, emb_array.shape[1] + 1)]
        )

        final_df['SMILES'] = parsed_smiles
        final_df = final_df.add_suffix(f"_chemberta_{pooling}")
        final_df.index.name, final_df.index = "ID", parsed_ids

        if drop_cols:
            final_df = self._drop_columns(df=final_df, min_unique=min_unique)
            self.logger.info(f"Embedding data frame created with shape: {final_df.shape}")

        return final_df

    def calcMolFormer(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            batch_size: int=64,
            max_token_len: int=202,
            drop_cols: bool=False,
            min_unique: int=MIN_UNIQUE,
            pooling: str ="mean"
        ) -> pd.DataFrame:

        """
        Description
        -----------
        Function to calculate MolFormer embeddings for a list of smiles

        Parameters
        ----------
        smiles_ls           list[str]   List of SMILES
        id_ls               list[str]   List of IDs corresponding to each SMILE
        batch_size          int         Size of batch to process
        max_token_len       int         Maximum length of the generated embeddings
        drop_cols           bool        Flag to drop columns with low variance
        min_unique          int         Variance threshold for dopping columns
        pooling             str         Pooling strategy: "cls" or "mean"
        """

        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")

        if len(smiles_ls) != len(id_ls):
                    self.logger.error(
                        f"Length of SMILES and IDs not the same."
                        f"(SMILES = {len(smiles_ls)}, IDs = {len(id_ls)})"
                    )
                    raise ValueError("len(smiles_ls) != len(id_ls)")

        self.logger.info(f"Creating MolFormer embeddings for {len(smiles_ls)} smiles.")
        self.logger.debug(f"Tokeniser:\n{TOKENIZERS['MolFormer']}\nEncoder:\n{MODELS['MolFormer']}")
        self.logger.info(f"Pooling strategy: {pooling}")

        (parsed_ids, parsed_smiles, parsed_mols), _ = self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

        embeddings = []
        device = next(self.encoder.parameters()).device

        n_smiles = len(parsed_smiles)
        total_batches = (n_smiles + batch_size - 1) // batch_size
        self.logger.info(f"Generating embeddings in {total_batches} batches")

        for i in range(0, n_smiles, batch_size):
            batch = [
                s for s in parsed_smiles[i: i + batch_size]
            ]
            current_batch_no = i // batch_size + 1

            self.logger.debug(f"SMILES for batch {current_batch_no}:\n{batch}")

            enc = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=max_token_len,
                return_tensors="pt",
                add_special_tokens=True
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                output = self.encoder(**enc)

                if hasattr(output, "last_hidden_state"):
                    hidden = output.last_hidden_state  # (B, L, H)

                    if pooling == "cls":
                        pooled = hidden[:, 0, :]  # CLS token

                    elif pooling == "mean":
                        attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
                        masked_hidden = hidden * attention_mask
                        summed = masked_hidden.sum(dim=1)
                        counts = attention_mask.sum(dim=1).clamp(min=1)
                        pooled = summed / counts

                elif isinstance(output, torch.Tensor):
                    pooled = output

                elif hasattr(output, "pooler_output"):
                    pooled = output.pooler_output

                else:
                    raise RuntimeError("Unknown output structure from encoder")
            
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")

        try:
            emb_array = np.vstack(embeddings)
        except ValueError as e:
            self.logger.error(f"Could not stack the embedding arrays:\n{e}")
            return pd.DataFrame()

        final_df = pd.DataFrame(
            emb_array,
            index=parsed_ids,
            columns=[f"emb_{i}" for i in range(1, emb_array.shape[1] + 1)]
        )

        final_df['SMILES'] = parsed_smiles
        final_df.index.name, final_df.index = "ID", parsed_ids
        final_df = final_df.add_suffix(f"_molformer_{pooling}")

        if drop_cols:
            final_df = self._drop_columns(df=final_df, min_unique=min_unique)
            self.logger.info(f"Embedding data frame created with shape: {final_df.shape}")

        return final_df
    
    # ====== Batch Feature Calculations
 
    def calcBatchFeatures(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            batch_size: int=2000,
            ignore_3D: bool=False,
            max_token_len: int=400,
            fpath: str | Path = "./",
            compression: str | None=None,
            drop_cols: bool=False,
    ):
        
        """
        Description
        -----------
        Function to calculate features across batches

        Parameters
        ----------
        smiles_ls           list[str]   List of SMILES
        id_ls               list[str]   List of IDs corresponding to each SMILE
        batch_size          int         Size of batch to process
        ignore_3d           bool        Flag to ignore 3D mordred descriptors
        max_token_len       int         Maximum length of the generated embeddings
        fpath               str | Path  Path to save the descriptor datasets to
        compression         str         Type of compression to save dataset as (e.g., "gzip")
        drop_cols           bool        Flag to drop columns with low variance
        """

        self.logger.info(f"Calculating {self.feature_set} descriptors in batches of {batch_size}.")

        if isinstance(fpath, str):
            fpath = Path(fpath)

        n_smiles = len(smiles_ls)
        total_batches = (n_smiles + batch_size - 1) // batch_size

        saved_path_ls = []
        for i in range(0, n_smiles, batch_size):

            current_batch_no = i // batch_size + 1

            self.logger.info(f"Processing batch {current_batch_no}/{total_batches}")
            
            smi_batch = [
                smi for smi in smiles_ls[i : i + batch_size]
            ]

            id_batch = [
                id for id in id_ls[i : i + batch_size]
            ]

            df = self._calculate_features(smi_batch, id_batch, ignore_3D, max_token_len)
            
            fpath_str = str(fpath).replace('*', str(current_batch_no))
            base_path = Path(fpath_str)

            save_path = (
                base_path.with_suffix(".csv")
                if not drop_cols
                else base_path.with_name(base_path.stem + "_tmp.csv")
            )

            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index_label="ID", compression=compression)
            self.logger.info(f"Saved batch {current_batch_no} to:\n{save_path}\n\n")
            
            saved_path_ls.append(save_path)
        
        if drop_cols:
            self.logger.info(f"Determining columns to drop across all batches.")
            cols_to_drop = self._determine_columns_to_drop(saved_path_ls)
            self.logger.info(f"Dropping {len(cols_to_drop)} columns from all batches:\n{cols_to_drop}")

            final_saved_paths = []

            for i, temp_path in enumerate(saved_path_ls, 1):
                df = pd.read_csv(temp_path, index_col="ID")
                df = df.drop(columns=cols_to_drop, errors="ignore")

                final_path = temp_path.parent / temp_path.name.replace("_tmp.csv", ".csv")
                df.to_csv(final_path, index_label="ID", compression=compression)
                final_saved_paths.append(final_path)

                temp_path.unlink()
                self.logger.info(f"Saved final batch {i} with shape {df.shape} to:\n{final_path}")
            
            saved_path_ls = final_saved_paths

        self.logger.info(
            "\n=====================\n"
            " All batches complete"
            "\n=====================\n"
        )

        
        return df

    # ====== Hidden Functions
    def _mol2mol3d(
        self,
        mol: Chem.Mol,
        num_confs: int = 20,
        max_iters: int = 500,
        seed: int = 42,
    ) -> Chem.Mol | None:
        """
        parsed RDKit Mol (2D) and return the same molecule with Hs + an embedded,
        optimized 3D conformer (lowest-energy if multiple conformers are generated).
        """
        if mol is None:
            return None

        # Work on a copy; don't mutate caller's mol
        mol3d = Chem.Mol(mol)

        # Add Hs (needed for sane 3D/forcefield)
        mol3d = Chem.AddHs(mol3d, addCoords=False)

        # ETKDG params (modern distance geometry)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useSmallRingTorsions = True
        params.enforceChirality = True

        # Generate multiple conformers, then pick lowest energy after optimization
        try:
            cids = list(AllChem.EmbedMultipleConfs(mol3d, numConfs=int(num_confs), params=params))
        except Exception:
            return None

        if not cids:
            return None

        # Prefer MMFF; fall back to UFF
        energies = {}
        try:
            results = AllChem.MMFFOptimizeMoleculeConfs(mol3d, maxIters=int(max_iters))
            # results: List[(status, energy)]
            for cid, (status, energy) in zip(cids, results):
                if status == 0:  # converged
                    energies[cid] = float(energy)
        except Exception:
            try:
                results = AllChem.UFFOptimizeMoleculeConfs(mol3d, maxIters=int(max_iters))
                for cid, (status, energy) in zip(cids, results):
                    if status == 0:
                        energies[cid] = float(energy)
            except Exception:
                return None

        # If nothing converged, keep the first conformer (still embedded) or fail
        if not energies:
            # Keep first embedded conformer to be permissive
            keep = cids[0]
        else:
            keep = min(energies, key=energies.get)

        # Drop all but the chosen conformer
        keep_conf = mol3d.GetConformer(int(keep))
        new_mol = Chem.Mol(mol3d)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(keep_conf, assignId=True)

        return new_mol

    def _parse_smiles(
        self,
        smiles_ls: list[str],
        id_ls: list[str]
    ):
        """
        Returns:
        (
        [parsed_ids, parsed_smiles, parsed_mols3d], 
        [failed_parse_ids, failed_parse_smiles], 
        [failed3d_ids, failed3d_smiles]
        )
        
        where parsed_mols3d are RDKit Mol objects containing exactly one (chosen) 3D conformer.
        """
        self.logger.debug("Parsing SMILES...")

        parsed_ids, parsed_smiles, parsed_mols = [], [], []
        failed_ids, failed_smiles = [], []
        failed3d_ids, failed3d_smiles = [], []

        for _id, smi in zip(id_ls, smiles_ls):
            if not isinstance(smi, str):
                smi = str(smi)

            mol2d = Chem.MolFromSmiles(smi)
            if mol2d is None:
                failed_ids.append(_id)
                failed_smiles.append(smi)
                continue

            mol3d = self._mol2mol3d(mol2d)
            if mol3d is None:
                failed3d_ids.append(_id)
                failed3d_smiles.append(smi)
                continue

            parsed_ids.append(_id)
            parsed_smiles.append(smi)
            parsed_mols.append(mol3d)

        if not parsed_mols:
            self.logger.warning("No valid 3D molecules generated.")
            return pd.DataFrame(), [failed_ids, failed_smiles], [failed3d_ids, failed3d_smiles]

        if failed_smiles:
            self.logger.debug(f"{len(failed_ids)} failed parse:\n{failed_smiles}")

        if failed3d_smiles:
            self.logger.debug(f"{len(failed3d_ids)} failed 3D embed:\n{failed3d_smiles}")

        self.logger.debug(f"{len(parsed_smiles)} molecules parsed + 3D-embedded.")

        return (
            [parsed_ids, parsed_smiles, parsed_mols],
            [failed_ids, failed_smiles]
            )

    def _setup_molecule_df(
            self,
            id_ls: list[str],
            smiles_ls: list[str],
            mol_ls: list[Chem.rdchem.Mol],
    ):
        """
        Making a dataframe with smiles, mol objects and IDs
        """

        df = pd.DataFrame()
        df.index.name, df.index = "ID", id_ls
        df["SMILES"] = smiles_ls
        df["Mols"] = mol_ls
        
        return df
    
    def _initialise_embedding_models(
            self,
            tokeniser: str,
            model: str,
    ):
        """
        Initialising the embedding models
        """
        if tokeniser and model:

            self.logger.info(
                "Initialising tokeniser and model"
                f"Tokeniser = {tokeniser}"
                f"Model = {model}"
                )
            
            self.tokeniser = AutoTokenizer.from_pretrained(tokeniser, trust_remote_code=True)
            self.encoder =  AutoModel.from_pretrained(tokeniser, trust_remote_code=True).eval().to("cpu")
        else:
            self.logger.error("No valid tokeniser or model provided:"
                              f"Tokeniser = {tokeniser}",
                              f"Model = {model}"
            )

    def _drop_columns(
            self,
            df: pd.DataFrame,
            min_unique: int=2,
            protected_cols: list[str] = ["SMILES", "ID"]
    ):
        """
        Dropping columns based on low variability (subject to min_unique)
        """
        
        low_var = df.nunique(dropna=True) < min_unique
        has_nan = df.isna().any(axis=0)

        mask = (low_var | has_nan) & ~df.columns.isin(protected_cols)
        to_drop = df.columns[mask]

        df = df.drop(columns=to_drop)

        return df
    
    def _determine_columns_to_drop(
        self,
        file_paths: list[Path],
        min_unique: int = MIN_UNIQUE,
        protected_cols: list[str] = ["SMILES"],
        force_float32: bool=True
    ) -> list[str]:
        """Determine columns to drop using streaming approach"""
        
        column_stats = {}
        f32_max = float(np.finfo(np.float32).max)
        
        for path in file_paths:
            self.logger.info(f"Analyzing {path.name} for problematic columns...")
            
            # Read in chunks to save memory
            for chunk in pd.read_csv(path, index_col="ID", chunksize=500, low_memory=False):
                for col in chunk.columns:
                    if col in protected_cols:
                        continue
                        
                    if col not in column_stats:
                        column_stats[col] = {
                            'unique_values': set(),
                            'has_nan': False,
                            'has_non_numeric': False,
                            'has_inf': False,
                            "removed_non_f32": 0
                        }
                    s = chunk[col]
                    # Check if column contains non-numeric values
                    if not pd.api.types.is_numeric_dtype(s):
                        # Try to convert to numeric
                        numeric_col = pd.to_numeric(s, errors='coerce')
                        # If conversion created NaNs where there weren't any, it's non-numeric
                        if numeric_col.isna().sum() > s.isna().sum():
                            column_stats[col]['has_non_numeric'] = True
                        s = numeric_col
                    
                    # Checking for 'infinite' values
                    if pd.api.types.is_numeric_dtype(s):
                        if np.isinf(s).any():
                            column_stats[col]['has_inf'] = True

                    if force_float32 and pd.api.types.is_numeric_dtype(s):
                        mask_ok = np.isfinite(s) & (s.abs() <= f32_max)
                        removed = int((~mask_ok & s.notna()).sum())
                        if removed:
                            column_stats[col]["removed_non_f32"] += removed
                        s = s.where(mask_ok, np.nan)

                    chunk[col] = s
                    
                    # Track unique values (limit to min_unique to save memory)
                    if len(column_stats[col]['unique_values']) < min_unique:
                        unique_vals = chunk[col].dropna().unique()
                        # Filter out inf values from unique tracking
                        unique_vals = unique_vals[np.isfinite(unique_vals)]
                        column_stats[col]['unique_values'].update(unique_vals.tolist()[:min_unique])
                    
                    # Track NaN presence
                    if chunk[col].isna().any():
                        column_stats[col]['has_nan'] = True
        
        # Determine columns to drop - UPDATE THIS
        cols_to_drop = [
            col for col, stats in column_stats.items()
            if (len(stats['unique_values']) < min_unique or 
                stats['has_nan'] or 
                stats['has_non_numeric'] or
                stats['has_inf'])
        ]
            
        self.logger.info(
            "Columns to drop: %d total\n"
            "  - Low variance: %d\n"
            "  - Has NaN: %d\n"
            "  - Has Inf: %d\n"
            "  - Non-numeric: %d\n"
            "  - Removed non-float32-finite values in kept/dropped cols: %d cols"
            % (
                len(cols_to_drop),
                sum(1 for c in cols_to_drop if len(column_stats[c]['unique_values']) < min_unique),
                sum(1 for c in cols_to_drop if column_stats[c]['has_nan']),
                sum(1 for c in cols_to_drop if column_stats[c]['has_inf']),
                sum(1 for c in cols_to_drop if column_stats[c]['has_non_numeric']),
                sum(1 for c, st in column_stats.items() if st.get('removed_non_f32', 0) > 0),
            )
        )
        
        return cols_to_drop
    
    def _calculate_features(
            self,
            smi_batch: list[str],
            id_batch: list[str],
            ignore_3D: bool,
            max_token_len: int
    ):
        """
        Function to package the feature generation functions
        """
        
        if self.feature_set == "rdkit":
            return self.calcRDKit(
                smiles_ls=smi_batch,
                id_ls=id_batch,
            )

        elif self.feature_set == "mordred":
            return self.calcMordred(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                ignore_3D=ignore_3D
            )

        elif self.feature_set == "chemberta":
            return self.calcChemBERTa(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len
            )
        
        elif self.feature_set == "molformer":
            return self.calcMolFormer(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len
            )
        
        elif self.feature_set == "morgan":
            return (self.calcMorganFingerprints(
                smiles_ls=smi_batch,
                id_ls=id_batch,
            ))
        
        else:
            self.logger.error(f"Feature set ({self.feature_set}) not valid"
                                f"Select from: [rdkit, mordred, chemberta, molformer]")
            raise ValueError(f"Feature set not allowed {self.feature_set}")