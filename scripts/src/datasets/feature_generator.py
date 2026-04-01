# %% --- Imports
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdFingerprintGenerator, MACCSkeys
from mordred import Calculator, descriptors
from transformers import AutoTokenizer, AutoModel
import torch
import selfies

try:
    import deepchem as dc
except ImportError:
    dc = None

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import setupLogger
sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import TRANSFORMER_FEATURE_SPECS

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
        feature_set     (str)      Name of the feature set to generate (currently supports:
                                    [rdkit, mordred, molformer, chemberta, selformer])
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

        transformer_spec = TRANSFORMER_FEATURE_SPECS.get(self.feature_set)
        if transformer_spec:
            self._initialise_embedding_models(
                tokeniser=transformer_spec["tokeniser"],
                model=transformer_spec["model"],
            )

        if self.encoder != "Uninitialised":
            self.encoder.eval()

        return

    def _require_deepchem(self) -> None:
        """Guard DeepChem-dependent feature paths without breaking other backends."""
        if dc is None:
            raise ImportError(
                "DeepChem is required for this feature set. "
                "Use the tlp_py311 environment or install deepchem in the active env."
            )

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
        
        (parsed_ids, parsed_smiles, parsed_mols), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )

        self.logger.info(f"Creating RDKit descriptors for {len(smiles_ls)} smiles.")
        
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

        (parsed_ids, parsed_smiles, parsed_mols), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )

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
        (parsed_ids, parsed_smiles, parsed_mols), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )

        df = self._setup_molecule_df(
            id_ls=parsed_ids, smiles_ls=parsed_smiles, mol_ls=parsed_mols
            )

        self.logger.info(f"Generating Morgan Fingerprints for {len(df)} valid molecules")

        fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, 
            fpSize=fingerprint_size,
            includeChirality=True,
            useBondTypes=True
            )

        fps = []

        for mol in df["Mols"]:
            fp = fpgen.GetFingerprintAsNumPy(mol)
            fps.append(fp)
        
        fp_df = pd.DataFrame(
            fps,
            index=parsed_ids,
            columns=[f"{i}_maccs" for i in range(len(fps[0]))]
        )
        fp_df.index.name = "ID"

        return fp_df
        

    def calcMACCSKeys(
      self,
      smiles_ls: list[str],
      id_ls: list[str]      
    ):
          
        self.logger.info(f"Creating MACCS Keys fingerprints for {len(smiles_ls)} smiles.")
        (parsed_ids, parsed_smiles, parsed_mols), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )

        df = self._setup_molecule_df(
            id_ls=parsed_ids,
            smiles_ls=parsed_smiles,
            mol_ls=parsed_mols
        )

        self.logger.info(f"Generating MACCS Keys for {len(df)} valid molecules")

        fps = []

        for mol in df["Mols"]:
            fp = MACCSkeys.GenMACCSKeys(mol)
            fps.append(list(fp))

        fp_df = pd.DataFrame(
            fps,
            index=parsed_ids,
            columns=[f"{i}_maccs" for i in range(len(fps[0]))]
        )
        fp_df.index.name = "ID"

        return fp_df
        
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
        Function to calculate DeepChem's ChemBERTa embeddings for a list of smiles

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

        (parsed_ids, parsed_smiles, _), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )

        return self._calc_named_transformer_embeddings(
            input_texts=parsed_smiles,
            ids=parsed_ids,
            batch_size=batch_size,
            max_token_len=max_token_len,
            drop_cols=drop_cols,
            min_unique=min_unique,
            pooling=pooling,
        )

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

        (parsed_ids, parsed_smiles, _), _ = self._check_and_parse_smiles(
            smiles_ls=smiles_ls,
            id_ls=id_ls
        )
        return self._calc_named_transformer_embeddings(
            input_texts=parsed_smiles,
            ids=parsed_ids,
            batch_size=batch_size,
            max_token_len=max_token_len,
            drop_cols=drop_cols,
            min_unique=min_unique,
            pooling=pooling,
        )
    
    def calcSELFormer(
            self,
            smiles_ls: list[str],
            id_ls: list[str],
            batch_size: int=64,
            max_token_len: int=512,
            drop_cols: bool=False,
            min_unique: int=MIN_UNIQUE,
            pooling: str ="mean"
        ) -> pd.DataFrame:
        """
        Description
        -----------
        Function to calculate SELFormer embeddings for a list of smiles

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

        valid_ids, valid_selfies, _ = self._smiles2selfies(
            smiles_ls=smiles_ls, id_ls=id_ls
            )
        
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")
        
        if len(valid_selfies) != len(valid_ids):
            self.logger.error(
                f"Length of SMILES and IDs not the same."
                f"(SMILES = {len(valid_selfies)}, IDs = {len(valid_ids)})"
            )
            raise ValueError("len(smiles_ls) != len(id_ls)")

        return self._calc_named_transformer_embeddings(
            input_texts=valid_selfies,
            ids=valid_ids,
            batch_size=batch_size,
            max_token_len=max_token_len,
            drop_cols=drop_cols,
            min_unique=min_unique,
            pooling=pooling,
        )
        
    def calcSMILESBERT(
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
            Function to calculate SMILES-BERT embeddings for a list of smiles

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
            
            (parsed_ids, parsed_smiles, _), _ = self._check_and_parse_smiles(
                smiles_ls=smiles_ls,
                id_ls=id_ls
            )

            return self._calc_named_transformer_embeddings(
                input_texts=parsed_smiles,
                ids=parsed_ids,
                batch_size=batch_size,
                max_token_len=max_token_len,
                drop_cols=drop_cols,
                min_unique=min_unique,
                pooling=pooling,
            )
    

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

    # ====== Trimming Rows that are outliers
    def trimRowsByPercentile(
            self,
            input_df: str | Path | pd.DataFrame,
            columns: list[str] | None = None,
            percentile: float = 0.99,
            tail: str = "both",
            index_col: str | None = "ID",
            exclude_columns: list[str] | None = None,
            return_removed_rows: bool = False,
        ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove rows whose values fall outside a percentile cutoff.

        This is intended for trimming rows with extreme feature values before model
        training. By default, it removes rows above the 99th percentile for the
        selected numeric columns.

        Parameters
        ----------
        input_df : str | Path | pd.DataFrame
            Input feature table or path to a CSV file.
        columns : list[str] | None, optional
            Columns to evaluate. If None, all numeric columns are used except any
            listed in ``exclude_columns``.
        percentile : float, optional
            Percentile cutoff expressed as a fraction between 0 and 1.
            Default = 0.99.
        tail : str, optional
            Which tail to trim. Choose from ``"upper"``, ``"lower"``, or
            ``"both"``. Default = "upper".
        index_col : str | None, optional
            Column to use as the index when reading a CSV path. Default = "ID".
        exclude_columns : list[str] | None, optional
            Columns to skip when ``columns=None``. Default = None.
        return_removed_rows : bool, optional
            If True, also return the removed rows as a second dataframe.
            Default = False.

        Returns
        -------
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]
            Trimmed dataframe, and optionally the removed rows.
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
                raise ValueError(f"These columns are missing from the input data: {missing}")

        if not columns:
            raise ValueError("No columns available for percentile trimming.")

        keep_mask = pd.Series(True, index=df.index)

        for col in columns:
            s = df[col]

            if not pd.api.types.is_numeric_dtype(s):
                raise TypeError(f"Column '{col}' must be numeric for percentile trimming.")

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


    # ====== Hidden Functions

    def _calc_transformer_embeddings(
                self,
                input_texts: list[str],
                ids: list[str],
                model_label: str,
                suffix_label: str,
                batch_size: int,
                max_token_len: int,
                drop_cols: bool,
                min_unique: int,
                pooling: str,
                metadata_col_name: str,
        ) -> pd.DataFrame:
            """Shared embedding routine for transformer-style text encoders."""

            if pooling not in {"cls", "mean"}:
                raise ValueError("pooling must be either 'cls' or 'mean'.")

            if len(input_texts) != len(ids):
                self.logger.error(
                    f"Length of input texts and IDs not the same."
                    f"(texts = {len(input_texts)}, IDs = {len(ids)})"
                )
                raise ValueError("len(input_texts) != len(ids)")

            self.logger.info(f"Creating {model_label} embeddings for {len(input_texts)} entries.")
            self.logger.debug(f"Tokeniser:\n{self.tokeniser}\nEncoder:\n{self.encoder}")
            self.logger.info(f"Pooling strategy: {pooling}")

            embeddings = []
            device = next(self.encoder.parameters()).device

            n_texts = len(input_texts)
            total_batches = (n_texts + batch_size - 1) // batch_size
            self.logger.info(f"Generating embeddings in {total_batches} batches")

            for i in range(0, n_texts, batch_size):
                batch = [text for text in input_texts[i: i + batch_size]]
                current_batch_no = i // batch_size + 1

                self.logger.debug(f"Inputs for batch {current_batch_no}:\n{batch}")

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
                        hidden = output.last_hidden_state

                        if pooling == "cls":
                            pooled = hidden[:, 0, :]
                        elif pooling == "mean":
                            mask = enc["attention_mask"].unsqueeze(-1).float()
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
                index=ids,
                columns=[f"emb_{i}" for i in range(1, emb_array.shape[1] + 1)]
            )

            final_df[metadata_col_name] = input_texts
            final_df = final_df.add_suffix(f"_{suffix_label}_{pooling}")
            final_df.index.name, final_df.index = "ID", ids

            if drop_cols:
                final_df = self._drop_columns(df=final_df, min_unique=min_unique)
                self.logger.info(f"Embedding data frame created with shape: {final_df.shape}")

            return final_df

    def _calc_named_transformer_embeddings(
        self,
        input_texts: list[str],
        ids: list[str],
        batch_size: int,
        max_token_len: int,
        drop_cols: bool,
        min_unique: int,
        pooling: str,
    ) -> pd.DataFrame:
        """Dispatch to the shared embedding routine using the current feature-set config."""

        spec = TRANSFORMER_FEATURE_SPECS.get(self.feature_set)
        if spec is None:
            raise ValueError(f"No transformer config found for feature set '{self.feature_set}'")

        return self._calc_transformer_embeddings(
            input_texts=input_texts,
            ids=ids,
            model_label=spec["model_label"],
            suffix_label=spec["suffix_label"],
            batch_size=batch_size,
            max_token_len=max_token_len,
            drop_cols=drop_cols,
            min_unique=min_unique,
            pooling=pooling,
            metadata_col_name=spec["metadata_col_name"],
        )
    
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

    def _check_and_parse_smiles(
        self,
        smiles_ls: list[str],
        id_ls: list[str],
    ):
        """Validate SMILES/ID lengths and return parsed IDs, SMILES and molecules."""

        if len(smiles_ls) != len(id_ls):
            self.logger.error(
                f"Length of SMILES and IDs not the same."
                f"(SMILES = {len(smiles_ls)}, IDs = {len(id_ls)})"
            )
            raise ValueError("len(smiles_ls) != len(id_ls)")

        return self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

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
            
            try:
                self.tokeniser = AutoTokenizer.from_pretrained(
                    tokeniser,
                    trust_remote_code=True
                )
                self.encoder = AutoModel.from_pretrained(
                    model,
                    trust_remote_code=True
                ).eval().to("cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to initialise tokenizer/model for '{model}'. "
                    "The configured Hugging Face repository may be incomplete "
                    "(for example missing tokenizer vocab files or weights)."
                ) from exc
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
        
        calculators = {
            "rdkit": lambda: self.calcRDKit(smiles_ls=smi_batch, id_ls=id_batch),
            "mordred": lambda: self.calcMordred(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                ignore_3D=ignore_3D,
            ),
            "chemberta": lambda: self.calcChemBERTa(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "chembertasey": lambda: self.calcChemBERTa(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "molformer": lambda: self.calcMolFormer(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "molformer-c3-1b": lambda: self.calcMolFormer(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "selformer": lambda: self.calcSELFormer(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "smilesbert": lambda: self.calcSMILESBERT(
                smiles_ls=smi_batch,
                id_ls=id_batch,
                batch_size=64,
                max_token_len=max_token_len,
            ),
            "morgan": lambda: self.calcMorganFingerprints(smiles_ls=smi_batch, id_ls=id_batch),
            "maccs": lambda: self.calcMACCSKeys(smiles_ls=smi_batch, id_ls=id_batch),
        }

        try:
            return calculators[self.feature_set]()
        except KeyError as exc:
            raise ValueError(f"Feature set not allowed {self.feature_set}") from exc
        
    def _smiles2selfies(self, 
                        smiles_ls:list[str]=None, 
                        id_ls:list[str]=None, 
                        smiles_df:pd.DataFrame=None, 
                        smiles_col: str="SMILES",
                        id_col:str="ID"):

        if smiles_df is not None:
            smiles_df = smiles_df.copy().reset_index()
            smiles_ls = smiles_df[smiles_col].tolist()
            id_ls = smiles_df[id_col]

        (parsed_ids, parsed_smiles, parsed_mols), _ =  self._parse_smiles(smiles_ls=smiles_ls, id_ls=id_ls)

        valid_selfies = []
        valid_ids = []

        invalid_ids = []

        for mol_id, smi in zip(parsed_ids, parsed_smiles):
            try:
                selfies_str = selfies.encoder(smi)
                valid_selfies.append(selfies_str)
                valid_ids.append(mol_id)
            except Exception as e:
                invalid_ids.append(mol_id)
        
        self.logger.warning(
            f"SMILES unable to be converted to SELFIES:\n{invalid_ids}"
            )
            
        return valid_ids, valid_selfies, invalid_ids
