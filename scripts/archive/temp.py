    def embedSMILESChemBERTa(
            self, 
            smiles_col: str, 
            max_len: int, 
            batch_size: int,
            save_path: Path=None
    ) -> pd.DataFrame:
        
        self.logger.info("Embedding SMILES using ChemBERTa")
        self.logger.debug(f"Tokeniser:\n{TOKENIZER}\nEncoder:\n{MODEL}")
        
        smiles_list = self.unembedded_df[smiles_col].tolist()
        
        # Initialise empty list for embeddings to reside
        embeddings = []

        # Loop over smiles in batches
        self.logger.info("Looping over SMILES in batches")
        total_batches = len(smiles_list) // batch_size + (len(smiles_list) % batch_size > 0)

        for i in range(0, len(smiles_list), batch_size):
            batch = [str(s) for s in smiles_list[i:i+batch_size]]
            current_batch_no = i // batch_size + 1

            # Create tokens from SMILES
            enc = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
                add_special_tokens=True)
            
            # Forward pass through encoder
            with torch.no_grad():
                hidden = self.encoder(**enc).last_hidden_state
                mask   = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))

            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")


        if embeddings:
            arr = np.vstack(embeddings)
        else:
            arr = np.zeros((0, self.encoder.config.hidden_size), np.float32)

        emb_df = pd.DataFrame(
            arr,
            index=self.unembedded_df.index,
            columns=[f"emb_{i}" for i in range(1, arr.shape[1] + 1)]
        )

        self.logger.info(f"Embeddings data frame created with a shape: {emb_df.shape}")
        
        if save_path:
            if save_path.exists():
                self.logger.warning(f"File already exists: {save_path}. Skipping save.")
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                emb_df.to_csv(save_path, index_label='ID')
                self.logger.info(f"Embeddings saved to: {save_path}")
            
        return emb_df
    
    def embedSMILESMolFormer(
        self,
        smiles_col: str,
        batch_size: int,
        save_path: Path=None
    ) -> pd.DataFrame:

        self.logger.info("Embedding SMILES using MolFormer")
        self.logger.debug(f"Tokeniser:\n{self.tokeniser}\nEncoder:\n{self.encoder}")

        smiles_list = self.unembedded_df[smiles_col].tolist()
        embeddings = []

        self.encoder.eval()
        device = next(self.encoder.parameters()).device

        # Loop over smiles in batches
        self.logger.info("Looping over SMILES in batches")
        total_batches = len(smiles_list) // batch_size + (len(smiles_list) % batch_size > 0)

        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            current_batch_no = i // batch_size + 1

            # === Preprocessing step: moved internally ===
            inputs = self.tokeniser(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = self.encoder(**inputs)

                if isinstance(output, torch.Tensor):
                    pooled = output
                elif hasattr(output, "last_hidden_state"):
                    pooled = output.last_hidden_state[:, 0, :]
                elif hasattr(output, "pooler_output"):
                    pooled = output.pooler_output
                else:
                    raise RuntimeError("Unknown output structure from encoder")

            embeddings.append(pooled.cpu().numpy().astype(np.float32))
            self.logger.info(f"Processed batch {current_batch_no} of {total_batches}")

        arr = np.vstack(embeddings)
        emb_df = pd.DataFrame(
            arr,
            index=self.unembedded_df.index,
            columns=[f"emb_{i}" for i in range(1, arr.shape[1] + 1)]
        )

        if save_path:
            if save_path.exists():
                self.logger.warning(f"File already exists: {save_path}. Skipping save.")
            else:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                emb_df.to_csv(save_path, index_label='ID')
                self.logger.info(f"Embeddings saved to: {save_path}")

        self.logger.info(f"Embeddings data frame created with shape: {emb_df.shape}")
        return emb_df

    def calculateDescriptors(
            self, 
            smiles: pd.Series,
            descriptor_set: str="rdkit",
            ignore_3D = True,
            drop_constant: bool=True,
            drop_fragments: bool=False,
            drop_classes: bool=False,
        ) -> pd.DataFrame:

        descriptor_set = descriptor_set.lower()
        self.logger.info(f"Creating {descriptor_set} descriptors")

        parsed = []
        n_processed = 0
        self.logger.debug(f"Total length of smiles input: {len(smiles)}")
        for mol_id, smi in smiles.items():
            n_processed +=1
            mol = Chem.MolFromSmiles(str(smi))

            if mol is not None:
                parsed.append((mol_id, smi, mol))
                print(f"SMILE n {n_processed} Parsed: {smi}")
            if mol is None:
                print(f"ERROR: SMILE n {n_processed} NOT Parsed: {smi}")

        if not parsed:
            return pd.DataFrame(columns=["SMILES"])
    
        ids = [p[0] for p in parsed]
        smis = [p[1] for p in parsed]
        mols = [p[2] for p in parsed]

        print(f"ID, SMILES, MOLS length: {len(ids)}, {len(smis)}, {len(mols)}")
        self.logger.debug(f"Total length of smiles output: {len(smis)}")

        if descriptor_set == "rdkit":
            self.logger.info("Calculating RDKit descriptors...")

            descs = Descriptors.descList
            desc_names = [d[0] for d in descs]
            desc_funcs = [d[1] for d in descs]

            rows = []
            for mol_id, smi, mol in zip(ids, smis, mols):
                vals = []
                for f in desc_funcs:
                    try:
                        v = f(mol)
                        # Robust finiteness check (handles None / non-numerics)
                        try:
                            if not np.isfinite(v):
                                v = pd.NA
                        except Exception:
                            v = pd.NA
                    except Exception:
                        v = pd.NA
                    vals.append(v)
                rows.append({"ID": mol_id, "SMILES": smi, **dict(zip(desc_names, vals))})

            df = pd.DataFrame(rows).set_index("ID")

            # Coerce descriptors to numeric; keep NaNs (they’ll be “NA” on disk with na_rep="NA")
            desc_cols = [c for c in df.columns if c != "SMILES"]
            df[desc_cols] = (
                df[desc_cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
            df.index.name = "ID"

            # Optional: shrink memory (nullable Float32 keeps NaNs nicely)
            try:
                df[desc_cols] = df[desc_cols].astype("Float32")
            except Exception:
                pass

        elif descriptor_set == "mordred":
            self.logger.info("Calculating Mordred descriptors...")

            # Build calculator (optionally exclude heavy 3D families)
            if ignore_3D:
                self.logger.info("Ignoring 3D descriptors")
                exclude = {"RDF", "WHIM", "MoRSE", "GETAWAY"}
                all_descs = Calculator(descriptors, ignore_3D=True).descriptors
                descs = [d for d in all_descs if d.__module__.split(".")[-1] not in exclude]
                calc = Calculator(descs, ignore_3D=True)
            else:
                calc = Calculator(descriptors, ignore_3D=False)
                descs = list(calc.descriptors)

            print(f"Length of descriptors: {len(descs)}")
            self.logger.debug(f"Length of descriptor set: {len(descs)}")

            names = [str(d) for d in calc.descriptors]
            n_desc = len(names)

            rows = []
            for idx, (mol_id, smi, mol) in enumerate(zip(ids, smis, mols)):
                self.logger.debug(f"\nProcessing ID: {mol_id}\nSMILES:{smi}")
                try:
                    res = calc(mol)
                    vals = list(res)

                    # normalize length (usually already n_desc)
                    if len(vals) < n_desc:
                        vals += [pd.NA] * (n_desc - len(vals))
                    elif len(vals) > n_desc:
                        vals = vals[:n_desc]

                    # replace None/NaN with pd.NA
                    def _to_na(v):
                        if v is None:
                            return pd.NA
                        if isinstance(v, float) and math.isnan(v):
                            return pd.NA
                        return v
                    vals = [_to_na(v) for v in vals]

                except Exception as e:
                    self.logger.debug(f"ERROR for ID {mol_id}: {e}")
                    vals = [pd.NA] * n_desc

                rows.append([mol_id, smi, *vals])
                print(f"Total length of rows:{len(rows)} before gc")

                mol = None
                if (idx + 1) % 50 == 0:
                    gc.collect()
                print(f"Total length of rows:{len(rows)} after gc")

            # Build DataFrame with stable schema
            df = pd.DataFrame(rows, columns=["ID", "SMILES"] + names).set_index("ID")
            print(f"Length of df: {len(df)}")

            # Coerce descriptors to numeric; keep NaNs (they'll be "NA" on disk if na_rep="NA")
            desc_cols = names
            df[desc_cols] = (
                df[desc_cols]
                .apply(pd.to_numeric, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )

            # Optional: reduce memory footprint
            try:
                df[desc_cols] = df[desc_cols].astype(np.float32)
            except Exception:
                pass

            # Cleanup
            del rows, names, descs, calc
            gc.collect()

            
        else:
            self.logger.error("Argument 'descriptor_set' not valid. This version only accepts 'rdkit' or 'mordred'.")
            raise ValueError("Descriptor set not supported")

        if drop_constant:
            self.logger.debug("Dropping columns with constant values")
            const_cols = [c for c in df.columns if c != "SMILES" and df[c].nunique() <= 1]
            if const_cols:
                df = df.drop(columns=const_cols, errors="ignore")

        if descriptor_set == "rdkit" and drop_fragments:
            self.logger.debug("Dropping fragment columns")
            frag_cols = [c for c in df.columns if c.startswith("fr_")]
            if frag_cols:
                df = df.drop(columns=frag_cols, errors="ignore")

        if drop_classes:
            self.logger.debug("Dropping classification-type columns (non-numeric or binary)")

            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            non_numeric_cols = [c for c in non_numeric_cols if c != "SMILES"]

            binary_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if df[c].nunique(dropna=True) == 2
            ]

            class_cols = non_numeric_cols + binary_cols

            if class_cols:
                self.logger.debug(f"Dropping columns: {class_cols}")
                df = df.drop(columns=class_cols, errors="ignore")

        return df
    
    def calculateDescriptorsRDKit(
        self,
        smiles: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute RDKit descriptors for a batch of SMILES.
        No column dropping here; pure computation + numeric coercion.
        Returns a DataFrame indexed by ID with columns: SMILES + RDKit descriptors.
        """
        self.logger.info("Creating rdkit descriptors")

        parsed = []
        self.logger.debug(f"Total length of smiles input: {len(smiles)}")
        for mol_id, smi in smiles.items():
            mol = Chem.MolFromSmiles(str(smi))
            if mol is not None:
                parsed.append((mol_id, smi, mol))
            else:
                self.logger.warning(f"Could not parse SMILES for ID={mol_id}: {smi}")

        if not parsed:
            return pd.DataFrame(columns=["SMILES"])

        ids  = [p[0] for p in parsed]
        smis = [p[1] for p in parsed]
        mols = [p[2] for p in parsed]

        descs = Descriptors.descList
        desc_names = [d[0] for d in descs]
        desc_funcs = [d[1] for d in descs]

        rows = []
        for mol_id, smi, mol in zip(ids, smis, mols):
            vals = []
            for f in desc_funcs:
                try:
                    v = f(mol)
                    try:
                        if not np.isfinite(v):
                            v = pd.NA
                    except Exception:
                        v = pd.NA
                except Exception:
                    v = pd.NA
                vals.append(v)
            rows.append({"ID": mol_id, "SMILES": smi, **dict(zip(desc_names, vals))})

        df = pd.DataFrame(rows).set_index("ID")

        # Coerce to numeric; keep NaNs
        desc_cols = [c for c in df.columns if c != "SMILES"]
        df[desc_cols] = (
            df[desc_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )
        df.index.name = "ID"

        try:
            df[desc_cols] = df[desc_cols].astype("Float32")
        except Exception:
            pass

        return df
    
    def calculateDescriptorsMordred(
        self,
        smiles: pd.Series,
        ignore_3D: bool = True,
    ) -> pd.DataFrame:
        """
        Compute Mordred descriptors for a batch of SMILES.
        No column dropping here; pure computation + numeric coercion.
        Returns a DataFrame indexed by ID with columns: SMILES + Mordred descriptors.
        """
        self.logger.info("Creating mordred descriptors")

        # Build Mordred calculator once
        calc = Calculator(descriptors, ignore_3D=ignore_3D)
        names = [str(d) for d in calc.descriptors]
        n_desc = len(names)

        rows = []
        for i, (mol_id, smi) in enumerate(smiles.items(), 1):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                # keep row with NaNs so downstream alignment stays stable
                rows.append([mol_id, str(smi)] + [pd.NA] * n_desc)
                continue

            try:
                vals = list(calc(mol))  # descriptor order matches `names`
            except Exception:
                vals = [pd.NA] * n_desc

            # normalize length defensively
            if len(vals) != n_desc:
                if len(vals) < n_desc:
                    vals += [pd.NA] * (n_desc - len(vals))
                else:
                    vals = vals[:n_desc]

            rows.append([mol_id, str(smi)] + vals)

            if i % 50 == 0:
                gc.collect()

        # Build DataFrame in one shot
        df = pd.DataFrame(rows, columns=["ID", "SMILES"] + names).set_index("ID")

        # Batch numeric coercion (keeps NaNs)
        desc_cols = names
        df[desc_cols] = df[desc_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

        # Optional: reduce memory
        try:
            df[desc_cols] = df[desc_cols].astype("Float32")
        except Exception:
            pass

        # Cleanup
        del rows, calc
        gc.collect()

        return df
    
    def _batch_path_for(self, base_path: Path, batch_no: int) -> Path:
        stem, suffix = base_path.stem, (base_path.suffix or ".csv")
        return base_path.with_name(f"{stem}_{batch_no}{suffix}")

    def _find_existing_batch_files(self, base_path: Path) -> list[Path]:
        stem, suffix = base_path.stem, (base_path.suffix or ".csv")
        pat = re.compile(rf"^{re.escape(stem)}_(\d+){re.escape(suffix)}$")
        files = [p for p in base_path.parent.glob(f"{stem}_*{suffix}") if pat.match(p.name)]
        files.sort(key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name).group(1)))
        return files
        
    def calculateDescriptorsInBatches(
        self,
        smiles: pd.Series,
        csv_path: str | Path,
        descriptor_set: str,
        batch_size: int = 5000,
        ignore_3D: bool = True,
    ) -> list[Path]:
        """
        Compute descriptors in batches and write EACH batch to its own CSV:
        <stem>_1.csv, <stem>_2.csv, ...

        Returns: list of Paths written in THIS call (existing files untouched).
        """
        if smiles is None or smiles.empty:
            self.logger.error("Arg 'smiles' is empty. Nothing to do.")
            return []

        descriptor_set = descriptor_set.lower()
        base_path = Path(csv_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Resume: discover existing batch files and skip their SMILES
        existing_batches = self._find_existing_batch_files(base_path)
        processed_smiles: set[str] = set()
        for f in existing_batches:
            try:
                for ch in pd.read_csv(f, usecols=["SMILES"], chunksize=200_000):
                    processed_smiles.update(ch["SMILES"].astype(str).str.strip())
            except Exception as e:
                self.logger.warning(f"Could not read SMILES from {f}: {e}")

        # Filter input to keep only NEW SMILES (string, stripped)
        smiles = smiles.astype(str).str.strip()
        initial_n = len(smiles)
        if processed_smiles:
            smiles = smiles[~smiles.isin(processed_smiles)]
        N = len(smiles)

        if N == 0:
            self.logger.info(f"Nothing new to compute (incoming={initial_n}, already have={len(processed_smiles)}).")
            return []
        
        ids = smiles.index.to_numpy()
        vals = smiles.values
        n_batches = math.ceil(N / batch_size)

        # Next batch number continues after the last existing file
        start_batch_no = (int(re.search(r"_(\d+)\.csv$", existing_batches[-1].name).group(1)) + 1) if existing_batches else 1
        self.logger.info(f"New work: {N} of {initial_n} → {n_batches} batch(es) @ {batch_size}; starting #{start_batch_no}")

        written: list[Path] = []
        total_rows_written = 0
        pbar = tqdm(range(n_batches), desc="Batches", unit="batch", total=n_batches, dynamic_ncols=True, leave=True)

        for bi in pbar:
            start = bi * batch_size
            stop  = min((bi + 1) * batch_size, N)
            batch_no = start_batch_no + bi
            out_file = self._batch_path_for(base_path, batch_no)

            batch = pd.Series(vals[start:stop], index=ids[start:stop], name="SMILES")

            try:
                if descriptor_set == "rdkit":
                    df_batch = self.calculateDescriptorsRDKit(batch)
                elif descriptor_set == "mordred":
                    df_batch = self.calculateDescriptorsMordred(batch, ignore_3D=ignore_3D)
                else:
                    raise ValueError("Descriptor set not supported (use 'rdkit' or 'mordred').")

                if df_batch.empty:
                    self.logger.warning(f"[Batch {batch_no}] empty output; skipped writing.")
                    pbar.set_postfix({"last_rows": 0, "total_rows": total_rows_written}, refresh=False)
                    continue

                # Write this batch as its own file (header included)
                df_batch.to_csv(out_file, index_label="ID")
                written.append(out_file)

                total_rows_written += len(df_batch)
                self.logger.info(f"[Batch {batch_no}] wrote {len(df_batch)} rows → {out_file}")
                pbar.set_postfix({"last_rows": len(df_batch), "total_rows": total_rows_written}, refresh=False)

            except Exception as e:
                self.logger.exception(f"[Batch {batch_no}] failed: {e}")
                pbar.set_postfix({"last_rows": "ERR", "total_rows": total_rows_written}, refresh=False)
            finally:
                try: del df_batch
                except NameError: pass
                del batch
                gc.collect()

        self.logger.info(f"Completed. Wrote {len(written)} batch file(s), {total_rows_written} rows total.")
        return written      

    def removeUnuniqueColumns(
        self,
        csv_path: str | Path,
        drop_if_missing: bool = True,
        min_unique: int = 2,
        protected_cols: set[str] | None = None,
        chunksize: int = 1000,
    ) -> list[str]:
        """
        Streaming two-pass prune:
        - Drop any non-protected column with ANY missing value (if enabled)
        - Drop any non-protected column with < min_unique unique values
        Writes back atomically. Returns kept columns.
        """
        from uuid import uuid4
        csv_path = Path(csv_path)

        if not csv_path.exists():
            self.logger.error(f"CSV not found: {csv_path}")
            return []

        protected = {"ID", "SMILES"}
        if protected_cols:
            protected |= set(protected_cols)

        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        candidate = [c for c in header_cols if c not in protected]
        if not candidate:
            self.logger.info("No candidate columns to prune.")
            return header_cols

        any_missing = {c: False for c in candidate}
        uniq_seen = {c: set() for c in candidate}

        # Pass 1: profile
        for ch in pd.read_csv(csv_path, chunksize=chunksize):
            for c in candidate:
                s = ch[c]
                if drop_if_missing and s.isna().any():
                    any_missing[c] = True
                if len(uniq_seen[c]) < min_unique:
                    for v in pd.unique(s.dropna().values):
                        uniq_seen[c].add(v)
                        if len(uniq_seen[c]) >= min_unique:
                            break

        miss_cols = {c for c, m in any_missing.items() if m} if drop_if_missing else set()
        low_card  = {c for c in candidate if len(uniq_seen[c]) < min_unique}
        drop_set  = miss_cols | low_card
        if not drop_set:
            self.logger.info("No columns qualified for pruning.")
            return header_cols

        keep_cols = [c for c in header_cols if c not in drop_set]
        tmp_path = csv_path.with_suffix(csv_path.suffix + f".{uuid4().hex}.tmp")

        # Pass 2: rewrite
        first = True
        for ch in pd.read_csv(csv_path, chunksize=chunksize, usecols=keep_cols):
            ch.to_csv(tmp_path, mode="w" if first else "a", header=first, index=False)
            first = False

        tmp_path.replace(csv_path)
        self.logger.info(f"Pruned {len(drop_set)} columns. Kept {len(keep_cols)} columns.")
        return keep_cols
