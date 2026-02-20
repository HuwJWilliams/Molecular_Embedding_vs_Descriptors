from models.transfer_model import TL
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from pathlib import Path
import argparse
import sys
from path.get_paths import getPaths

data_paths = getPaths()

TOKENIZERS = {
    "molformer": "ibm/MoLFormer-XL-both-10pct",
    "chemberta": "DeepChem/ChemBERTa-100M-MLM",
}
MODELS = TOKENIZERS.copy()

# --- CLI
p = argparse.ArgumentParser(description="Generate descriptors/embeddings for a target dataset.")
p.add_argument("--task", required=True, choices=list(data_paths["targets"].keys()))
p.add_argument("--feature-set", required=True, dest="feature_set",
               choices=["rdkit", "mordred", "chemberta", "molformer"])
p.add_argument("--batch-size", type=int, default=1000,
               help="Batch size for RDKit/Mordred; Mordred may cap internally for memory.")
p.add_argument("--no-ignore-3d", action="store_true",
               help="For Mordred: include 3D descriptors (requires conformers).")
# pruning controls
pm = p.add_mutually_exclusive_group()
pm.add_argument("--prune-missing", dest="prune_missing", action="store_true", default=True,
                help="Drop non-protected columns that contain any missing values (default).")
pm.add_argument("--no-prune-missing", dest="prune_missing", action="store_false",
                help="Do not drop columns based on missingness.")
p.add_argument("--min-unique", type=int, default=2,
               help="Drop non-protected columns with fewer than this many unique non-NA values (default: 2).")
p.add_argument("--protect", action="append", default=[],
               help="Extra protected columns to NEVER drop (can be passed multiple times).")

args = p.parse_args()

# --- Paths
in_path = data_paths["targets"][args.task]
outs_for_task = data_paths["features"][args.task]
if args.feature_set not in outs_for_task:
    sys.exit(f"No output path for ({args.task}, {args.feature_set}).")

out_path = outs_for_task[args.feature_set]
out_path.parent.mkdir(parents=True, exist_ok=True)

print(f"Loading input: {in_path}")
df = pd.read_csv(in_path, index_col="ID")

feature_set = args.feature_set.lower()
ignore_3d = not args.no_ignore_3d
batch_size = args.batch_size

# --- Run generator
model = TL(
    tokeniser=None,
    encoder=None,
    unembedded_df=df if feature_set in ("chemberta", "molformer") else None,
    log_identifier=f"{'embed' if feature_set in ('chemberta','molformer') else 'desc_calc'}_{args.task}_{feature_set}",
)

if feature_set in ("rdkit", "mordred"):
    # Writes to out_path internally; returns a small preview
    preview = model.calculateDescriptorsInBatches(
        smiles=df["SMILES"],
        csv_path=out_path,
        descriptor_set=feature_set,
        batch_size=batch_size,
        ignore_3D=ignore_3d,
    )
    if isinstance(preview, pd.DataFrame) and not preview.empty:
        print("Preview of written file:")
        with pd.option_context("display.max_columns", 8, "display.width", 120):
            print(preview.head())
    print(f"Wrote/updated descriptor file at: {out_path}")

elif feature_set == "chemberta":
    print("Initializing ChemBERTa…")
    tok = AutoTokenizer.from_pretrained(TOKENIZERS["chemberta"], trust_remote_code=True)
    enc = AutoModel.from_pretrained(MODELS["chemberta"], trust_remote_code=True).eval().to("cpu")
    model.tokeniser, model.encoder = tok, enc

    emb = model.embedSMILESChemBERTa(
        smiles_col="SMILES",
        max_len=400,
        batch_size=64,
        save_path=out_path,
    )
    if isinstance(emb, pd.DataFrame) and not emb.empty:
        emb.to_csv(out_path, index_label="ID")
    print(f"Wrote: {out_path}")

elif feature_set == "molformer":
    print("Initializing MoLFormer…")
    tok = AutoTokenizer.from_pretrained(TOKENIZERS["molformer"], trust_remote_code=True)
    enc = AutoModel.from_pretrained(MODELS["molformer"], trust_remote_code=True).eval().to("cpu")
    model.tokeniser, model.encoder = tok, enc

    emb = model.embedSMILESMolFormer(
        smiles_col="SMILES",
        batch_size=64,
        save_path=out_path,
    )
    if isinstance(emb, pd.DataFrame) and not emb.empty:
        emb.to_csv(out_path, index_label="ID")
    print(f"Wrote: {out_path}")

else:
    sys.exit(f"Unknown feature_set '{feature_set}'")

# --- Final prune step (always run at the end)
protected = set(args.protect) if args.protect else set()
kept = model.removeUnuniqueColumns(
    csv_path=out_path,
    drop_if_missing=args.prune_missing,
    min_unique=args.min_unique,
    protected_cols=protected,
    chunksize=1000,
)

print(f"Pruning complete. Kept {len(kept)} columns.")
