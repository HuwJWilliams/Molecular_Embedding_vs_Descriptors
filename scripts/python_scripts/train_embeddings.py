"""Make embeddings for molecules"""
from models.transfer_model import TL
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pandas as pd


# === Constants
FILE_DIR        = Path(__file__).resolve()
PROJ_DIR        = FILE_DIR.parents[1]
TOKENIZERS      = ["ibm/MoLFormer-XL-both-10pct", "DeepChem/ChemBERTa-100M-MLM"]
MODELS          = [ "ibm/MoLFormer-XL-both-10pct", "DeepChem/ChemBERTa-100M-MLM"]
DATA_PATH       = Path(PROJ_DIR / "datasets"/"LD50"/"ld50_rat-mouse_oral_lt_9500.csv")
SAVE_PATH       = Path(PROJ_DIR / "datasets"/"embeddings"/"chemberta_embeddings.csv")

# === Read Data
df = pd.read_csv(DATA_PATH, index_col="ID")

# === Run Workflow
model = TL(
    tokeniser = AutoTokenizer.from_pretrained(TOKENIZERS[1], trust_remote_code=True),
    encoder   = AutoModel.from_pretrained(MODELS[1], trust_remote_code=True).eval().to("cpu"),
    unembedded_df = df,
    log_identifier = "embed_chemberta"
)

model.runWorkflow(
    smiles_col="SMILES",
    max_len=400,
    batch_size=64,
    test_frac=0.3,
    model="chemberta",
    save_embeddings_path=SAVE_PATH
)
