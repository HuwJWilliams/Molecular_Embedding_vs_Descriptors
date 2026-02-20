import pandas as pd
import sys
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent / "path"))
from get_paths import getPaths

paths = getPaths()
feat_paths = paths["full_features"]
aligned_paths = paths["aligned_features"]
all_desc = feat_paths['all']
desc = "rdkit"

for (fkey, fval), (akey, aval) in zip(feat_paths.items(), aligned_paths.items()):
    print(fkey)
    fdf = pd.read_csv(str(fval[desc]).replace("*", "1"), index_col="ID")
    adf = pd.read_csv(str(aval[desc]).replace("*", "1"), index_col="ID")


    print("full_feat shape:", fdf.shape)
    print("align_feat shape:", adf.shape)
    print("\n")