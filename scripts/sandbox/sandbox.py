from pathlib import Path
import pandas as pd
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths


paths = getPaths()

fingerprints = paths["full_features"]["all"]
for key, path in fingerprints.items():
    print(key)
    print(len(pd.read_csv(path, index_col=0).columns))