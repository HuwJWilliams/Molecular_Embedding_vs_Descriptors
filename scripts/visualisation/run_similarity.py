"""
Script to assess the similarity spaces between different molecular representation types
"""
# region Setup
import argparse
import sys
from pathlib import Path


FILE_PATH = Path(__file__).resolve()
SRC_DIR = FILE_PATH.parents[1] / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
sys.path.insert(0, str(SRC_DIR / "visualisation"))

from get_paths import getPaths
from vis import Visualise

v = Visualise(save_all=False)
paths=getPaths()
feat_paths = paths["full_features"]["fit_lipinski"]
# endregion

# region Adding argument parser
parser = argparse.ArgumentParser()

parser.add_argument(
    "--feature-sets",
    nargs="+",
    choices=["rdkit", "mordred", 
             "chemberta", "chembertasey", "molformer", "molformer-c3-1b", "selformer",
             "maccs", "morgan", 
             ],
    help="Feature sets to calculate similarities for."
)

parser.add_argument(
    "--feature-paths",
    default="fit_lipinski",
    help="Pathing dictionary key for feature data"
)

parser.add_argument(
    "--n-mols",
    type=int,
    default=100,
    help="Number of molecules to calculate similarities for"
)

parser.add_argument(
    "--show-n",
    type=int,
    default=3,
    help="Number of most similar pairs to save structures for"
)

parser.add_argument(
    "--molids",
    nargs="+",
    default=[],
    help="Molecule IDs to generate similarity analysis for"
)

parser.add_argument(
    "--save-dir",
    default=str(FILE_PATH.parents[2] / "results" / "similarity"),
    help="Directory to save results to"
)

args = parser.parse_args()
#endregion

v.plotSimilarities(
    feature_sets=args.feature_sets,
    feature_paths=paths["full_features"][args.feature_paths],
    n_mols=args.n_mols,
    show_top_n_pairs=args.show_n,
    molids=args.molids,
    save_dir=Path(args.save_dir),
)
