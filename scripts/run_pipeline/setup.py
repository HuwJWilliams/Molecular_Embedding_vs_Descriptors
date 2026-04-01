import subprocess
import sys
from pathlib import Path
import json
import argparse

CONFIG_DIR = Path(__file__).parent
SCRIPTS_DIR = CONFIG_DIR.parent
SRC_DR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DR / "pathing"))
from get_paths import createPathingJSON, addRawDataPaths

parser = argparse.ArgumentParser(description="Correctly setting up the project directory")

# Adding arguments to parser
parser.add_argument(
    "--create-pathing", action="store_true", help="Flag to create pathing json for project"
    )

parser.add_argument(
    "--json-name", default="test_paths.json", help="Name to set pathing json"
)

parser.add_argument(
    "--set-paths", 
    nargs="+",
    required=False, 
    help="Adding new raw data set to the project directory"
)

parser.add_argument(
    "--set-names",
    nargs="+",
    required=False,
    help="Set names of the new raw data sets"
)


args = parser.parse_args()

if args.create_pathing:
    createPathingJSON(json_name=args.json_name)


if bool(args.set_names) != bool(args.set_paths):
    raise ValueError("Set both '--set-paths' and '--set-names' to save raw data")

else:
    addRawDataPaths(
        raw_data_paths=args.set_paths,
        set_names=args.set_names,
        json_name=args.json_name
    )