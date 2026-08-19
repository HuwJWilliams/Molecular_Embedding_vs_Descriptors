"""Script to run tests to check the full workflow"""

# %% ===== Imports
import pandas as pd
from pathlib import Path
import sys
import importlib

# %% ==== Pathing
ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "run"
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import createPathingJSON, addRawDataPaths, addNewDatasetPaths, loadJSON

# %%
test_json_name = "test.json"

expected_json = {
    "imp_dirs": {
        "proj_dir": "${PROJ_DIR}",
        "scripts_dir": "${SCRIPTS_DIR}",
        "src_dir": "${SRC_DIR}",
        "datasets_dir": "${DATASETS_DIR}",
        "results_dir": "${RESULTS_DIR}",
    },
    "train_test_splits": {},
    "raw_data": {"test": str(RUN_DIR / "test" / "dummy_data.csv")},
    "targets": {"test": "${DATASETS_DIR}/test/dummy_data.csv"},
    "full_features": {"test": {"rdkit": "${DATASETS_DIR}/test/test_rdkit.csv"}},
    "prediction_output_dirs": {
        "rf": {"test": {"rdkit": "${RESULTS_DIR}/TEST_predictions_rf/rdkit"}},
        "cross_feature_predictions": {"test": {}},
        "lipinski_cross_feature_predictions": {"test": {}},
    },
    "dataset_analysis": {},
    "config": {},
}


def testSetup():
    createPathingJSON(json_name=test_json_name)

    addRawDataPaths(
        raw_data_paths=[str(RUN_DIR / "test" / "dummy_data.csv")],
        set_names=["test"],
        json_name=test_json_name,
    )

    addNewDatasetPaths(
        dataset_key="test",
        target_file="dummy_data.csv",
        dataset_prefix="TEST",
        dataset_folder_name="test",
        feature_sets=["rdkit"],
        json_name=test_json_name,
    )

    generated_json = loadJSON(SRC_DIR / "pathing" / test_json_name)

    assert (
        generated_json == expected_json
    ), "FAIL: Generated pathing JSON does not match the expected standard"

    print("PASS: Pathing setup")


testSetup()
