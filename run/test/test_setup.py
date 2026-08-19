"""Script to run tests to check the full workflow"""

# %% ===== Imports
from pathlib import Path
import sys
import pandas as pd

import pytest

# %% ==== Pathing
ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "run"
SCRIPTS_DIR = ROOT / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
FEATURES = [
    "rdkit",
    "mordred",
    "maccs",
    "morgan",
    "chemberta-dc",
    "chemberta-sey",
    "molformer-ibm",
    "molformer-dc",
    "selformer",
]

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import (
    createPathingJSON,
    addRawDataPaths,
    addNewDatasetPaths,
    loadJSON,
    getPaths,
)

# %% ========== PATHING JSON SETUP TESTING
TEST_JSON_NAME = "test.json"
TEST_JSON_PATH = SRC_DIR / "pathing" / TEST_JSON_NAME

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


@pytest.fixture
def clean_test_json():
    TEST_JSON_PATH.unlink(missing_ok=True)
    yield


def test_setup_creates_expected_pathing_json(clean_test_json):
    createPathingJSON(json_name=TEST_JSON_NAME)

    addRawDataPaths(
        raw_data_paths=[str(RUN_DIR / "test" / "dummy_data.csv")],
        set_names=["test"],
        json_name=TEST_JSON_NAME,
    )

    addNewDatasetPaths(
        dataset_key="test",
        target_file="dummy_data.csv",
        dataset_prefix="TEST",
        dataset_folder_name="test",
        feature_sets=FEATURES,
        json_name=TEST_JSON_NAME,
    )

    generated_json = loadJSON(TEST_JSON_PATH)

    assert generated_json == expected_json


# %% ========== FEATURE GENERATION TESTING
sys.path.insert(0, str(SRC_DIR / "datasets"))
FULL_TEST_PATHING = getPaths(TEST_JSON_PATH)

from feature_generator import FeatureGenerator


@pytest.mark.parametrize("feature_set", FEATURES)
def test_generate_features(feature_set):
    input_df = pd.read_csv(FULL_TEST_PATHING["targets"]["test"], index_col="ID")
    output_path = Path(FULL_TEST_PATHING["full_features"]["test"][feature_set])
    output_path.unlink(missing_ok=True)

    fg = FeatureGenerator(feature_set)
    fg.calcBatchFeatures(
        smiles_ls=input_df["SMILES"].tolist(),
        id_ls=input_df.index.to_list(),
        fpath=output_path,
        batch_size=25,
    )

    assert output_path.exists()
    generated_df = pd.read_csv(output_path, index_col="ID")

    assert len(generated_df) > 0
    assert generated_df.index.notna().all()
