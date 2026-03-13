from pathlib import Path
import json

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"
DATASETS_DIR = PROJ_DIR / "datasets"
RESULTS_DIR = PROJ_DIR / 'results'

def loadJSON(
        json_path: str | Path = FILE_DIR / "paths.json"
):
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    if json_path.stat().st_size == 0:
        raise ValueError(f"JSON file is empty: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def saveJSON(
        json_content: dict, 
        json_path: str | Path = FILE_DIR / "paths.json"
):
    json_path = Path(json_path)
    with open(json_path, 'w', encoding="utf-8") as f:
        json.dump(json_content, f , indent=2)


def getPaths():
    paths = loadJSON()

    replacements = {
        "${PROJ_DIR}": str(PROJ_DIR),
        "${SCRIPTS_DIR}": str(SCRIPTS_DIR),
        "${DATASETS_DIR}": str(DATASETS_DIR),
        "${RESULTS_DIR}": str(RESULTS_DIR)
    }

    def replace_paths(obj):
        if isinstance(obj, dict):
            return {k: replace_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [replace_paths(v) for v in obj]
        elif isinstance(obj, str):
            for key, val in replacements.items():
                obj = obj.replace(key, val)
            return Path(obj)
        return obj

    return replace_paths(paths)

def addNewDatasetPaths(
        dataset_key: str,
        target_file: str,
        dataset_prefix: str,
        dataset_folder_name: str
):
    json_contents = loadJSON()

    dataset_key = dataset_key.lower()

    # Making Target Path
    json_contents["targets"][dataset_key] = (
        f"${{DATASETS_DIR}}/{dataset_folder_name}/{target_file}"
    )

    # Making full feature Path
    json_contents.setdefault("full_features", {})
    json_contents["full_features"][dataset_key] = {}

    # Making Inner feature families
    json_contents["full_features"][dataset_key] = {}

    template = json_contents["full_features"]["bp"]

    for feature, path in template.items():

        if "descriptors" in path:
            family = "descriptors"
            suffix = "descriptors"
        elif "embeddings" in path:
            family = "embeddings"
            suffix = "embeddings"
        elif "fingerprints" in path:
            family = "fingerprints"
            suffix = "fingerprints"
        else:
            raise ValueError(f"Cannot infer folder type for {feature}")

        json_contents["full_features"][dataset_key][feature] = (
            f"${{DATASETS_DIR}}/{family}/{dataset_prefix}_{suffix}/{dataset_key}_{feature}_*.csv"
        )

    # Making Prediction Directories
    for model in ["rf", "lr"]:

        json_contents["prediction_output_dirs"][model][dataset_key] = {}

        for feature in template.keys():

            json_contents["prediction_output_dirs"][model][dataset_key][feature] = (
                f"${{RESULTS_DIR}}/{dataset_prefix}_predictions_{model}/{feature}/{feature}_pred_{dataset_key}"
            )

    saveJSON(json_contents, str(FILE_DIR / "paths_test.json"))


def addFeatureSetPaths(
        feature_name:str,
        family:str,
        all_filename:str=None
):

    json_contents = loadJSON()
    feature_name = feature_name.lower()

    dataset_prefix_map = {
        "bp": "BP",
        "logd": "LOGD",
        "pka": "PKA",
        "ld50": "LD50",
        "pic50": "pIC50",
    }

    # -------------------
    # Full features
    # -------------------
    for dataset, block in json_contents["full_features"].items():
        if dataset == "all":
            filename = all_filename or f"all_{feature_name}.csv"
            block[feature_name] = f"${{DATASETS_DIR}}/all/{filename}"
            continue

        prefix = dataset_prefix_map.get(dataset, dataset.upper())

        block[feature_name] = (
            f"${{DATASETS_DIR}}/{family}/{prefix}_{family}/{dataset}_{feature_name}_*.csv"
        )

    # -------------------
    # Prediction outputs
    # -------------------
    for model in ["rf", "lr"]:
        for dataset, block in json_contents["prediction_output_dirs"][model].items():
            prefix = dataset_prefix_map.get(dataset, dataset.upper())

            # prediction folders use PIC50 not pIC50 in your current JSON
            result_prefix = prefix.upper() if prefix == "pIC50" else prefix

            block[feature_name] = (
                f"${{RESULTS_DIR}}/{result_prefix}_predictions_{model}/{feature_name}/{feature_name}_pred_{dataset}"
            )

    # -------------------
    # Dataset analysis
    # -------------------
    json_contents["dataset_analysis"]["descriptor_analysis"][feature_name] = (
        f"${{DATASETS_DIR}}/all/descriptor_analysis/{feature_name}.csv"
    )

    saveJSON(json_contents)
