from pathlib import Path
import json

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[2]
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
DATASETS_DIR = PROJ_DIR / "datasets"
RESULTS_DIR = PROJ_DIR / "results"


def createPathingJSON(json_name="test_paths.json"):
    pathing_json = {
        "imp_dirs": {
            "proj_dir": "${PROJ_DIR}",
            "scripts_dir": "${SCRIPTS_DIR}",
            "src_dir": "${SRC_DIR}",
            "datasets_dir": "${DATASETS_DIR}",
            "results_dir": "${RESULTS_DIR}",
        },
        "train_test_splits": {},
        "raw_data": {},
        "targets": {},
        "full_features": {},
        "prediction_output_dirs": {
            "rf": {},
            "cross_feature_predictions": {},
            "lipinski_cross_feature_predictions": {},
        },
        "dataset_analysis": {},
        "config": {},
    }

    with open(str(FILE_DIR / json_name), "w", encoding="utf=8") as f:
        json.dump(pathing_json, f, indent=2)


def loadJSON(json_path: str | Path = FILE_DIR / "paths.json"):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def saveJSON(json_content: dict, json_path: str | Path = FILE_DIR / "paths.json"):
    json_path = Path(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, indent=2)


def getPaths(json_path: str | Path = FILE_DIR / "paths.json"):
    paths = loadJSON(json_path)

    replacements = {
        "${PROJ_DIR}": str(PROJ_DIR),
        "${SCRIPTS_DIR}": str(SCRIPTS_DIR),
        "${SRC_DIR}": str(SRC_DIR),
        "${DATASETS_DIR}": str(DATASETS_DIR),
        "${RESULTS_DIR}": str(RESULTS_DIR),
    }

    def replace_paths(obj):
        if isinstance(obj, dict):
            return {k: replace_paths(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [replace_paths(v) for v in obj]
        if isinstance(obj, str):
            replaced = obj
            for key, val in replacements.items():
                replaced = replaced.replace(key, val)
            if "${" in obj:
                return Path(replaced)
            return obj
        return obj

    return replace_paths(paths)


def _dataset_feature_prefix(dataset_key: str, for_results: bool = False) -> str:
    dataset_key = dataset_key.lower()

    feature_prefix_map = {
        "bp": "BP",
        "logd": "LOGD",
        "pka": "PKA",
        "ld50": "LD50",
        "pic50": "pIC50",
    }

    result_prefix_map = {
        "bp": "BP",
        "logd": "LOGD",
        "pka": "PKA",
        "ld50": "LD50",
        "pic50": "PIC50",
    }

    if for_results:
        return result_prefix_map.get(dataset_key, dataset_key.upper())
    return feature_prefix_map.get(dataset_key, dataset_key.upper())


def _add_dataset_cross_prediction_paths(
    prediction_output_dirs: dict,
    dataset_key: str,
    identifiers: list[str],
):
    cross_block = prediction_output_dirs.setdefault("cross_feature_predictions", {})
    lipinski_cross_block = prediction_output_dirs.setdefault(
        "lipinski_cross_feature_predictions", {}
    )

    cross_block.setdefault(dataset_key, {})
    lipinski_cross_block.setdefault(dataset_key, {})

    for identifier in identifiers:
        cross_block[dataset_key][
            identifier
        ] = f"${{RESULTS_DIR}}/cross_feature_predictions/{dataset_key}/{identifier}"
        lipinski_cross_block[dataset_key][
            identifier
        ] = f"${{RESULTS_DIR}}/lipinski_cross_feature_predictions/{dataset_key}/{identifier}"


def addNewDatasetPaths(
    dataset_key: str,
    target_file: str,
    dataset_prefix: str,
    dataset_folder_name: str,
    json_path: str | Path = FILE_DIR / "paths.json",
):
    json_contents = loadJSON(json_path)
    dataset_key = dataset_key.lower()
    dataset_prefix = dataset_prefix.strip()

    # targets
    json_contents["targets"][
        dataset_key
    ] = f"${{DATASETS_DIR}}/{dataset_folder_name}/{target_file}"

    # full_features
    json_contents["full_features"][dataset_key] = {}
    template_full = json_contents["full_features"]["bp"]

    for feature, path in template_full.items():
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
            raise ValueError(f"Cannot infer family for feature '{feature}'")

        json_contents["full_features"][dataset_key][
            feature
        ] = f"${{DATASETS_DIR}}/{family}/{dataset_prefix}_{suffix}/{dataset_key}_{feature}_*.csv"

    # prediction_output_dirs rf/lr
    result_prefix = dataset_prefix.upper()

    for model in ["rf"]:
        json_contents["prediction_output_dirs"][model][dataset_key] = {}

        for feature in template_full.keys():
            json_contents["prediction_output_dirs"][model][dataset_key][
                feature
            ] = f"${{RESULTS_DIR}}/{result_prefix}_predictions_{model}/{feature}"

    existing_features = list(
        json_contents["dataset_analysis"]["descriptor_analysis"].keys()
    )
    identifiers = [
        f"pred_{target_feature}_tr_{train_feature}"
        for train_feature in existing_features
        for target_feature in existing_features
        if train_feature != target_feature
    ]
    _add_dataset_cross_prediction_paths(
        prediction_output_dirs=json_contents["prediction_output_dirs"],
        dataset_key=dataset_key,
        identifiers=identifiers,
    )

    saveJSON(json_contents, json_path)


def addFeatureSetPaths(
    feature_name: str,
    family: str,
    all_filename: str | None = None,
    add_cross_predictions: bool = True,
    json_path: str | Path = FILE_DIR / "paths.json",
    full_cross_embedding_pathing: bool = False,
    limited_cross_embedding_pathing: list[str] = ["rdkit", "mordred"],
):
    json_contents = loadJSON(json_path)
    feature_name = feature_name.lower()
    family = family.lower()

    if family not in {"descriptors", "embeddings", "fingerprints"}:
        raise ValueError("family must be one of: descriptors, embeddings, fingerprints")

    # full_features
    for dataset, block in json_contents["full_features"].items():
        if dataset == "all":
            filename = all_filename or f"all_{feature_name}.csv"
            block[feature_name] = f"${{DATASETS_DIR}}/all/{filename}"
            continue

        prefix = _dataset_feature_prefix(dataset, for_results=False)
        block[feature_name] = (
            f"${{DATASETS_DIR}}/{family}/{prefix}_{family}/{dataset}_{feature_name}_*.csv"
        )

    # prediction_output_dirs rf/lr
    for model in ["rf", "lr"]:
        for dataset, block in json_contents["prediction_output_dirs"][model].items():
            result_prefix = _dataset_feature_prefix(dataset, for_results=True)
            block[feature_name] = (
                f"${{RESULTS_DIR}}/{result_prefix}_predictions_{model}/{feature_name}"
            )

    # dataset_analysis
    json_contents["dataset_analysis"]["descriptor_analysis"][
        feature_name
    ] = f"${{DATASETS_DIR}}/all/descriptor_analysis/{feature_name}.csv"

    # cross predictions
    if add_cross_predictions:
        existing_features = list(
            json_contents["dataset_analysis"]["descriptor_analysis"].keys()
        )

        for other_feature in existing_features:
            if other_feature == feature_name:
                continue

            if not full_cross_embedding_pathing:
                allowed = set(limited_cross_embedding_pathing or [])
                if feature_name not in allowed or other_feature not in allowed:
                    continue

            identifiers = [
                f"pred_{other_feature}_tr_{feature_name}",
                f"pred_{feature_name}_tr_{other_feature}",
            ]

            for dataset in json_contents["full_features"].keys():
                _add_dataset_cross_prediction_paths(
                    prediction_output_dirs=json_contents["prediction_output_dirs"],
                    dataset_key=dataset,
                    identifiers=identifiers,
                )

    saveJSON(json_contents, json_path)


def addRawDataPaths(
    raw_data_paths: list[str],
    set_names: list[str],
    json_name: str = "./test_paths.json",
):
    if len(set_names) != len(raw_data_paths):
        raise ValueError(
            "Make sure that the length of the set " "names and data paths are the same"
        )

    json_path = FILE_DIR / json_name
    with open(str(json_path), "r", encoding="utf-8") as f:
        pj = json.load(f)

    datasets_root = str(DATASETS_DIR)
    for name, rdp in zip(set_names, raw_data_paths):
        raw_path = str(rdp)
        if raw_path.startswith(datasets_root):
            raw_path = raw_path.replace(datasets_root, "${DATASETS_DIR}", 1)
        pj["raw_data"][name] = raw_path

    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(pj, f, indent=2)
