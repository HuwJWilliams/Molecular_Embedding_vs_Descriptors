from pathlib import Path
import json

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
SCRIPTS_DIR = PROJ_DIR / "scripts"
DATASETS_DIR = PROJ_DIR / "datasets"
RESULTS_DIR = PROJ_DIR / "results"


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


def _family_to_aligned_folder(family: str) -> str:
    family = family.lower()
    mapping = {
        "descriptors": "aligned_desc",
        "embeddings": "aligned_emb",
        "fingerprints": "aligned_fp",
    }
    if family not in mapping:
        raise ValueError(f"Unsupported family: {family}")
    return mapping[family]


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
    json_contents["targets"][dataset_key] = (
        f"${{DATASETS_DIR}}/{dataset_folder_name}/{target_file}"
    )

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

        json_contents["full_features"][dataset_key][feature] = (
            f"${{DATASETS_DIR}}/{family}/{dataset_prefix}_{suffix}/{dataset_key}_{feature}_*.csv"
        )

    # aligned_features
    json_contents["aligned_features"][dataset_key] = {}
    template_aligned = json_contents["aligned_features"]["bp"]

    for feature, path in template_aligned.items():
        if "descriptors" in path:
            family = "descriptors"
        elif "embeddings" in path:
            family = "embeddings"
        elif "fingerprints" in path:
            family = "fingerprints"
        else:
            raise ValueError(f"Cannot infer aligned family for feature '{feature}'")

        suffix = family
        aligned_folder = _family_to_aligned_folder(family)

        json_contents["aligned_features"][dataset_key][feature] = (
            f"${{DATASETS_DIR}}/{family}/{dataset_prefix}_{suffix}/{aligned_folder}/{dataset_key}_{feature}_*.csv"
        )

    # prediction_output_dirs rf/lr
    result_prefix = dataset_prefix.upper()

    for model in ["rf", "lr"]:
        json_contents["prediction_output_dirs"][model][dataset_key] = {}

        for feature in template_full.keys():
            json_contents["prediction_output_dirs"][model][dataset_key][feature] = (
                f"${{RESULTS_DIR}}/{result_prefix}_predictions_{model}/{feature}/{feature}_pred_{dataset_key}"
            )

    saveJSON(json_contents, json_path)


def addFeatureSetPaths(
    feature_name: str,
    family: str,
    all_filename: str | None = None,
    add_cross_predictions: bool = True,
    json_path: str | Path = FILE_DIR / "paths.json",
):
    json_contents = loadJSON(json_path)
    feature_name = feature_name.lower()
    family = family.lower()

    if family not in {"descriptors", "embeddings", "fingerprints"}:
        raise ValueError("family must be one of: descriptors, embeddings, fingerprints")

    aligned_folder = _family_to_aligned_folder(family)

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

    # aligned_features
    for dataset, block in json_contents["aligned_features"].items():
        if dataset == "all":
            filename = all_filename or f"all_{feature_name}.csv"
            block[feature_name] = f"${{DATASETS_DIR}}/all/{aligned_folder}/{filename}"
            continue

        prefix = _dataset_feature_prefix(dataset, for_results=False)
        block[feature_name] = (
            f"${{DATASETS_DIR}}/{family}/{prefix}_{family}/{aligned_folder}/{dataset}_{feature_name}_*.csv"
        )

    # prediction_output_dirs rf/lr
    for model in ["rf", "lr"]:
        for dataset, block in json_contents["prediction_output_dirs"][model].items():
            result_prefix = _dataset_feature_prefix(dataset, for_results=True)
            block[feature_name] = (
                f"${{RESULTS_DIR}}/{result_prefix}_predictions_{model}/{feature_name}/{feature_name}_pred_{dataset}"
            )

    # dataset_analysis
    json_contents["dataset_analysis"]["descriptor_analysis"][feature_name] = (
        f"${{DATASETS_DIR}}/all/descriptor_analysis/{feature_name}.csv"
    )

    # cross predictions
    if add_cross_predictions:
        cross_block = json_contents["prediction_output_dirs"].setdefault(
            "embedding_and_descriptor_cross_predictions", {}
        )

        existing_features = list(json_contents["dataset_analysis"]["descriptor_analysis"].keys())

        for other_feature in existing_features:
            if other_feature == feature_name:
                continue

            cross_block[f"pred_{other_feature}_tr_{feature_name}"] = (
                f"${{RESULTS_DIR}}/embeddings_and_descriptor_predictions/pred_{other_feature}_tr_{feature_name}"
            )
            cross_block[f"pred_{feature_name}_tr_{other_feature}"] = (
                f"${{RESULTS_DIR}}/embeddings_and_descriptor_predictions/pred_{feature_name}_tr_{other_feature}"
            )

    saveJSON(json_contents, json_path)

addFeatureSetPaths(
    feature_name="maccs",
    family="fingerprints",

)