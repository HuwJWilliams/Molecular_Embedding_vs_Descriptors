import argparse
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "datasets"))
from join_desc_df import makeUniqueSMILES, combineAllFeats

sys.path.insert(0, str(SCRIPTS_DIR / "config"))
from pipeline_config import DEFAULT_TARGET_COLUMNS, SUPPORTED_FEATURE_SETS

paths = getPaths()


def main():
    parser = argparse.ArgumentParser(description="Combine multiple datasets together")

    parser.add_argument(
        "--target-properties",
        nargs="+",
        required=True,
        choices=DEFAULT_TARGET_COLUMNS.keys(),
        help="Dataset keys from paths.json, e.g. bp logd pka",
    )
    parser.add_argument(
        "--override-existing",
        action="store_true",
        help="Override existing combined target file",
    )
    parser.add_argument(
        "--feature-list",
        nargs="+",
        required=True,
        choices=SUPPORTED_FEATURE_SETS,
        help="Feature names, e.g. rdkit mordred chemberta",
    )
    parser.add_argument(
        "--align-common-ids",
        action="store_true",
        help="Intersect IDs across feature tables before saving",
    )

    parser.add_argument(
        "--max-mw",
        type=float,
        default=None,
        help="Molecular weight threshold",
    )

    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Maximum allowed atom count",
    )

    parser.add_argument(
        "--save-path",
        default="all",
        choices=paths["full_features"].keys(),
        help="Key within the full_features portion of paths.json",
    )

    parser.add_argument(
        "--fit-lipinski",
        action="store_true",
        help="Flag checking if the user wants the data to fit Lipinski Rules"
    )

    parser.add_argument(
        "--lipinski-mw-limit",
        type=float,
        default=600.0
    )

    parser.add_argument(
        "--logp-limit",
        type=float,
        default=6.0
    )

    parser.add_argument(
        "--hbd-limit",
        type=int,
        default=6
    )

    parser.add_argument(
        "--hba-limit",
        type=int,
        default=11
    )

    args = parser.parse_args()

    target_columns = [DEFAULT_TARGET_COLUMNS[key] for key in args.target_properties]

    makeUniqueSMILES(
        properties=target_columns,
        override_full_csv=args.override_existing,
    )

    combineAllFeats(
        feat_set_ls=args.feature_list,
        properties=args.target_properties,
        align_common_ids=args.align_common_ids,
        max_mw=args.max_mw,
        max_atoms=args.max_atoms,
        feat_path=args.save_path,
        lipinski_criteria=args.fit_lipinski,
        lipinski_mw=args.lipinski_mw_limit,
        lipinski_logp=args.logp_limit,
        lipinski_n_hbd=args.hbd_limit,
        lipinski_n_hba=args.hba_limit
    )


if __name__ == "__main__":
    main()
