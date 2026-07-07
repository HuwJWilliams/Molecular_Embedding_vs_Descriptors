"""Shared pipeline metadata used across training, prediction, and feature generation."""

from __future__ import annotations

from typing import Any


DEFAULT_TARGET_COLUMNS = {
    "bp": "Boiling_Point",
    "logd": "LogD",
    "pka": "pKa",
    "ld50": "LD50",
    "log_ld50": "LOG_LD50",
    "pic50": "pIC50",
    "pka_paper1_acidic": "pKa",
    "pka_paper1_basic": "pKa",
    "elec_re": "Electron_Reorganisation_Energy",
    "hole_re": "Hole_Reorganisation_Energy",
    "aq_sol":  "Solubility",
    "homo_lumo_gap": "homolumogap",
    "egfr_pic50": "pIC50"
}

SUPPORTED_FEATURE_SETS = (
    "rdkit",
    "mordred",
    "morgan",
    "maccs",
    "chemberta",
    "chembertasey",
    "molformer",
    "molformer-c3-1b",
    "selformer",
    "ft-chemberta",
    "ft-chembertasey",
    "ft-molformer",
    "ft-molformer-c3-1b",
    "ft-selformer"
)

TRANSFORMER_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "chemberta": {
        "tokeniser": "DeepChem/ChemBERTa-100M-MLM",
        "model": "DeepChem/ChemBERTa-100M-MLM",
        "model_label": "ChemBERTa",
        "suffix_label": "chemberta",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "f5c45f44d3061f0346888f5c09db17ec1146d29d",
    },

    "chembertasey": {
        "tokeniser": "seyonec/ChemBERTa-zinc-base-v1",
        "model": "seyonec/ChemBERTa-zinc-base-v1",
        "model_label": "ChemBERTaSey",
        "suffix_label": "chembertasey",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "761d6a18cf99db371e0b43baf3e2d21b3e865a20",
    },

    "molformer": {
        "tokeniser": "ibm-research/MoLFormer-XL-both-10pct",
        "model": "ibm-research/MoLFormer-XL-both-10pct",
        "model_label": "MolFormer",
        "suffix_label": "molformer",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "7b12d946c181a37f6012b9dc3b002275de070314",
        "code_revision": "compat-v4",
    },

    "molformer-c3-1b": {
        "tokeniser": "DeepChem/MoLFormer-c3-1.1B",
        "model": "DeepChem/MoLFormer-c3-1.1B",
        "model_label": "MolFormer-c3-1B",
        "suffix_label": "molformer-c3-1b",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "3e289c74d01665ef1d86069da05656aef1702ba6",
        "code_revision": "compat-v4",
    },

    "selformer": {
        "tokeniser": "HUBioDataLab/SELFormer",
        "model": "HUBioDataLab/SELFormer",
        "model_label": "SELFormer",
        "suffix_label": "selformer",
        "input_kind": "selfies",
        "metadata_col_name": "SELFIES",
        "commit_hash": "177d98b158e999a6cb7fc9743dbfe1e8a17c57e5",
    },
}


def resolve_target_column(task: str, override: str | None = None) -> str:
    """Return the default target column for a task unless overridden."""
    return override or DEFAULT_TARGET_COLUMNS[task]
