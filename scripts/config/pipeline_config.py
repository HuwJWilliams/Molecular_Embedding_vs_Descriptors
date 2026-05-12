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

}

SUPPORTED_FEATURE_SETS = (
    "rdkit",
    "mordred",
    "chemberta",
    "chembertasey",
    "molformer",
    "molformer-c3-1b",
    "selformer",
    "morgan",
    "maccs",
)

TRANSFORMER_FEATURE_SPECS: dict[str, dict[str, Any]] = {
    "chemberta": {
        "tokeniser": "DeepChem/ChemBERTa-100M-MLM",
        "model": "DeepChem/ChemBERTa-100M-MLM",
        "model_label": "ChemBERTa",
        "suffix_label": "chemberta",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
    },
    "chembertasey": {
        "tokeniser": "seyonec/ChemBERTa-zinc-base-v1",
        "model": "seyonec/ChemBERTa-zinc-base-v1",
        "model_label": "ChemBERTaSey",
        "suffix_label": "chembertasey",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
    },
    "molformer": {
        "tokeniser": "ibm/MoLFormer-XL-both-10pct",
        "model": "ibm/MoLFormer-XL-both-10pct",
        "model_label": "MolFormer",
        "suffix_label": "molformer",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
    },
    "molformer-c3-1b": {
        "tokeniser": "DeepChem/MoLFormer-c3-1.1B",
        "model": "DeepChem/MoLFormer-c3-1.1B",
        "model_label": "MolFormer-c3-1B",
        "suffix_label": "molformer-c3-1b",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
    },
    "selformer": {
        "tokeniser": "HUBioDataLab/SELFormer",
        "model": "HUBioDataLab/SELFormer",
        "model_label": "SELFormer",
        "suffix_label": "selformer",
        "input_kind": "selfies",
        "metadata_col_name": "SELFIES",
    },
}


def resolve_target_column(task: str, override: str | None = None) -> str:
    """Return the default target column for a task unless overridden."""
    return override or DEFAULT_TARGET_COLUMNS[task]
