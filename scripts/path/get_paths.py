from pathlib import Path

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[2]
# print(f"Project Reference Dir for file {FILE_DIR.name}:\n{PROJ_DIR}")
SCRIPTS_DIR = PROJ_DIR / "scripts"
DATASETS_DIR = PROJ_DIR / "datasets"
RESULTS_DIR = PROJ_DIR / 'results'


def getPaths():
    paths = {
        # Important Directories
        "imp_dirs": {
            "proj_dir":     PROJ_DIR,
            "scripts_dir":  SCRIPTS_DIR,
            "datasets_dir": DATASETS_DIR,
            "results_dir":  RESULTS_DIR
        },


        # Paths to input target values
        "targets": {
            "bp":       DATASETS_DIR / "boiling_point" / "cleaned_boiling_point.csv",
            "logd":     DATASETS_DIR / "logD"          / "cleaned_logd.csv",
            "pka":      DATASETS_DIR / "pka"           / "cleaned_pka.csv",
            "ld50":     DATASETS_DIR / "LD50"          / "cleaned_ld50.csv",
            "pic50":    DATASETS_DIR / "pic50"         / "cleaned_pic50.csv",
            "all":      DATASETS_DIR /  "all"          / "cleaned_all.csv",
        },

        # Paths to input features
        "full_features": {
            "bp": {
                "rdkit":     DATASETS_DIR / "descriptors" / "BP_descriptors"  / "bp_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "BP_descriptors"  / "bp_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "BP_embeddings"   / "bp_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "BP_embeddings"   / "bp_molformer_*.csv",
            },
            "logd": {
                "rdkit":     DATASETS_DIR / "descriptors" / "LOGD_descriptors" / "logd_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "LOGD_descriptors" / "logd_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "LOGD_embeddings"  / "logd_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "LOGD_embeddings"  / "logd_molformer_*.csv",
            },
            "pka": {
                "rdkit":     DATASETS_DIR / "descriptors" / "PKA_descriptors"  / "pka_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "PKA_descriptors"  / "pka_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "PKA_embeddings"   / "pka_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "PKA_embeddings"   / "pka_molformer_*.csv",
            },
            "ld50": {
                "rdkit":     DATASETS_DIR / "descriptors" / "LD50_descriptors" / "ld50_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "LD50_descriptors" / "ld50_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "LD50_embeddings"  / "ld50_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "LD50_embeddings"  / "ld50_molformer_*.csv",
            },
            "pic50": {
                "rdkit":     DATASETS_DIR / "descriptors" / "pIC50_descriptors" / "pic50_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "pIC50_descriptors" / "pic50_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings" / "pIC50_embeddings" / "pic50_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings" / "pIC50_embeddings" / "pic50_molformer_*.csv",
            },
            "all": {
                "rdkit":     DATASETS_DIR / "all" / "all_rdkit.csv",
                "mordred":   DATASETS_DIR / "all" / "all_mordred.csv",
                "chemberta": DATASETS_DIR / "all" / "all_chemberta.csv",
                "molformer": DATASETS_DIR / "all" / "all_molformer.csv",
            },
        },

        # This is for when you have multiple batch files, you cal align the features across all batches to be the same
        # For this to work properly you need to have drop_cols set to False when generating features
        "aligned_features": {
            "bp": {
                "rdkit":     DATASETS_DIR / "descriptors" / "BP_descriptors"  / "aligned_desc" / "bp_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "BP_descriptors"  / "aligned_desc" / "bp_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "BP_embeddings"   / "aligned_emb"  / "bp_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "BP_embeddings"   / "aligned_emb"  / "bp_molformer_*.csv",
            },
            "logd": {
                "rdkit":     DATASETS_DIR / "descriptors" / "LOGD_descriptors" / "aligned_desc" / "logd_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "LOGD_descriptors" / "aligned_desc" / "logd_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "LOGD_embeddings"  / "aligned_emb"  / "logd_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "LOGD_embeddings"  / "aligned_emb"  / "logd_molformer_*.csv",
            },
            "pka": {
                "rdkit":     DATASETS_DIR / "descriptors" / "PKA_descriptors"  / "aligned_desc" / "pka_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "PKA_descriptors"  / "aligned_desc" / "pka_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "PKA_embeddings"   / "aligned_emb"  / "pka_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "PKA_embeddings"   / "aligned_emb"  / "pka_molformer_*.csv",
            },
            "ld50": {
                "rdkit":     DATASETS_DIR / "descriptors" / "LD50_descriptors" / "aligned_desc" / "ld50_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "LD50_descriptors" / "aligned_desc" / "ld50_mordred_*.csv",
                "chemberta": DATASETS_DIR / "embeddings"  / "LD50_embeddings"  / "aligned_emb"  / "ld50_chemberta_*.csv",
                "molformer": DATASETS_DIR / "embeddings"  / "LD50_embeddings"  / "aligned_emb"  / "ld50_molformer_*.csv",
            },
            "pic50": {
                "rdkit":     DATASETS_DIR / "descriptors" / "pIC50_descriptors" / "aligned_desc" / "pic50_rdkit_*.csv",
                "mordred":   DATASETS_DIR / "descriptors" / "pIC50_descriptors" / "aligned_desc" / "pic50_mordred_*.csv",
                "chemberta": DATASETS_DIR / "descriptors" / "pIC50_embeddings" / "aligned_emb" / "pic50_chemberta_*.csv",
                "molformer": DATASETS_DIR / "descriptors" / "pIC50_embeddings" / "aligned_emb" / "pic50_molformer_*.csv",
            },
            "all": {
                "rdkit":     DATASETS_DIR / "all" / "aligned_feats" / "all_rdkit.csv",
                "mordred":   DATASETS_DIR / "all" / "aligned_feats" / "all_mordred.csv",
                "chemberta": DATASETS_DIR / "all" / "aligned_feats" / "all_chemberta.csv",
                "molformer": DATASETS_DIR / "all" / "aligned_feats" / "all_molformer.csv",
            },
        },


        
        # Prediction Results
        "prediction_output_dirs": {
            # Random Forest Directories
            "rf": {
                "bp":   {
                    "rdkit":        RESULTS_DIR / "BP_predictions_rf"   / "rdkit"       / "rdkit_pred_bp",
                    "mordred":      RESULTS_DIR / "BP_predictions_rf"   / "mordred"     / "mordred_pred_bp",
                    "chemberta":    RESULTS_DIR / "BP_predictions_rf"   / "chemberta"   / "chemberta_pred_bp",
                    "molformer":    RESULTS_DIR / "BP_predictions_rf"   / "molformer"   / "molformer_pred_bp"    
                    },
                "logd": {
                    "rdkit":        RESULTS_DIR / "LOGD_predictions_rf" / "rdkit"       / "rdkit_pred_logd",
                    "mordred":      RESULTS_DIR / "LOGD_predictions_rf" / "mordred"     / "mordred_pred_logd",
                    "chemberta":    RESULTS_DIR / "LOGD_predictions_rf" / "chemberta"   / "chemberta_pred_logd",
                    "molformer":    RESULTS_DIR / "LOGD_predictions_rf" / "molformer"   / "molformer_pred_logd"
                    },
                "pka":  {
                    "rdkit":        RESULTS_DIR / "PKA_predictions_rf"  / "rdkit"       / "rdkit_pred_pka",
                    "mordred":      RESULTS_DIR / "PKA_predictions_rf"  / "mordred"     / "mordred_pred_pka",
                    "chemberta":    RESULTS_DIR / "PKA_predictions_rf"  / "chemberta"   / "chemberta_pred_pka",
                    "molformer":    RESULTS_DIR / "PKA_predictions_rf"  / "molformer"   / "molformer_pred_pka"
                    },
                "ld50": {
                    "rdkit":        RESULTS_DIR / "LD50_predictions_rf" / "rdkit"       / "rdkit_pred_ld50",
                    "mordred":      RESULTS_DIR / "LD50_predictions_rf" / "mordred"     / "mordred_pred_ld50",
                    "chemberta":    RESULTS_DIR / "LD50_predictions_rf" / "chemberta"   / "chemberta_pred_ld50",
                    "molformer":    RESULTS_DIR / "LD50_predictions_rf" / "molformer"   / "molformer_pred_ld50"
                         },
                "pic50": {
                    "rdkit":        RESULTS_DIR / "PIC50_predictions_rf" / "rdkit"      / "rdkit_pred_pic50",
                    "mordred":      RESULTS_DIR / "PIC50_predictions_rf" / "mordred"    / "mordred_pred_pic50",
                    "chemberta":    RESULTS_DIR / "PIC50_predictions_rf" / "chemberta"  / "chemberta_pred_pic50",
                    "molformer":    RESULTS_DIR / "PIC50_predictions_rf" / "molformer"  / "molformer_pred_pic50"

                }
            },

            # Linear Regression Directories
            "lr": {
                "bp":   {
                    "rdkit":     RESULTS_DIR / "BP_predictions_lr"   / "rdkit",
                    "mordred":   RESULTS_DIR / "BP_predictions_lr"   / "mordred",
                    "chemberta": RESULTS_DIR / "BP_predictions_lr"   / "chemberta",
                    "molformer": RESULTS_DIR / "BP_predictions_lr"   / "molformer"
                    },
                "logd": {
                    "rdkit":     RESULTS_DIR / "LOGD_predictions_lr" / "rdkit",
                    "mordred":   RESULTS_DIR / "LOGD_predictions_lr" / "mordred",
                    "chemberta": RESULTS_DIR / "LOGD_predictions_lr" / "chemberta",
                    "molformer": RESULTS_DIR / "LOGD_predictions_lr" / "molformer"
                    },
                "pka":  {
                    "rdkit":     RESULTS_DIR / "PKA_predictions_lr"  / "rdkit",
                    "mordred":   RESULTS_DIR / "PKA_predictions_lr"  / "mordred",
                    "chemberta": RESULTS_DIR / "PKA_predictions_lr"  / "chemberta",
                    "molformer": RESULTS_DIR / "PKA_predictions_lr"  / "molformer"
                    },
                "ld50": {
                    "rdkit":     RESULTS_DIR / "LD50_predictions_lr" / "rdkit",
                    "mordred":   RESULTS_DIR / "LD50_predictions_lr" / "mordred",
                    "chemberta": RESULTS_DIR / "LD50_predictions_lr" / "chemberta",
                    "molformer": RESULTS_DIR / "LD50_predictions_lr" / "molformer"
                    },
            },

            "embedding_and_descriptor_cross_predictions": {
                "pred_rdkit_tr_chemberta":    RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_rdkit_tr_chemberta",
                "pred_rdkit_tr_molformer":    RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_rdkit_tr_molformer",
                "pred_mordred_tr_chemberta":  RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_mordred_tr_chemberta",
                "pred_mordred_tr_molformer":  RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_mordred_tr_molformer",
                "pred_chemberta_tr_rdkit":    RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_chemberta_tr_rdkit",
                "pred_molformer_tr_mordred":  RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_molformer_tr_mordred",
                "pred_molformer_tr_rdkit":    RESULTS_DIR / "embeddings_and_descriptor_predictions" / "pred_molformer_tr_rdkit",
            },
        },

        # Dataset Analysis
        "dataset_analysis": {
            "descriptor_analysis": {
                "rdkit": DATASETS_DIR / "all" / "descriptor_analysis" / "rdkit.csv",
                "mordred": DATASETS_DIR / "all" / "descriptor_analysis" / "mordred.csv",
                "chemberta": DATASETS_DIR / "all" / "descriptor_analysis" / "chemberta.csv",
                "molformer": DATASETS_DIR / "all" / "descriptor_analysis" / "molformer.csv",

            }
        },

        # Config
        "config": {
            "logs": SCRIPTS_DIR / "models" / "logs"
        },
    }

    return paths
