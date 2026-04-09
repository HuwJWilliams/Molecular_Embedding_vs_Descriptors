# 1. Run PCA Analysis (`run_pca.py`)

## Example: Molecule PCA per Feature Set

```bash
python run_pca.py \
  --type mol \
  --feats rdkit chemberta \
  --targs bp logd pka ld50 pic50 \
  --n-mols 10000 \
  --biplot \
  --n-loadings 20
```

## Example: Feature PCA (Descriptors as Points)

```bash
python run_pca.py \
  --type feat \
  --feats rdkit mordred chemberta \
  --n-mols 2000 \
  --biplot \
  --remove-points \
  --heatmap \
  --heatmap-top-n 100
```

### Example: Joined PCA (Combined Feature Columns)

```bash
python run_pca.py \
  --type joined \
  --feats rdkit chemberta \
  --n-mols 5000 \
  --biplot \
  --heatmap
```

## Arguments

- `--type`
  PCA mode:
  `mol` (molecules as rows, one run per feature set),
  `feat` (transposed feature-space PCA),
  `joined` (single molecule PCA with all selected feature columns concatenated).
- `--feats`
  Feature sets to include. Choices come from `SUPPORTED_FEATURE_SETS`.
- `--targs`
  Target/property datasets to include (used in `mol` mode).
- `--filter-mw`
  Two molecular-weight bounds: `MIN_MW MAX_MW`.
- `--n-mols`
  Number of molecules to sample before PCA.
- `--seed`
  Random seed for reproducible sampling.
- `--n_comp`
  Number of PCA components to compute.
- `--scale-all`
  Apply row and column scaling in preprocessing.
- `--scale-r`
  Apply row-only scaling.
- `--scale-c`
  Apply column-only scaling.
- `--center-all`
  Apply row and column centering.
- `--center-r`
  Apply row-only centering.
- `--center-c`
  Apply column-only centering.
- `--biplot`
  Save PCA biplot output.
- `--pc-x`
  Principal component index for biplot x-axis.
- `--pc-y`
  Principal component index for biplot y-axis.
- `--remove-points`
  Hide scatter points in biplot (arrows only).
- `--n-loadings`
  Number of loading arrows to draw on biplot.
- `--not-pca`
  Skip the full PCA grid plot (`plotPCA`) and run only optional biplot/heatmap outputs.
- `--heatmap`
  Save loadings heatmap from selected PCs.
- `--heatmap-top-n`
  Number of top features to show in loadings heatmap.
- `--save-path`
  Directory to save outputs.
- `--save-name`
  Output filename prefix.

### Notes

- `mol` mode: one plot per feature set in `--feats`, with `Source` labels from `--targs`.
- `feat` mode: data are transposed before PCA; this is descriptor-space analysis.
- `joined` mode: selected feature sets are joined column-wise on common molecule IDs.
- If using preprocessing flags (`--scale-*`, `--center-*`), avoid over-normalising unintentionally.

---

# 2. Run SHAP Analysis (`run_shap.py`)

## Example: Beeswarm + Dependence for One Predicted Descriptor

```bash
python run_shap.py \
  --pred-set rdkit \
  --train-set maccs \
  --pred-feat MaxAbsEStateIndex \
  --results-dir lipinski_cross_feature_predictions \
  --plot-shap \
  --plot-dep \
  --max-bg 200 \
  --max-exp 500 \
  --n-display 20
```

## Example: Group SHAP (RDKit/Mordred only)

```bash
python run_shap.py \
  --pred-set rdkit \
  --train-set maccs \
  --pred-feat MaxAbsEStateIndex \
  --results-dir lipinski_cross_feature_predictions \
  --group-shap \
  --top-n 12 \
  --max-bg 100 \
  --max-exp 300
```

## Arguments

- `--pred-set`
  Predicted feature family. Choices are from `SUPPORTED_FEATURE_SETS`.
- `--train-set`
  Training feature family. Choices are from `SUPPORTED_FEATURE_SETS`.
- `--pred-feat`
  Descriptor/target name to select the model file (script appends `_{pred_set}` internally).
- `--results-dir`
  Key under `paths["prediction_output_dirs"]` (for example `cross_feature_predictions` or `lipinski_cross_feature_predictions`).
- `--max-bg`
  Maximum rows used for SHAP background baseline sampling.
- `--max-exp`
  Maximum rows used for SHAP explanation sampling.
- `--n-display`
  Number of features to show in SHAP beeswarm.
- `--plot-shap`
  Save SHAP beeswarm.
- `--plot-dep`
  Save SHAP dependence plots for top-ranked features.
- `--group-shap`
  Run grouped SHAP analysis (supported for `rdkit` and `mordred` predicted sets).
- `--top-n`
  Number of top features to include in grouped SHAP violin plots.

## Outputs

- Saved under `<prediction_run_dir>/shap/`
- Typical files:
  - `shap_beeswarm_top*.png`
  - `shap_dependence_*.png`
  - `*_shap_violin_signed_top*.png`
  - `*_shap_violin_abs_top*.png`
  - `descriptor_shap_summary.csv`
  - `group_shap_summary.csv`
