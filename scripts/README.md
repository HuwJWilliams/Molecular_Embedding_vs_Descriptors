# Project Setup

## 1. Create and activate the environment

```bash
mamba env create -f environment.yml
mamba activate tlp_py312
```

## 2. Generate pathing config

### Example

```bash
python setup.py --create-pathing --json-name paths.json --set-names bp logd pka
```

### Arguments

- `--create-pathing`
  Create the pathing JSON. Skip this if it already exists.
- `--json-name`
  Output pathing filename (default: `paths.json`).
- `--set-names`
  Dataset names to register before preprocessing.

## 3. Standardise datasets for pipeline use

### Example

```bash
python standardise_datasets.py \
  --dataset-name bp \
  --target-col "Boiling Point {measured, converted}" \
  --rename-cols '{"Boiling Point {measured, converted}": "Boiling_Point"}' \
  --smiles-col SMILES \
  --id-prefix bp \
  --shuffle-data \
  --plot-distribution
```

### Arguments

- `--dataset-name`
  Raw dataset name (as registered in `setup.py`).
- `--target-col`
  Target column in the raw dataset.
- `--rename-cols`
  Optional column rename mapping (JSON string).
- `--smiles-col`
  SMILES column name.
- `--id-col`
  Existing ID column name (optional).
- `--id-prefix`
  Prefix for generated IDs when no ID column is provided.
- `--shuffle-data`
  Shuffle dataset rows.
- `--plot-distribution`
  Plot descriptor distributions for:
  `MolWt`, `MolLogP`, `NumAromaticRings`, `NumRotatableBonds`, `NumHDonors`, `NumHAcceptors`.

### Notes

- Do not pass `--id-col None`; omit `--id-col` entirely if you do not have one.
- `--rename-cols` should be valid JSON (double quotes).

## 4. Generating Features for Datasets

### Example

```bash
python generate_features.py --task bp --feature-set rdkit --batch-size 100000
```

### Arguments

- `--task`
  Target dataset to generate features for
- `--feature-set`
  Feature set to generate
- `--batch-size`
  Batch size for processing molecules (early development)


## 5. Generating Cross-Feature Predictions

### Example

```bash
python cross_feature_predictions.py \
  --train chemberta \
  --test rdkit \
  --save-dir cross_feature_predictions \
  --n-estimators 200 \
  --max-features sqrt \
  --max-depth 50 \
  --min-samples-split 2 \
  --min-samples-leaf 2 \
  --n-resamples 1 \
  --test-size 0.3 \
  --skip-existing \
  --save-models
```

### Arguments

- `--train`
  Feature set to train the random forests on
- `--test`
  Feature set to predict with trained models
- `--save-dir`
  Directory name to save results under (name = key in paths.json)
- `--n-estimators`
  Number of trees in random forest
- `--max-features`
  Maximum number of features each spliut is allowed to consider
- `--max-depth`
  Maximum depth of each decision tree.
- `--min-samples-split`
  Minimum number of samples required at a node before it can be split.
- `--min-samples-leaf`
  Minimum number of samples required in each leaf node.
- `--n-resamples`
  Number of resamples in the outer loop for hyperparameter optimisation
- `--test-size`
  Fraction of the training set (0.3 = 30 %)
- `--skip-existing`
  Flag to skip already processed features
- `--save-models`
  Flag to save trained models

### Notes

- For more information about the Random Forest settings look at the SciKit-Learn documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

## 6. Run Cross-Prediction Analysis

## Example

```bash
python run_cross_feature_analysis.py \
  --results-dir \
  --run-all \
  --exclude-low-var \
  --show-var \
  --var-threshold 0.8
```

### Arguments
- `--results-dir`
  Name of the directory key in the pathing json under "prediction_output_dirs"
- `--run-all`
  Flag to run all available cross-feature analysis
- `--run-experiments`
  Option to run cross-feature analysis on a select number of experiments.
  Here you reference an experiment as its identifier (e.g., pred_rdkit_tr_chemberta)
- `--exclude-low-var`
  Flag to exclude features with particularly low variance which were missed in the initial
  filtering process. Variance defined as fraction of molecules which have the same common value.
-  `--show-var`
  Show the variance of the feature columns
-  `--var-threshold`
  Threshold for the fraction of molecules having the same common value
  (i.e., 0.7 means if more than 70% of the data in a column has the same value, remove it)