# %%
import sys
from misc_fns import loadData
from pathlib import Path

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/datasets")
# from feature_generator import FeatureGenerator
from standardise_dataset import getMoleculeIntersection

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/path")
from get_paths import getPaths

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/misc")
from misc_fns import loadData

paths = getPaths()

# %%

smi_ls = ["OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(C(F)(F)F)C(F)(F)F", "OC(=O)C(F)(F)C(F)(F)C(F)(F)F", "OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F"]
id_ls = ["id1", "id2", "id3"]

fg = FeatureGenerator("mordred")
df = fg.calcBatchFeatures(smi_ls, id_ls, drop_cols=True)
print(df)


# %%
df = fg.calcBatchFeatures(smi_ls, id_ls, drop_cols=False)
print(df)

# fg = FeatureGenerator("mordred")
# df = fg.calcMordred(smi_ls, id_ls)
# print(df)
# # %%d

# fg = FeatureGenerator("chemberta")
# df = fg.calcChemBERTa(smi_ls, id_ls)
# print(df)

# # %%
# fg = FeatureGenerator("molformer")
# df = fg.calcMolFormer(smi_ls, id_ls)
# print(df)

# %%


bp_desc_paths = Path("/users/yhb18174/TL_project/datasets/descriptors/BP_descriptors")
bp_emb_paths = Path("/users/yhb18174/TL_project/datasets/embeddings/BP_embeddings")

ls = getMoleculeIntersection(df_ls=[
    bp_desc_paths / "bp_mordred_1.csv",
    bp_desc_paths / "bp_rdkit_1.csv",
    bp_emb_paths / "bp_chemberta_1.csv",
    bp_emb_paths / "bp_molformer_1.csv",
    ],
    index_col='ID',
    on_smiles=True
    )

# %%
from standardise_dataset import filterMolWt

df = paths["aligned_features"]["logd"]["rdkit"]
loaded_df = loadData(df, index_col='ID', wildcard='*')
print(len(loaded_df))

filtered_df = filterMolWt(loaded_df, max_threshold=600)

print(filtered_df['MolWt'].max())
print(filtered_df['MolWt'].min())
print(len(filtered_df))

# %%

performance1 = [89, 89, 88, 78, 79]
performance2 = [93, 92, 94, 89, 88]
performance3 = [89, 88, 89, 93, 90]
performance4 = [81, 78, 81, 92, 82]

from scipy.stats import f_oneway

f_statistic, p_value = f_oneway(performance1, performance2, performance3, performance4)

print(f"F-statistic: {f_statistic}")
print(f"P-value: {p_value}")
# %%

sys.path.insert(0, "/users/yhb18174/TL_project/scripts/models")
from transfer_model import TL

paths=getPaths()

data = loadData(df=paths["aligned_features"]["bp"]["rdkit"],    
                index_col='ID',
    wildcard="*")


targets = loadData(df=paths['targets']["bp"])

data_cp = data.copy()

data_cp['Boiling Point'] = targets['Boiling Point']

data_cp

# %%
model = TL()
result = model.trainSingleLRModel(
    data = data_cp,
    target_column='Boiling Point',
    scale_data=True
)
# %%
result
# %%
from sklearn.feature_selection import f_regression

# Extract input (X) and target (y)
X = data_cp.drop(columns=['Boiling Point', "SMILES"])
y = data_cp['Boiling Point']

# Run F-test for regression
F, p = f_regression(X, y)

# %%
# Combine with feature names
import pandas as pd
f_test = pd.DataFrame({
    'Feature': X.columns,
    'F-Statistic': F,
    'p-Value': p
}).sort_values(by='F-Statistic', ascending=False)

top_features = f_test

top_features.reset_index(drop=True, inplace=True)
top_features.set_index("Feature")

# %%
anova_df = pd.DataFrame({
    'Feature': X.columns,
    'F-Statistic': F,
    'p-Value': p
})
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['F-Statistic'], color='skyblue')
plt.xlabel('F-Statistic')
plt.ylabel('Feature')
plt.title('Top 10 Features by ANOVA F-Statistic')
plt.gca().invert_yaxis()  # so the highest F is at the top
plt.tight_layout()
plt.show()


# %%
