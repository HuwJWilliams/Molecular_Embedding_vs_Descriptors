
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load descriptor performance data
data = pd.read_csv("./rdkit_total_performance.csv", index_col="Descriptor")
data = data.sort_values(by="Pearson_r", ascending=True)

# Select top and bottom performing descriptors
best_predicted = data.head(10)
worst_predicted = data.tail(10)

# Specific subsets (logP-related)
logp = data[data.index.astype(str).str.contains("logp", case=False, na=False)]

# ===== Histogram of Pearson R distribution ===== #
col = "Pearson_r"
s = pd.to_numeric(data[col], errors="coerce").dropna().clip(0, 1)

bins = np.linspace(0, 1, 11)
labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
cats = pd.cut(s, bins=bins, include_lowest=True, right=True, labels=labels)

counts = cats.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(counts.index.astype(str), counts.values)
ax.bar_label(bars, labels=[str(v) for v in counts.values], padding=2, fontsize=9)
ax.set_ylim(0, max(counts.values)*1.10)
ax.set_xlabel(col)
ax.set_ylabel("Count")
plt.title("RF Performance on RDKit Descriptors (Pearson r)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# ===== Horizontal bar chart of worst predicted ===== #
plt.figure(figsize=(8, 6))
plt.barh(worst_predicted.index, worst_predicted["Pearson_r"].values)
plt.title("Worst Predicted RDKit Descriptors (Pearson r)")
plt.yticks(rotation=15, fontsize=8)
plt.xlabel("Pearson r")
plt.tight_layout()
plt.show()


# ===== Horizontal bar chart of best predicted ===== #
plt.figure(figsize=(8, 6))
plt.barh(best_predicted.index, best_predicted["Pearson_r"].values)
plt.title("Best Predicted RDKit Descriptors (Pearson r)")
plt.yticks(rotation=15, fontsize=8)
plt.xlabel("Pearson r")
plt.tight_layout()
plt.show()

# %%
