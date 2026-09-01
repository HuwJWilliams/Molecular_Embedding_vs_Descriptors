import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(
    "/users/yhb18174/TL_project/datasets/embeddings/BP_embeddings/"
    "bp_ft-random-molformer-c3-1b_fine_tuned_model/fine_tuning_log_history.csv"
)

train_loss = data.dropna(subset=["loss"]).copy()

x_col = "epoch" if "epoch" in train_loss.columns else "step"

plt.figure(figsize=(7, 4))
plt.plot(train_loss[x_col], train_loss["loss"], marker="o")
plt.xlabel(x_col.capitalize())
plt.ylabel("Training loss")
plt.title("Fine-tuning Training Loss")
plt.tight_layout()
plt.show()
