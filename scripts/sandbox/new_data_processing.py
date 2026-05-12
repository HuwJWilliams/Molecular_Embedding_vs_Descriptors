import pandas as pd
from pathlib import Path

path_to_new_data = Path("~/Downloads")
save_path = Path("~/TL_project/datasets/pka")

filenames = ["Opt1_basic_tr.csv", "Opt2_basic_tr.csv", "Opt3_basic_tr.csv", 
             "Opt2_basic_tst.csv", "Opt2_basic_tst.csv", "Opt2_basic_tst.csv"]
keep_cols = ["pKa", "Canonical_QSARr"]


full_set = pd.DataFrame()
tst_set = pd.DataFrame()
tr_set = pd.DataFrame()

for fn in filenames:
    tmp_df = pd.read_csv(path_to_new_data / fn, usecols=keep_cols)

    tmp_df["ID"] = [
        f"pka_paper1_basic_{n}"
        for n in range(len(full_set), len(full_set) + len(tmp_df))
    ]

    full_set = pd.concat([full_set, tmp_df], ignore_index=True)

    if "tr.csv" in fn:
        tr_set = pd.concat([tr_set, tmp_df], ignore_index=True)
    elif "tst.csv" in fn:
        tst_set = pd.concat([tst_set, tmp_df], ignore_index=True)

full_set = full_set.set_index("ID").rename(columns={"Canonical_QSARr": "SMILES"})
tr_set = tr_set.set_index("ID").rename(columns={"Canonical_QSARr": "SMILES"})
tst_set = tst_set.set_index("ID").rename(columns={"Canonical_QSARr": "SMILES"})



full_set.to_csv(save_path / "cleaned_pka_paper1_basic.csv")
tst_set.index.to_frame(index=False, name="ID").to_csv(
    save_path.parent / "training_data" / "pka_paper1_basic_model_validation.csv",
    index=False,
)

tr_set.index.to_frame(index=False, name="ID").to_csv(
    save_path.parent / "training_data" / "pka_paper1_basic_model_training.csv",
    index=False,
)