import pandas as pd
import os

def load_and_prepare_association(base_path="C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data"):
    # ---- Load datasets ----
    diagnoses = pd.read_csv(os.path.join(base_path, "DIAGNOSES_ICD.csv"))
    diag_dict = pd.read_csv(os.path.join(base_path, "D_ICD_DIAGNOSES.csv"))
    procedures = pd.read_csv(os.path.join(base_path, "PROCEDURES_ICD.csv"))
    prescriptions = pd.read_csv(os.path.join(base_path, "PRESCRIPTIONS.csv"))

    # 🔑 Normalize column names to uppercase for consistency
    diagnoses.columns = diagnoses.columns.str.strip().str.upper()
    diag_dict.columns = diag_dict.columns.str.strip().str.upper()
    procedures.columns = procedures.columns.str.strip().str.upper()
    prescriptions.columns = prescriptions.columns.str.strip().str.upper()

    # ---- Map ICD9 codes to human-readable text ----
    if "ICD9_CODE" in diagnoses.columns and "ICD9_CODE" in diag_dict.columns:
        diagnoses = diagnoses.merge(diag_dict, on="ICD9_CODE", how="left")

    # ---- Build transactions (per admission) ----
    diag_tx = (
        diagnoses.groupby("HADM_ID")["LONG_TITLE"]
        .apply(list)
        .reset_index()
    )
    proc_tx = (
        procedures.groupby("HADM_ID")["ICD9_CODE"]
        .apply(list)
        .reset_index()
    )
    drug_tx = (
        prescriptions.groupby("HADM_ID")["DRUG"]
        .apply(lambda x: list(set(x)))
        .reset_index()
    )

    # ---- Merge into single transactions ----
    df = diag_tx.merge(proc_tx, on="HADM_ID", how="outer").merge(drug_tx, on="HADM_ID", how="outer")

    # Fill NaNs and convert lists properly
    df = df.fillna("").astype(str)

    # ---- Build final "items" list ----
    df["items"] = df[["LONG_TITLE", "ICD9_CODE", "DRUG"]].apply(
        lambda row: [x for x in row if x not in ["", "nan", "NaT"]], axis=1
    )

    return df[["HADM_ID", "items"]]
