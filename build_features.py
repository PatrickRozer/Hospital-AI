import pandas as pd

def build_classification_dataset(patients, admissions, diagnoses, diag_dict, labs, lab_dict):
    # --- Standardize column names ---
    patients.columns = patients.columns.str.lower()
    admissions.columns = admissions.columns.str.lower()
    diagnoses.columns = diagnoses.columns.str.lower()
    diag_dict.columns = diag_dict.columns.str.lower()
    labs.columns = labs.columns.str.lower()
    lab_dict.columns = lab_dict.columns.str.lower()

    # --- Robust renaming in case lowercase didn't apply properly ---
    if "SUBJECT_ID" in patients.columns:
        patients = patients.rename(columns={"SUBJECT_ID": "subject_id"})
    if "SUBJECT_ID" in admissions.columns:
        admissions = admissions.rename(columns={"SUBJECT_ID": "subject_id"})
    if "SUBJECT_ID" in diagnoses.columns:
        diagnoses = diagnoses.rename(columns={"SUBJECT_ID": "subject_id"})
    if "SUBJECT_ID" in labs.columns:
        labs = labs.rename(columns={"SUBJECT_ID": "subject_id"})

    # --- Debugging: show first 10 column names ---
    print("Patients columns:", patients.columns.tolist()[:10])
    print("Admissions columns:", admissions.columns.tolist()[:10])
    print("Diagnoses columns:", diagnoses.columns.tolist()[:10])
    print("Labs columns:", labs.columns.tolist()[:10])

    # --- Merge admissions + patients ---
    df = admissions.merge(patients, on="subject_id", how="left")

    # --- Target variable: diabetes flag (ICD9 250.xx) ---
    diagnoses = diagnoses.merge(diag_dict, on="icd9_code", how="left")
    diagnoses["is_diabetes"] = diagnoses["icd9_code"].astype(str).str.startswith("250").astype(int)
    diabetes_flags = diagnoses.groupby("subject_id")["is_diabetes"].max().reset_index()

    df = df.merge(diabetes_flags, on="subject_id", how="left")
    df["is_diabetes"] = df["is_diabetes"].fillna(0).astype(int)

    # --- Lab features (subset of common labs) ---
    labs = labs.merge(lab_dict, on="itemid", how="left")
    selected_labs = ["GLUCOSE", "CREATININE", "HEMOGLOBIN", "POTASSIUM", "SODIUM"]
    labs = labs[labs["label"].isin(selected_labs)]

    lab_features = (
        labs.groupby(["subject_id", "label"])["valuenum"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    lab_features = lab_features.pivot(index="subject_id", columns="label", values="mean").reset_index()
    lab_features.columns = [str(col).lower() for col in lab_features.columns]

    df = df.merge(lab_features, on="subject_id", how="left")

    # --- Feature / Target split ---
    features = [c for c in df.columns if c not in ["is_diabetes"]]
    target = "is_diabetes"

    return df[features], df[target]
