import os
import pandas as pd

def load_and_build_clustering(base_path="C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data"):
    admissions = pd.read_csv(os.path.join(base_path, "ADMISSIONS.csv"))
    patients = pd.read_csv(os.path.join(base_path, "PATIENTS.csv"))
    diagnoses = pd.read_csv(os.path.join(base_path, "DIAGNOSES_ICD.csv"))
    labs = pd.read_csv(os.path.join(base_path, "LABEVENTS.csv"))

    # 🔑 Normalize column names to uppercase
    admissions.columns = admissions.columns.str.strip().str.upper()
    patients.columns = patients.columns.str.strip().str.upper()
    diagnoses.columns = diagnoses.columns.str.strip().str.upper()
    labs.columns = labs.columns.str.strip().str.upper()

    print("Admissions cols:", admissions.columns.tolist())  # 👈 debug
    print("Patients cols:", patients.columns.tolist())      # 👈 debug

    # ✅ Always merge on uppercase SUBJECT_ID
    df = admissions.merge(patients, on="SUBJECT_ID", how="left")

    # Age
    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
    df["age"] = df["ADMITTIME"].dt.year - df["DOB"].dt.year
    df["age"] = df["age"].clip(0, 120)

    # Gender
    df["gender"] = df["GENDER"].map({"M": 0, "F": 1})

    features = ["age", "gender"]
    X = df[features].fillna(0)

    return X, df
