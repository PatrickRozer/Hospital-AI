import pandas as pd
import numpy as np
import os

def load_and_build_los(base_path="C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data"):
    admissions = pd.read_csv(os.path.join(base_path, "ADMISSIONS.csv"))
    patients = pd.read_csv(os.path.join(base_path, "PATIENTS.csv"))

    admissions.columns = admissions.columns.str.strip().str.upper()
    patients.columns = patients.columns.str.strip().str.upper()

    df = admissions.merge(patients, on="SUBJECT_ID", how="left")

    df["ADMITTIME"] = pd.to_datetime(df["ADMITTIME"], errors="coerce")
    df["DISCHTIME"] = pd.to_datetime(df["DISCHTIME"], errors="coerce")
    df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")

    # LOS in days
    df["los_days"] = (df["DISCHTIME"] - df["ADMITTIME"]).dt.total_seconds() / (3600 * 24)
    df = df[df["los_days"] > 0]

       # ✅ Safe age calculation
    df["age"] = df["ADMITTIME"].dt.year - df["DOB"].dt.year
    df["age"] = df["age"] - (
        (df["ADMITTIME"].dt.month < df["DOB"].dt.month) |
        ((df["ADMITTIME"].dt.month == df["DOB"].dt.month) & (df["ADMITTIME"].dt.day < df["DOB"].dt.day))
    ).astype(int)

    df.loc[df["age"] < 0, "age"] = np.nan
    df["age"] = df["age"].clip(0, 120)
    df = df.dropna(subset=["age"])


    # Encode gender
    df["gender"] = df["GENDER"].map({"M": 0, "F": 1})

    features = ["age", "gender", "ETHNICITY", "ADMISSION_TYPE"]
    target = "los_days"

    return df[features], df[target]

