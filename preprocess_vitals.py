"""
Load timeseries (or raw) CSV and produce scaled sequences for LSTM.
Exports:
 - load_timeseries() -> DataFrame (timeseries)
 - make_sequences(df, seq_length) -> (X, y, scaler)
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TS_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/src/timeseries", "human_vital_signs_timeseries_1.csv")
RAW_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/human_vital_signs_dataset_2024.csv", "human_vital_signs_dataset_2024.csv")

def _rename_columns(df):
    rename_map = {
        "Patient ID": "patient_id",
        "PatientID": "patient_id",
        "Timestamp": "time",
        "Time": "time",
        "Heart Rate": "heart_rate",
        "HeartRate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "RespiratoryRate": "respiratory_rate",
        "Body Temperature": "temperature",
        "Temperature": "temperature",
        "Oxygen Saturation": "spo2",
        "Systolic Blood Pressure": "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp"
    }
    available = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=available)

def load_timeseries():
    """
    Prefer an existing timeseries file. If not present, try to read raw file and:
    - if raw appears already time-series (multiple rows per patient), use it
    - otherwise raise error (user should run generator)
    """
    if os.path.exists(TS_FILE):
        df = pd.read_csv(TS_FILE, low_memory=False)
        df = _rename_columns(df)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        return df

    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Neither timeseries ({TS_FILE}) nor raw ({RAW_FILE}) found. Place your CSV in data/vitals/")

    raw = pd.read_csv(RAW_FILE, low_memory=False)
    raw = _rename_columns(raw)
    if "time" in raw.columns:
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    # check readings per patient
    if "patient_id" in raw.columns:
        counts = raw.groupby("patient_id").size()
        if counts.min() > 1:
            # treat raw as time-series and save a copy
            os.makedirs(os.path.dirname(TS_FILE), exist_ok=True)
            raw.to_csv(TS_FILE, index=False)
            print(f"Saved raw CSV as timeseries (detected multiple rows per patient): {TS_FILE}")
            return raw
    raise ValueError("No timeseries found and raw CSV does not appear to have multiple time-rows per patient. Run the generator script first.")

def make_sequences(df, seq_length=10, dropna=True):
    """
    Return X (samples, seq_length, features), y (samples, features), scaler
    features: heart_rate, respiratory_rate, temperature, spo2, systolic_bp, diastolic_bp
    """
    df = _rename_columns(df)
    if "time" not in df.columns:
        raise ValueError("DataFrame missing 'time' column. Ensure timeseries file or raw transformed by generator.")

    # canonical column names and ensure dtype
    vitals = ["heart_rate", "respiratory_rate", "temperature", "spo2", "systolic_bp", "diastolic_bp"]
    for c in vitals:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows with missing vitals
    if dropna:
        before = len(df)
        df = df.dropna(subset=vitals)
        after = len(df)
        if after == 0:
            raise ValueError("All rows have missing vital columns after dropna. Check input CSV or generator.")
        print(f"Dropped {before-after} rows with missing vitals. Rows remaining: {after}")

    # ensure time and sorting
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(["patient_id", "time"])

    # global scaling
    scaler = MinMaxScaler()
    df[vitals] = scaler.fit_transform(df[vitals])

    sequences = []
    targets = []
    patient_ids = []
    for pid, group in df.groupby("patient_id"):
        vals = group[vitals].values
        n = len(vals)
        if n <= seq_length:
            continue
        for i in range(n - seq_length):
            sequences.append(vals[i:i+seq_length])
            targets.append(vals[i+seq_length])
            patient_ids.append(pid)

    X = np.array(sequences, dtype=np.float32)
    y = np.array(targets, dtype=np.float32)
    print(f"Created sequences: X.shape={X.shape}, y.shape={y.shape}, patients_with_sequences={len(set(patient_ids))}")
    return X, y, scaler, vitals
