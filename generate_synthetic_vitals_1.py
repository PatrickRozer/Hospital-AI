"""
Generate synthetic time-series vitals from a snapshot CSV.
Saves: data/vitals/human_vital_signs_timeseries.csv
"""
import os
import pandas as pd
import numpy as np

RAW_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/human_vital_signs_dataset_2024.csv", "human_vital_signs_dataset_2024.csv")
OUT_FILE = os.path.join("C:/Users/Bernietta/OneDrive", "guvi/guvi_project/main_project/src/timeseries", "human_vital_signs_timeseries_1.csv")
READINGS_PER_PATIENT = 20   # change if you need longer sequences
MINUTES_INTERVAL = 5

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

def generate(readings_per_patient=READINGS_PER_PATIENT, minutes_interval=MINUTES_INTERVAL):
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(f"Raw file not found: {RAW_FILE}. Put your CSV at this path.")
    df = pd.read_csv(RAW_FILE, low_memory=False)
    df = _rename_columns(df)

    # Ensure patient_id present
    if "patient_id" not in df.columns:
        raise ValueError("Input file missing 'Patient ID' column. Please check header.")

    numeric_cols = ["heart_rate", "respiratory_rate", "temperature",
                    "spo2", "systolic_bp", "diastolic_bp"]
    # coerce numeric where possible
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    rows = []
    np.random.seed(42)
    for _, r in df.iterrows():
        pid = r["patient_id"]
        # if time present convert, else use now as base
        base_time = pd.to_datetime(r["time"]) if "time" in r and pd.notna(r["time"]) else pd.Timestamp.now()
        for i in range(readings_per_patient):
            ts = base_time + pd.Timedelta(minutes=minutes_interval * i)
            rec = {"patient_id": pid, "time": ts}
            # jitter numeric vitals with small physiological noise
            rec["heart_rate"] = float(r["heart_rate"]) + np.random.normal(0, 3) if pd.notna(r["heart_rate"]) else np.nan
            rec["respiratory_rate"] = float(r["respiratory_rate"]) + np.random.normal(0, 0.8) if pd.notna(r["respiratory_rate"]) else np.nan
            rec["temperature"] = float(r["temperature"]) + np.random.normal(0, 0.08) if pd.notna(r["temperature"]) else np.nan
            rec["spo2"] = float(r["spo2"]) + np.random.normal(0, 0.4) if pd.notna(r["spo2"]) else np.nan
            rec["systolic_bp"] = float(r["systolic_bp"]) + np.random.normal(0, 2) if pd.notna(r["systolic_bp"]) else np.nan
            rec["diastolic_bp"] = float(r["diastolic_bp"]) + np.random.normal(0, 1.5) if pd.notna(r["diastolic_bp"]) else np.nan
            rows.append(rec)

    df_long = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df_long.to_csv(OUT_FILE, index=False)
    print(f"✅ Generated timeseries: {OUT_FILE}  (patients: {df_long['patient_id'].nunique()}, rows: {len(df_long)})")
    return OUT_FILE

if __name__ == "__main__":
    generate()
