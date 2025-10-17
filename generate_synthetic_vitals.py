import pandas as pd
import numpy as np
import os

INPUT_FILE = r"C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/human_vital_signs_dataset_2024.csv/human_vital_signs_dataset_2024.csv"
OUTPUT_FILE = r"C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/timeseries/human_vital_signs_timeseries.csv"

def generate_time_series():
    df = pd.read_csv(INPUT_FILE)
    df = df.rename(columns={
        "Patient ID": "patient_id",
        "Timestamp": "time",
        "Heart Rate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "Body Temperature": "temperature",
        "Oxygen Saturation": "spo2",
        "Systolic Blood Pressure": "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp"
    })

    all_records = []
    np.random.seed(42)

    for _, row in df.iterrows():
        patient_id = row["patient_id"]
        base_time = pd.to_datetime(row["time"])

        # Generate 20 readings per patient (every 5 minutes)
        for i in range(20):
            timestamp = base_time + pd.Timedelta(minutes=5 * i)
            rec = row.copy()
            rec["time"] = timestamp

            # add small physiological noise
            rec["heart_rate"] += np.random.normal(0, 3)
            rec["respiratory_rate"] += np.random.normal(0, 0.8)
            rec["temperature"] += np.random.normal(0, 0.1)
            rec["spo2"] += np.random.normal(0, 0.4)
            rec["systolic_bp"] += np.random.normal(0, 2)
            rec["diastolic_bp"] += np.random.normal(0, 1.5)

            all_records.append(rec)

    df_long = pd.DataFrame(all_records)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_long.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Generated time-series vitals for {df['patient_id'].nunique()} patients × 20 readings each")
    print(f"📄 Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_time_series()
