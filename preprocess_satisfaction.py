"""
preprocess_satisfaction.py
- Loads merged HCAHPS CSV
- Creates a sentiment label from 'HCAHPS Answer Description' and 'Patient Survey Star Rating'
- Produces a small text column 'text_input' (HCAHPS Question + Answer Description)
- Saves prepared CSV for modeling: models/chatbot/satisfaction_for_model.csv
"""
import os
import pandas as pd
import numpy as np

IN_FILE = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data/patient_satisfaction/merged_patient_satisfaction.csv"  # adjust path if needed
OUT_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/chatbot"
OUT_FILE = os.path.join(OUT_DIR, "satisfaction_for_model.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# mapping rules
POSITIVE_ANSWERS = {"always", "yes", "excellent", "very good", "4", "5"}
NEGATIVE_ANSWERS = {"never", "no", "poor", "fair", "sometimes", "1", "2"}
# We'll treat 'usually' and '3' as neutral by default
NEUTRAL_ANSWERS = {"usually", "sometimes", "3", "neutral"}

def map_answer_to_sentiment(ans_desc, star):
    # Normalize
    if pd.isna(ans_desc):
        ans_desc = ""
    a = str(ans_desc).strip().lower()
    # try answer text first
    if any(tok in a for tok in POSITIVE_ANSWERS):
        return 2  # 2 = positive
    if any(tok in a for tok in NEGATIVE_ANSWERS):
        return 0  # 0 = negative
    if any(tok in a for tok in NEUTRAL_ANSWERS):
        return 1  # 1 = neutral
    # fallback to star rating if present
    try:
        s = float(star)
        if s >= 4.0:
            return 2
        elif s >= 3.0:
            return 1
        else:
            return 0
    except Exception:
        return 1  # neutral default

def main():
    print("Loading merged HCAHPS dataset...")
    df = pd.read_csv(IN_FILE, low_memory=False)
    # create text_input
    df["HCAHPS Question"] = df["HCAHPS Question"].astype(str)
    df["HCAHPS Answer Description"] = df["HCAHPS Answer Description"].astype(str)
    df["text_input"] = df["HCAHPS Question"].str.strip() + " -- " + df["HCAHPS Answer Description"].str.strip()

    # create label
    df["sentiment"] = df.apply(lambda r: map_answer_to_sentiment(r["HCAHPS Answer Description"], r.get("Patient Survey Star Rating", np.nan)), axis=1)

    # keep columns useful for modeling and context
    keep = ["Facility ID","Facility Name","City","State","Year","text_input","HCAHPS Answer Percent","Patient Survey Star Rating","sentiment","HCAHPS Measure ID","HCAHPS Question","HCAHPS Answer Description"]
    keep_existing = [c for c in keep if c in df.columns]
    df_out = df[keep_existing].copy()

    df_out.to_csv(OUT_FILE, index=False)
    print(f"Saved preprocessed data to {OUT_FILE} (rows: {len(df_out)})")

if __name__ == "__main__":
    main()
