"""
prepare_multimodal.py
- Loads note embeddings and clustered notes
- Maps notes to SUBJECT_ID (using NOTEEVENTS if needed)
- Aggregates embeddings per SUBJECT_ID (mean pooling)
- Loads vitals data (with 'Risk Category'), computes per-patient features
- Merges into final multimodal dataset with risk label
"""

import os
import pandas as pd
import numpy as np
import torch

# -------------------- PATHS --------------------
EMB_FILE = os.path.join("models", "pretrained", "note_embeddings.pt")
CLUSTERED_NOTES = os.path.join(
    "C:/Users/Bernietta/OneDrive",
    "guvi/guvi_project/main_project/src/pretrained_models",
    "cleaned_notes.csv"
)
NOTEEVENTS = os.path.join(
    "C:/Users/Bernietta/OneDrive",
    "guvi/guvi_project/main_project/testing data",
    "NOTEEVENTS_sorted.csv"
)
VITALS_FILE = os.path.join(
    "C:/Users/Bernietta/OneDrive",
    "guvi/guvi_project/main_project/human_vital_signs_dataset_2024.csv",
    "human_vital_signs_dataset_2024.csv"
)
OUT_DIR = os.path.join(
    "C:/Users/Bernietta/OneDrive",
    "guvi/guvi_project/main_project/src/pretrained_models",
    "pretrained"
)
OUT_FILE = os.path.join(OUT_DIR, "multimodal_dataset.csv")

# -------------------------------------------------

def load_embeddings():
    if not os.path.exists(EMB_FILE):
        raise FileNotFoundError(f"Embeddings not found: {EMB_FILE}")
    emb = torch.load(EMB_FILE)
    emb = emb.numpy() if hasattr(emb, "numpy") else np.array(emb)
    print(f"✅ Loaded embeddings: shape={emb.shape}")
    return emb


def load_clustered_notes():
    if not os.path.exists(CLUSTERED_NOTES):
        raise FileNotFoundError(f"Clustered notes not found at {CLUSTERED_NOTES}")
    df = pd.read_csv(CLUSTERED_NOTES, low_memory=False)
    print(f"✅ Loaded clustered notes: {len(df)} rows")
    return df


def map_notes_to_subjects(df_notes):
    """Map notes to SUBJECT_ID using NOTEEVENTS if necessary"""
    df = df_notes.copy()
    if "SUBJECT_ID" in df.columns:
        print("Using SUBJECT_ID column from notes CSV.")
        return df.rename(columns={"SUBJECT_ID": "subject_id"})

    if "ROW_ID" in df.columns:
        print("Notes CSV has ROW_ID — mapping to SUBJECT_ID using NOTEEVENTS.csv...")
        notes_map = pd.read_csv(NOTEEVENTS, usecols=["ROW_ID", "SUBJECT_ID"], low_memory=False)
        merged = df.merge(notes_map, on="ROW_ID", how="left")
        merged = merged.rename(columns={"SUBJECT_ID": "subject_id"})
        print(f"✅ Mapped to {merged['subject_id'].nunique()} unique patients.")
        return merged

    raise ValueError("❌ No SUBJECT_ID or ROW_ID column found in clustered notes.")


def aggregate_embeddings_per_subject(df_notes, embeddings):
    """Aggregate embeddings by patient"""
    if len(df_notes) != embeddings.shape[0]:
        print("⚠️ Warning: Embedding count doesn't match notes rows; aligning by index.")
    df_notes = df_notes.reset_index(drop=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    merged = pd.concat([df_notes.reset_index(drop=True), emb_df], axis=1)
    merged = merged.dropna(subset=["subject_id"])
    merged["subject_id"] = merged["subject_id"].astype(int)
    # Only aggregate embedding columns
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    agg = merged.groupby("subject_id")[emb_cols].mean().reset_index()
    print(f"✅ Aggregated embeddings: {agg.shape[0]} patients.")
    return agg

def build_vitals_features():
    """Compute per-patient aggregated vitals with 'Risk Category' label"""
    if not os.path.exists(VITALS_FILE):
        raise FileNotFoundError(f"Vitals file not found: {VITALS_FILE}")
    df = pd.read_csv(VITALS_FILE, low_memory=False)

    # Standardize column names
    df = df.rename(columns={
        "Patient ID": "patient_id",
        "Heart Rate": "heart_rate",
        "Respiratory Rate": "respiratory_rate",
        "Body Temperature": "temperature",
        "Oxygen Saturation": "spo2",
        "Systolic Blood Pressure": "systolic_bp",
        "Diastolic Blood Pressure": "diastolic_bp",
        "Risk Category": "risk_category"
    })

    # Convert vitals to numeric
    vitals_cols = ["heart_rate", "respiratory_rate", "temperature", "spo2", "systolic_bp", "diastolic_bp"]
    for col in vitals_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Aggregate stats per patient
    agg = df.groupby("patient_id")[vitals_cols].agg(["mean", "std", "min", "max"])
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    # Risk label (convert to 0/1)
    df["risk_category"] = df["risk_category"].astype(str).str.strip().str.lower()
    risk_map = {"low risk": 0, "medium risk": 1, "high risk": 2}
    df["risk"] = df["risk_category"].map(risk_map)

    # Attach most frequent risk label per patient
    risk_df = df.groupby("patient_id")["risk"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
    agg = agg.merge(risk_df, on="patient_id", how="left")

    print(f"✅ Built vitals features for {len(agg)} patients. Label found: True")
    return agg


def main():
    emb = load_embeddings()
    notes = load_clustered_notes()
    notes_mapped = map_notes_to_subjects(notes)
    emb_per_patient = aggregate_embeddings_per_subject(notes_mapped, emb)
    vitals_feats = build_vitals_features()

    merged = emb_per_patient.merge(vitals_feats, left_on="subject_id", right_on="patient_id", how="inner")

    if merged.empty:
        print("❌ No merged rows found! Check ID consistency between notes and vitals.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    merged.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved multimodal dataset to {OUT_FILE} (rows: {len(merged)})")


if __name__ == "__main__":
    main()
