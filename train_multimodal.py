"""
Train multimodal classifier on concatenated text embeddings + vitals features.
Saves: RandomForest baseline and a Keras NN (saved in models/pretrained/)
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import tensorflow as tf

DATA_FILE = os.path.join("C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/pretrained_models/pretrained/multimodal_dataset.csv")
OUT_DIR = os.path.join("C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/src/pretrained_models/pretrained")
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_SEED = 42

def load_dataset(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Multimodal dataset not found at {path}. Run prepare_multimodal.py first.")
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded multimodal dataset: {len(df)} rows, columns: {df.columns.tolist()}")
    return df

def split_features_labels(df):
    # find embedding columns emb_0..emb_n
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    # find vitals aggregate columns (others)
    other_cols = [c for c in df.columns if c not in emb_cols + ["subject_id","patient_id","risk_label_raw","risk"]]
    # pick numeric other cols
    numeric_other = [c for c in other_cols if df[c].dtype.kind in "fi"]
    X_emb = df[emb_cols].values
    X_tab = df[numeric_other].fillna(0).values
    X = np.hstack([X_emb, X_tab])
    if "risk" in df.columns:
        y = df["risk"].astype(int).values
    else:
        raise ValueError("No 'risk' label present in multimodal dataset.")
    print(f"Feature sizes -> emb:{X_emb.shape}, tab:{X_tab.shape}, combined:{X.shape}")
    return X, y, emb_cols + numeric_other

def train_rf(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    probs = rf.predict_proba(X_val)[:,1] if hasattr(rf, "predict_proba") else None
    print("RandomForest Results:")
    print(classification_report(y_val, preds))
    if probs is not None:
        print("ROC AUC:", roc_auc_score(y_val, probs))
    return rf

def build_keras_nn(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    return model

def main():
    df = load_dataset()
    X, y, feature_names = split_features_labels(df)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RANDOM_SEED, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # RandomForest baseline
    rf = train_rf(X_train, y_train, X_val, y_val)
    joblib.dump(rf, os.path.join(OUT_DIR, "multimodal_rf.joblib"))

    # Keras NN
    nn = build_keras_nn(X_train.shape[1])
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    nn.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, callbacks=[early], verbose=1)
    nn.save(os.path.join(OUT_DIR, "multimodal_nn.h5"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "multimodal_scaler.joblib"))
    print("✅ Saved RandomForest, NN model and scaler in", OUT_DIR)

    # Evaluate NN
    preds = (nn.predict(X_val).ravel() >= 0.5).astype(int)
    probs = nn.predict(X_val).ravel()
    print("Neural Net Results:")
    print(classification_report(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))

if __name__ == "__main__":
    main()
