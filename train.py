import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths to uploaded CSVs
DATA_DIR = "C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/testing data"

PATIENTS = os.path.join(DATA_DIR, "PATIENTS.csv")
ADMISSIONS = os.path.join(DATA_DIR, "ADMISSIONS.csv")
DIAGNOSES = os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv")
D_ICD = os.path.join(DATA_DIR, "D_ICD_DIAGNOSES.csv")
LABS = os.path.join(DATA_DIR, "LABEVENTS.csv")
D_LABS = os.path.join(DATA_DIR, "D_LABITEMS.csv")

MODEL_DIR = "models/classification"
os.makedirs(MODEL_DIR, exist_ok=True)


# ----------------- Feature Builder -----------------
def build_classification_dataset(patients, admissions, diagnoses, diag_dict, labs, lab_dict):
    # Lowercase all column names
    for df in [patients, admissions, diagnoses, diag_dict, labs, lab_dict]:
        df.columns = df.columns.str.lower()

    # Merge admissions + patients
    df = admissions.merge(patients, on="subject_id", how="left")

    # Diabetes flag from diagnoses
    diagnoses = diagnoses.merge(diag_dict, on="icd9_code", how="left")
    diagnoses["is_diabetes"] = diagnoses["icd9_code"].astype(str).str.startswith("250").astype(int)
    diabetes_flags = diagnoses.groupby("subject_id")["is_diabetes"].max().reset_index()
    df = df.merge(diabetes_flags, on="subject_id", how="left")
    df["is_diabetes"] = df["is_diabetes"].fillna(0).astype(int)

    # Lab features (subset of common ones)
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

    features = [c for c in df.columns if c not in ["is_diabetes"]]
    target = "is_diabetes"
    return df[features], df[target]


# ----------------- Models -----------------
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)
    return model

def build_nn(input_dim):
    model = Sequential([
        Dense(64, activation="relu", input_dim=input_dim),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def evaluate(model, X_test, y_test, keras=False):
    if keras:
        preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    else:
        preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


# ----------------- Main -----------------
def main():
    # Load raw tables
    patients = pd.read_csv(PATIENTS)
    admissions = pd.read_csv(ADMISSIONS)
    diagnoses = pd.read_csv(DIAGNOSES)
    diag_dict = pd.read_csv(D_ICD)
    labs = pd.read_csv(LABS)
    lab_dict = pd.read_csv(D_LABS)

    # Build dataset
    X, y = build_classification_dataset(patients, admissions, diagnoses, diag_dict, labs, lab_dict)

    # Handle missing values (numeric only)
    X_numeric = X.select_dtypes(include=["number"])
    X_numeric = X_numeric.fillna(X_numeric.median())

    # Keep non-numeric columns separate
    X_non_numeric = X.select_dtypes(exclude=["number"])

    # Combine back
    X = pd.concat([X_numeric, X_non_numeric], axis=1)

    # Scale only numeric columns
    scaler = StandardScaler()
    X_scaled_numeric = scaler.fit_transform(X_numeric)

    # Replace numeric part with scaled values
    X_scaled = pd.DataFrame(X_scaled_numeric, columns=X_numeric.columns, index=X.index)

    # Add back non-numeric features (optional: encode later)
    X_scaled = pd.concat([X_scaled, X_non_numeric.reset_index(drop=True)], axis=1)


    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=["float64", "int64"]))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = train_logistic(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic.pkl"))
    print("LR:", evaluate(lr, X_test, y_test))

    # XGBoost
    xgb_model = train_xgb(X_train, y_train)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb.pkl"))
    print("XGB:", evaluate(xgb_model, X_test, y_test))

    # Neural Net
    nn = build_nn(X_train.shape[1])
    nn.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)
    nn.save(os.path.join(MODEL_DIR, "nn.h5"))
    print("NN:", evaluate(nn, X_test, y_test, keras=True))


if __name__ == "__main__":
    main()

