import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.regression.preprocess_regression import load_and_build_los
from src.regression.models_regression import train_linear, train_rf, build_lstm, evaluate_regression

MODEL_DIR = "models/regression"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    X, y = load_and_build_los()

    # Identify feature types
    numeric_feats = ["age"]
    categorical_feats = ["gender", "ETHNICITY", "ADMISSION_TYPE"]


    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats)
    ])

    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = train_linear(X_train, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR, "linear.pkl"))
    print("Linear:", evaluate_regression(lr, X_test, y_test))

    # Random Forest
    rf = train_rf(X_train, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf.pkl"))
    print("RandomForest:", evaluate_regression(rf, X_test, y_test))

    # LSTM (toy example: reshape features as sequences)
    import numpy as np
    X_train_seq = np.expand_dims(X_train.toarray() if hasattr(X_train, "toarray") else X_train, axis=1)
    X_test_seq = np.expand_dims(X_test.toarray() if hasattr(X_test, "toarray") else X_test, axis=1)

    lstm = build_lstm(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm.fit(X_train_seq, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1)
    lstm.save(os.path.join(MODEL_DIR, "lstm.h5"))
    print("LSTM:", evaluate_regression(lstm, X_test_seq, y_test, keras=True))

if __name__ == "__main__":
    main()
