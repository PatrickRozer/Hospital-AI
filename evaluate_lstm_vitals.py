"""
Evaluate saved LSTM on full dataset sequences (prints MAE, RMSE, R2).
Saves predictions CSV: models/timeseries/predictions.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.timeseries.preprocess_vitals import load_timeseries, make_sequences

MODEL_DIR = os.path.join("models", "timeseries")
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_vitals.h5")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_lstm_vitals.py first.")
    df_ts = load_timeseries()
    X, y, scaler, vitals = make_sequences(df_ts, seq_length=10)

    if X.size == 0:
        print("No sequences to evaluate.")
        return

    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    try:
        r2 = r2_score(y, preds)
    except Exception:
        r2 = float("nan")

    print(f"Evaluation -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

    # Save a CSV with a few predictions (inverse-scaling not applied because saved scaler is in models folder)
    out_df = pd.DataFrame(np.hstack([y[:200], preds[:200]]),
                          columns=[f"true_{c}" for c in vitals] + [f"pred_{c}" for c in vitals])
    out_path = os.path.join(MODEL_DIR, "predictions_preview.csv")
    out_df.to_csv(out_path, index=False)
    print(f"Saved predictions preview to {out_path}")

if __name__ == "__main__":
    main()

