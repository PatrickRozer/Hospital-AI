"""
Train LSTM on vitals timeseries. This script:
 - loads or requires a timeseries CSV at data/vitals/human_vital_signs_timeseries.csv
 - builds sequences (seq_length adjustable)
 - trains an LSTM (early stopping & checkpoint)
 - saves model and scaler to models/timeseries/
"""
import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.abspath("src"))  # Ensure src/ is in the import path
from timeseries.preprocess_vitals import load_timeseries, make_sequences

MODEL_DIR = os.path.join("models", "timeseries")
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LENGTH = 10
BATCH_SIZE = 32
EPOCHS = 20
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(input_shape[-1], name="output_vitals")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def main():
    print("Loading timeseries (or error if none)...")
    df_ts = load_timeseries()
    X, y, scaler, vitals = make_sequences(df_ts, seq_length=SEQ_LENGTH)

    if X.size == 0:
        print("No sequences created. Exiting.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

    model = build_model(input_shape=(SEQ_LENGTH, X.shape[2]))
    ckpt_path = os.path.join(MODEL_DIR, "best_lstm_vitals.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_loss")
    ]

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1)

    # Save final model and artifacts
    final_model_path = os.path.join(MODEL_DIR, "lstm_vitals.h5")
    model.save(final_model_path)
    print(f"Saved final model: {final_model_path}")

    # Save scaler and a few samples for later inference / debugging
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    np.save(os.path.join(MODEL_DIR, "X_train_sample.npy"), X_train[:min(200, len(X_train))])
    np.save(os.path.join(MODEL_DIR, "y_train_sample.npy"), y_train[:min(200, len(y_train))])
    print(f"Saved scaler and sample arrays to {MODEL_DIR}")

    # Save training history plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,4))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss (mse)")
        plt.title("Training Loss")
        plt.grid(True)
        plt.savefig(os.path.join(MODEL_DIR, "training_loss.png"))
        print("Saved training_loss.png")
    except Exception as e:
        print("Could not save training plot (matplotlib missing?). Error:", e)

if __name__ == "__main__":
    main()
