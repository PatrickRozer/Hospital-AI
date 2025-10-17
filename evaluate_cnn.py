import tensorflow as tf
import os
from src.imaging.preprocess_images import load_datasets
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_DIR = "models/imaging"

def main():
    _, _, test_ds = load_datasets()
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_chestxray.h5"))

    y_true = []
    y_pred = []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
