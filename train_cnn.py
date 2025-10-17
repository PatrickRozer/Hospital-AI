import tensorflow as tf
import os
from src.imaging.preprocess_images import load_datasets

MODEL_DIR = "models/imaging"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_cnn(input_shape=(224, 224, 3), num_classes=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    train_ds, val_ds, test_ds = load_datasets()

    model = build_cnn()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Save
    model.save(os.path.join(MODEL_DIR, "cnn_chestxray.h5"))
    print("✅ CNN model saved to models/imaging/cnn_chestxray.h5")

if __name__ == "__main__":
    main()
