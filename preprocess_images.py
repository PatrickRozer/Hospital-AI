import tensorflow as tf
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 100

def load_datasets(data_dir=r"C:/Users/Bernietta/OneDrive/guvi/guvi_project/main_project/images"):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Normalize
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds
