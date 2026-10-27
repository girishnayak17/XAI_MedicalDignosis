
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# Dataset and model paths
DATASET_DIR = "/mnt/d/MCA_Final_Project/KDatasets"
MODEL_PATHS = {
    "Original": "models/kaggle_chest_xray_best.keras",
    "V2": "models/kaggle_chest_xray_v2_best.keras",
}

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))

# Load the same test images for both models.
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    label_mode="int",
)

# Read labels before applying prefetch.
y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
test_ds = test_ds.prefetch(1)

print("\nTest images:", len(y_true))
print("Class order:", CLASS_NAMES)

# Inspect a sample image's pixel range.
sample_images, _ = next(iter(test_ds))
print("Input shape:", sample_images.shape)
print("Input pixel range:",
      float(tf.reduce_min(sample_images)),
      "to",
      float(tf.reduce_max(sample_images)))

for name, path in MODEL_PATHS.items():
    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("PATH:", path)

    if not os.path.exists(path):
        print("ERROR: Model file not found.")
        continue

    model = tf.keras.models.load_model(path, compile=False)
    print("Output shape:", model.output_shape)

    # Inspect preprocessing layers.
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Rescaling):
            print("Rescaling layer:", layer.get_config())

    # Predict using the exact same test dataset.
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)

    print("\nTest accuracy:",
          f"{np.mean(y_pred == y_true) * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]))

    # Display predictions for the first 10 test images.
    print("\nFirst 10 test predictions:")
    for i in range(min(10, len(y_true))):
        scores = probabilities[i]
        print(
            f"Image {i + 1:02d}: "
            f"actual={CLASS_NAMES[y_true[i]]:12s} "
            f"predicted={CLASS_NAMES[y_pred[i]]:12s} "
            f"scores={np.round(scores, 4)}"
        )
