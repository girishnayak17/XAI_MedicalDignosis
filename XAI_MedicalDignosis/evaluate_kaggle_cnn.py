
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets")
MODEL_PATH = PROJECT_DIR / "models/kaggle_chest_xray_best.keras"
RESULTS_DIR = PROJECT_DIR / "results"

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

test_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR / "test",
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
).prefetch(1)

model = keras.models.load_model(MODEL_PATH)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

print("\nTest results")
print("=" * 55)
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
))

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
print("Confusion matrix:")
print(cm)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
)
display.plot(values_format="d")
plt.title("Kaggle CNN — Test Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "kaggle_cnn_confusion_matrix.png",
    dpi=150,
)
plt.close()

np.savez_compressed(
    RESULTS_DIR / "kaggle_cnn_test_predictions.npz",
    y_true=y_true,
    y_pred=y_pred,
    probabilities=probabilities,
    class_names=np.array(CLASS_NAMES),
)

print("\nSaved confusion matrix and test predictions in results/.")
