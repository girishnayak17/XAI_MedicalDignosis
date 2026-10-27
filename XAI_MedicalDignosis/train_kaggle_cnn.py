
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets_Clean")

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# New filenames: never overwrite the old model.
BEST_MODEL_PATH = MODEL_DIR / "kaggle_chest_xray_clean_best.keras"
LOG_PATH = RESULTS_DIR / "kaggle_chest_xray_clean_training_log.csv"
REPORT_PATH = RESULTS_DIR / "kaggle_chest_xray_clean_test_report.txt"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
SEED = 42

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SAFETY CHECKS
# ============================================================

if BEST_MODEL_PATH.exists():
    raise FileExistsError(
        f"{BEST_MODEL_PATH} already exists. "
        "Choose another output filename to avoid overwriting it."
    )

for split in ["train", "val", "test"]:
    for class_name in CLASS_NAMES:
        folder = DATA_DIR / split / class_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Missing directory: {folder}")

        count = sum(1 for p in folder.iterdir() if p.is_file())
        if count == 0:
            raise ValueError(f"No images found in {folder}")

print("=" * 65)
print("CUSTOM CNN TRAINING ON CANDIDATE CLEANED DATASET")
print("=" * 65)
print("TensorFlow:", tf.__version__)
print("Dataset:", DATA_DIR)
print("Classes:", CLASS_NAMES)

# ============================================================
# GPU SETUP
# ============================================================

gpus = tf.config.list_physical_devices("GPU")
print("GPUs detected:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU configuration notice:", exc)

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA_DIR / split,
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    ).prefetch(1)


print("\nLoading datasets...")

train_ds = load_dataset("train", shuffle=True)
val_ds = load_dataset("val", shuffle=False)
test_ds = load_dataset("test", shuffle=False)

print("Class index mapping:", dict(enumerate(CLASS_NAMES)))

# ============================================================
# RECREATE THE EXISTING CNN ARCHITECTURE
# ============================================================

def build_custom_cnn():
    inputs = keras.Input(shape=(224, 224, 3), name="input_layer")

    # Match the existing model's augmentation layers.
    x = keras.layers.RandomFlip("horizontal", name="random_flip")(inputs)
    x = keras.layers.RandomRotation(
        0.05, name="random_rotation"
    )(x)
    x = keras.layers.RandomZoom(
        0.10, name="random_zoom"
    )(x)

    # Input images are 0-255. Normalize inside the model.
    x = keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(x)

    for filters in [32, 64, 128, 256]:
        x = keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=True,
        )(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling2d"
    )(x)
    x = keras.layers.Dense(256, activation="relu", name="dense")(x)
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    outputs = keras.layers.Dense(
        3, activation="softmax", name="dense_1"
    )(x)

    return keras.Model(inputs, outputs, name="custom_chest_xray_cnn")


model = build_custom_cnn()
model.summary()

# ============================================================
# CLASS WEIGHTS
# Training counts in KDatasets_Clean:
# normal=2829, pneumonia=3401, tuberculosis=5305
# ============================================================

train_counts = {
    0: 2829,
    1: 3401,
    2: 5305,
}

total = sum(train_counts.values())
class_weights = {
    class_id: total / (len(CLASS_NAMES) * count)
    for class_id, count in train_counts.items()
}

print("\nClass weights:", class_weights)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# ============================================================
# CALLBACKS
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    keras.callbacks.CSVLogger(str(LOG_PATH)),
    keras.callbacks.TerminateOnNaN(),
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")
print("Batch size:", BATCH_SIZE)
print("Maximum epochs:", EPOCHS)
print("Checkpoint:", BEST_MODEL_PATH)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

if not BEST_MODEL_PATH.is_file():
    raise RuntimeError("No best checkpoint was saved.")

# ============================================================
# FINAL TEST EVALUATION
# Use the best validation-loss checkpoint.
# Do not tune the model on test results.
# ============================================================

print("\nLoading best checkpoint...")
best_model = keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False,
)

y_true = np.concatenate([
    labels.numpy() for _, labels in test_ds
])

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

print("\n" + "=" * 65)
print("HELD-OUT TEST RESULTS")
print("=" * 65)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print("\nClassification report:\n", report)
print("Confusion matrix (true rows, predicted columns):")
print(matrix)

with REPORT_PATH.open("w", encoding="utf-8") as f:
    f.write("Custom CNN - Candidate Cleaned Dataset\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n")
    f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nConfusion matrix (true rows, predicted columns):\n")
    f.write(np.array2string(matrix))
    f.write("\n")

print("\nBest model saved to:", BEST_MODEL_PATH)
print("Training log saved to:", LOG_PATH)
print("Test report saved to:", REPORT_PATH)
print("\nTraining and evaluation completed.")
