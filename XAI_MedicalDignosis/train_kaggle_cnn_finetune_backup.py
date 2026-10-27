
from pathlib import Path
import shutil
import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

OLD_DATA_DIR = Path(
    "/mnt/d/MCA_Final_Project/KDatasets"
)

NEW_DATA_DIR = Path(
    "/mnt/d/MCA_Final_Project/PDatasets/chest_xray"
)

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Keep the existing primary model filename.
BEST_MODEL_PATH = MODEL_DIR / "kaggle_chest_xray_best.keras"

# Temporary candidate and safety backup.
CANDIDATE_PATH = MODEL_DIR / "kaggle_chest_xray_candidate.keras"
BACKUP_PATH = MODEL_DIR / "kaggle_chest_xray_best_backup.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 8
SEED = 42

CLASS_NAMES = ["normal", "pneumonia", "tuberculosis"]

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# GPU SETUP
# ============================================================

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs detected:", gpus)

for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print("GPU configuration:", exc)

# ============================================================
# VALIDATE PATHS
# ============================================================

if not BEST_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Existing model not found: {BEST_MODEL_PATH}"
    )

for split in ["train", "val"]:
    if not (OLD_DATA_DIR / split).is_dir():
        raise FileNotFoundError(
            f"Missing original {split} folder"
        )

if not (NEW_DATA_DIR / "train").is_dir():
    raise FileNotFoundError(
        f"Missing new training folder: {NEW_DATA_DIR / 'train'}"
    )

for label in ["NORMAL", "PNEUMONIA"]:
    if not (NEW_DATA_DIR / "train" / label).is_dir():
        raise FileNotFoundError(
            f"Missing new training class: {label}"
        )

# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

print("\nLoading existing model...")
model = keras.models.load_model(
    BEST_MODEL_PATH,
    compile=False,
)

if model.output_shape[-1] != 3:
    raise ValueError(
        f"Expected 3 output classes, got {model.output_shape}"
    )

print("Model loaded:", BEST_MODEL_PATH)
print("Output classes:", CLASS_NAMES)

# ============================================================
# DATASETS
# ============================================================

def load_original_training():
    return keras.utils.image_dataset_from_directory(
        OLD_DATA_DIR / "train",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

def load_new_training():
    dataset = keras.utils.image_dataset_from_directory(
        NEW_DATA_DIR / "train",
        class_names=["NORMAL", "PNEUMONIA"],
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=SEED,
    )

    # NORMAL -> 0, PNEUMONIA -> 1.
    # These are compatible with the existing three-class output.
    return dataset.map(
        lambda images, labels: (
            images,
            tf.cast(labels, tf.int32),
        ),
        num_parallel_calls=1,
    )

def load_original_validation():
    return keras.utils.image_dataset_from_directory(
        OLD_DATA_DIR / "val",
        class_names=CLASS_NAMES,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    ).prefetch(1)

old_train = load_original_training()
new_train = load_new_training()
val_ds = load_original_validation()

print("\nOriginal dataset class order:", CLASS_NAMES)
print("New dataset class order: NORMAL=0, PNEUMONIA=1")

# Sample complete batches from both sources.
# Retain most batches from the original dataset so that
# Tuberculosis examples remain represented during training.
combined_train = tf.data.Dataset.sample_from_datasets(
    [old_train.repeat(), new_train.repeat()],
    weights=[0.7, 0.3],
    seed=SEED,
).take(2500).prefetch(1)

# ============================================================
# BASELINE VALIDATION
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nEvaluating original model on original validation set...")
baseline_results = model.evaluate(
    val_ds,
    verbose=1,
    return_dict=True,
)

baseline_loss = float(baseline_results["loss"])
baseline_accuracy = float(baseline_results["accuracy"])

print(f"Original validation loss: {baseline_loss:.4f}")
print(f"Original validation accuracy: {baseline_accuracy:.4f}")

# ============================================================
# CONTINUE TRAINING FROM EXISTING WEIGHTS
# ============================================================

callbacks = [
    keras.callbacks.ModelCheckpoint(
        str(CANDIDATE_PATH),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1,
    ),
]

print("\nStarting continued training...")
print("Original batches: 70% sampling weight")
print("New Normal/Pneumonia batches: 30% sampling weight")
print("Maximum batches per epoch: 2500")

history = model.fit(
    combined_train,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

if not CANDIDATE_PATH.is_file():
    raise RuntimeError(
        "No candidate checkpoint was saved. "
        "The original model has not been replaced."
    )

# ============================================================
# COMPARE CANDIDATE AGAINST ORIGINAL
# ============================================================

print("\nLoading best candidate...")
candidate = keras.models.load_model(
    CANDIDATE_PATH,
    compile=False,
)

candidate.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

candidate_results = candidate.evaluate(
    val_ds,
    verbose=1,
    return_dict=True,
)

candidate_loss = float(candidate_results["loss"])
candidate_accuracy = float(candidate_results["accuracy"])

print("\n========== VALIDATION COMPARISON ==========")
print(f"Original loss:    {baseline_loss:.4f}")
print(f"Candidate loss:   {candidate_loss:.4f}")
print(f"Original accuracy:  {baseline_accuracy:.4f}")
print(f"Candidate accuracy: {candidate_accuracy:.4f}")

# ============================================================
# REPLACE ONLY IF VALIDATION LOSS IMPROVES
# ============================================================

if candidate_loss < baseline_loss:
    print("\nCandidate has lower validation loss.")

    # Preserve an untouched backup of the current model.
    shutil.copy2(BEST_MODEL_PATH, BACKUP_PATH)

    # Replace the original filename with the candidate.
    shutil.copy2(CANDIDATE_PATH, BEST_MODEL_PATH)

    print("Updated primary model:", BEST_MODEL_PATH)
    print("Backup saved at:", BACKUP_PATH)
else:
    print(
        "\nCandidate did not improve validation loss. "
        "Keeping the original model unchanged."
    )

print("\nTraining and comparison finished.")
print("Primary model path:", BEST_MODEL_PATH)
