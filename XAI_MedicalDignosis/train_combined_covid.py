from pathlib import Path
from collections import defaultdict
from datetime import datetime
import hashlib
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path("/mnt/d/MCA_Final_Project/XAI_MedicalDignosis")
OLD_DATA_DIR = Path("/mnt/d/MCA_Final_Project/KDatasets_Clean")

# Correct path based on the folder location you found with `find`.
COVID_DATA_DIR = Path(
    "/mnt/d/MCA_Final_Project/covid19Dataset/COVID-19_Radiography_Dataset"
)

OLD_MODEL_PATH = PROJECT_DIR / "models" / "kaggle_chest_xray_clean_best.keras"

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

STAGE1_PATH = MODEL_DIR / "covid5_stage1_best.keras"
STAGE2_PATH = MODEL_DIR / "covid5_stage2_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "kaggle_chest_xray_covid5_best.keras"

IMAGE_SIZE = 224
BATCH_SIZE = 8
SEED = 42
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 12

CLASS_NAMES = [
    "normal",
    "pneumonia",
    "tuberculosis",
    "covid",
    "lung_opacity",
]
NUM_CLASSES = len(CLASS_NAMES)

LABELS = {
    "normal": 0,
    "pneumonia": 1,
    "tuberculosis": 2,
    "covid": 3,
    "lung_opacity": 4,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
AUTOTUNE = tf.data.AUTOTUNE

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. VALIDATE PATHS AND OUTPUT FILES
# ============================================================

if not OLD_DATA_DIR.is_dir():
    raise FileNotFoundError(f"Clean dataset not found: {OLD_DATA_DIR}")

if not COVID_DATA_DIR.is_dir():
    raise FileNotFoundError(
        f"COVID dataset not found: {COVID_DATA_DIR}\n"
        "Check the path in COVID_DATA_DIR near the top of this script."
    )

if not OLD_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Existing checkpoint not found: {OLD_MODEL_PATH}")

for output_path in (STAGE1_PATH, STAGE2_PATH, FINAL_MODEL_PATH):
    if output_path.exists():
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "To avoid overwriting an earlier experiment, rename or move "
            "the existing file(s) before running this script again."
        )

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
print("COVID dataset directory:", COVID_DATA_DIR)


# ============================================================
# 3. DATASET HELPERS
# ============================================================

def normalize_name(name):
    return "".join(char.lower() for char in name if char.isalnum())


def find_class_folder(root, aliases):
    """Find exactly one direct child folder matching an alias."""
    if not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {root}")

    alias_names = {normalize_name(alias) for alias in aliases}
    matches = [
        path for path in root.iterdir()
        if path.is_dir() and normalize_name(path.name) in alias_names
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one folder matching {aliases} inside {root}; "
            f"found {[path.name for path in matches]}"
        )
    return matches[0]


def image_files(folder):
    """
    Prefer a class-level 'images' directory when present, so mask images
    are not accidentally included. Otherwise search within the class folder.
    """
    images_folder = folder / "images"
    search_root = images_folder if images_folder.is_dir() else folder

    # If there is no images/ directory but a masks/ directory exists,
    # exclude files under masks/ as a safety measure.
    files = []
    for path in search_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not images_folder.is_dir() and "masks" in {
            normalize_name(part) for part in path.parts
        }:
            continue
        files.append(path)
    return sorted(files)


def make_record(path, label, source, split=None, source_class=None):
    return {
        "path": str(path),
        "label": int(label),
        "source": source,
        "split": split,
        "source_class": source_class,
    }


# ============================================================
# 4. READ EXISTING CLEAN DATASET
# ============================================================

def collect_old_records():
    records = []
    old_folder_mapping = [
        (["normal"], LABELS["normal"]),
        (["pneumonia"], LABELS["pneumonia"]),
        (["tuberculosis", "tb"], LABELS["tuberculosis"]),
    ]

    for split in ("train", "val", "test"):
        split_dir = OLD_DATA_DIR / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Expected old dataset split: {split_dir}")

        for aliases, label in old_folder_mapping:
            folder = find_class_folder(split_dir, aliases)
            files = image_files(folder)

            if not files:
                raise ValueError(f"No images found in old dataset folder: {folder}")

            records.extend(
                make_record(
                    path=path,
                    label=label,
                    source="old",
                    split=split,
                    source_class=folder.name,
                )
                for path in files
            )
            print(f"Old {split:5s} | {folder.name:18s} | {len(files)} images")

    return records


# ============================================================
# 5. READ COVID RADIOGRAPHY DATASET
# ============================================================

def collect_covid_records():
    records = []
    category_mapping = [
        (["COVID", "COVID-19"], LABELS["covid"], "covid"),
        (["Normal"], LABELS["normal"], "normal"),
        (["Lung_Opacity", "Lung Opacity"], LABELS["lung_opacity"], "lung_opacity"),
        (["Viral Pneumonia", "Viral_Pneumonia"], LABELS["pneumonia"], "viral_pneumonia"),
    ]

    for aliases, label, source_class in category_mapping:
        folder = find_class_folder(COVID_DATA_DIR, aliases)
        files = image_files(folder)
        if not files:
            raise ValueError(
                f"No images found in {folder}. Check whether this class has "
                "an 'images' subfolder and whether it contains image files."
            )

        records.extend(
            make_record(
                path=path,
                label=label,
                source="covid_dataset",
                split=None,
                source_class=source_class,
            )
            for path in files
        )
        print(f"COVID source | {folder.name:20s} | {len(files)} images")

    return records


# ============================================================
# 6. REMOVE EXACT DUPLICATES AND CONFLICTING LABELS
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deduplicate_records(records):
    """
    Remove byte-identical duplicates. Exact duplicates with conflicting
    labels are excluded. Same-label duplicates prefer the old dataset record.
    This does not detect visually similar, resized, or re-encoded images.
    """
    groups = defaultdict(list)
    print("\nChecking exact image hashes; this may take a few minutes.")

    for index, record in enumerate(records, start=1):
        try:
            digest = sha256_file(record["path"])
        except OSError as error:
            print("Unreadable file skipped:", record["path"], error)
            continue

        groups[digest].append(record)
        if index % 2000 == 0:
            print(f"Hashed {index}/{len(records)} images")

    kept = []
    conflicting_groups = 0
    duplicate_groups = 0
    split_priority = {"train": 0, "val": 1, "test": 2, None: 3}

    for group in groups.values():
        labels_in_group = {record["label"] for record in group}
        if len(labels_in_group) > 1:
            conflicting_groups += 1
            continue

        if len(group) > 1:
            duplicate_groups += 1

        old_records = [record for record in group if record["source"] == "old"]
        if old_records:
            chosen = sorted(
                old_records,
                key=lambda record: split_priority[record["split"]],
            )[0]
        else:
            chosen = group[0]
        kept.append(chosen)

    print("Exact duplicate groups:", duplicate_groups)
    print("Conflicting-label groups excluded:", conflicting_groups)
    print("Images retained:", len(kept))
    return kept


# ============================================================
# 7. SPLIT NEW DATA; PRESERVE OLD DATASET SPLITS
# ============================================================

def split_new_records(records):
    if len(records) < 10:
        raise ValueError("Too few COVID-dataset images remain after deduplication.")

    source_labels = [record["source_class"] for record in records]
    train_records, temporary_records = train_test_split(
        records,
        test_size=0.30,
        random_state=SEED,
        stratify=source_labels,
    )

    temporary_labels = [record["source_class"] for record in temporary_records]
    val_records, test_records = train_test_split(
        temporary_records,
        test_size=0.50,
        random_state=SEED,
        stratify=temporary_labels,
    )
    return train_records, val_records, test_records


# ============================================================
# 8. COMBINE DATASETS
# ============================================================

all_records = collect_old_records() + collect_covid_records()
all_records = deduplicate_records(all_records)

old_train = [
    record for record in all_records
    if record["source"] == "old" and record["split"] == "train"
]
old_val = [
    record for record in all_records
    if record["source"] == "old" and record["split"] == "val"
]
old_test = [
    record for record in all_records
    if record["source"] == "old" and record["split"] == "test"
]
new_records = [record for record in all_records if record["source"] == "covid_dataset"]

covid_train, covid_val, covid_test = split_new_records(new_records)
train_records = old_train + covid_train
val_records = old_val + covid_val
test_records = old_test + covid_test

if not train_records or not val_records or not test_records:
    raise ValueError("One or more combined data splits are empty.")


def print_distribution(name, records):
    counts = np.bincount(
        [record["label"] for record in records],
        minlength=NUM_CLASSES,
    )
    print(f"\n{name} distribution:")
    for class_name, count in zip(CLASS_NAMES, counts):
        print(f"  {class_name:15s}: {count}")
    print("  Total:", len(records))


print_distribution("TRAIN", train_records)
print_distribution("VALIDATION", val_records)
print_distribution("TEST", test_records)


# ============================================================
# 9. BUILD TF.DATA PIPELINES
# ============================================================

def decode_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    # The transferred model includes Rescaling(1/255), so input stays 0-255.
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(records, training=False, class_weights=None):
    paths = np.asarray([record["path"] for record in records], dtype=str)
    labels = np.asarray([record["label"] for record in records], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    if class_weights is None:
        dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    else:
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        def decode_with_weight(path, label):
            image, label = decode_image(path, label)
            weight = tf.gather(weights_tensor, label)
            return image, label, weight

        dataset = dataset.map(decode_with_weight, num_parallel_calls=AUTOTUNE)

    return dataset.batch(BATCH_SIZE).prefetch(1)


y_train = np.asarray([record["label"] for record in train_records], dtype=np.int32)
present_classes = np.unique(y_train)
if len(present_classes) != NUM_CLASSES:
    raise ValueError("Every one of the five classes must have training images.")

computed_weights = compute_class_weight(
    class_weight="balanced",
    classes=present_classes,
    y=y_train,
)
class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
class_weights[present_classes] = computed_weights
print("\nClass weights:", class_weights)

train_ds = make_dataset(train_records, training=True, class_weights=class_weights)
val_ds = make_dataset(val_records)
test_ds = make_dataset(test_records)


# ============================================================
# 10. INITIALIZE FIVE-CLASS MODEL FROM THE EXISTING CNN
# ============================================================

print("\nLoading original model for feature transfer...")
old_model = tf.keras.models.load_model(OLD_MODEL_PATH, compile=False)

if old_model.output_shape[-1] != 3:
    raise ValueError(
        f"Expected existing model to have 3 outputs, got {old_model.output_shape}."
    )

# Existing architecture ends with Dense(256), Dropout, Dense(3).
# The layer before the original output is the Dropout representation.
feature_model = tf.keras.Model(
    inputs=old_model.input,
    outputs=old_model.layers[-2].output,
    name="transferred_chest_xray_features",
)

new_output = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="combined_classifier",
)(feature_model.output)

model = tf.keras.Model(
    inputs=feature_model.input,
    outputs=new_output,
    name="chest_xray_five_class_model",
)

print("New model output shape:", model.output_shape)
print("Class order:", CLASS_NAMES)


# ============================================================
# 11. PHASE ONE: TRAIN THE NEW CLASSIFICATION HEAD
# ============================================================

print("\nPHASE 1: Training the new five-class output head")
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stage1_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(STAGE1_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / "covid5_stage1_log.csv")),
    tf.keras.callbacks.TerminateOnNaN(),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=stage1_callbacks,
)


# ============================================================
# 12. PHASE TWO: FINE-TUNE SELECTED CNN LAYERS
# ============================================================

print("\nPHASE 2: Fine-tuning selected layers")
model = tf.keras.models.load_model(STAGE1_PATH, compile=False)

for layer in model.layers:
    layer.trainable = False

conv_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Conv2D)
]
if len(conv_layers) < 2:
    raise ValueError("Could not find the expected convolutional layers.")

for layer in conv_layers[-2:]:
    layer.trainable = True

# Dense feature layer and new output head train; BatchNorm stays frozen.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

stage2_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(STAGE2_PATH),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(RESULTS_DIR / "covid5_stage2_log.csv")),
    tf.keras.callbacks.TerminateOnNaN(),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=stage2_callbacks,
)


# ============================================================
# 13. SELECT CHECKPOINT USING VALIDATION LOSS
# ============================================================

print("\nComparing checkpoints on validation data...")


def validation_loss(checkpoint_path):
    candidate = tf.keras.models.load_model(checkpoint_path, compile=False)
    candidate.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    values = candidate.evaluate(val_ds, verbose=0, return_dict=True)
    return float(values["loss"]), float(values["accuracy"])


stage1_loss, stage1_accuracy = validation_loss(STAGE1_PATH)
stage2_loss, stage2_accuracy = validation_loss(STAGE2_PATH)

print(
    f"Stage 1 validation loss={stage1_loss:.4f}, "
    f"accuracy={stage1_accuracy:.4f}"
)
print(
    f"Stage 2 validation loss={stage2_loss:.4f}, "
    f"accuracy={stage2_accuracy:.4f}"
)

best_checkpoint = STAGE1_PATH if stage1_loss <= stage2_loss else STAGE2_PATH
shutil.copy2(best_checkpoint, FINAL_MODEL_PATH)

print("Selected checkpoint:", best_checkpoint)
print("New final model:", FINAL_MODEL_PATH)


# ============================================================
# 14. FINAL EVALUATION ON THE HELD-OUT TEST SET
# ============================================================

print("\nEvaluating selected model on the test set...")
final_model = tf.keras.models.load_model(FINAL_MODEL_PATH, compile=False)

# Preserve test order; test_ds is not shuffled.
test_labels = np.concatenate([labels.numpy() for _, labels in test_ds], axis=0)
scores = final_model.predict(test_ds, verbose=1)
predictions = np.argmax(scores, axis=1)

accuracy = accuracy_score(test_labels, predictions)
precision_macro = precision_score(
    test_labels, predictions, average="macro", zero_division=0
)
recall_macro = recall_score(
    test_labels, predictions, average="macro", zero_division=0
)
f1_macro = f1_score(
    test_labels, predictions, average="macro", zero_division=0
)
f1_weighted = f1_score(
    test_labels, predictions, average="weighted", zero_division=0
)

cm = confusion_matrix(test_labels, predictions, labels=list(range(NUM_CLASSES)))
report = classification_report(
    test_labels,
    predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)

print("\n" + "=" * 60)
print("FIVE-CLASS TEST RESULTS")
print("=" * 60)
print(f"Accuracy:            {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Macro precision:     {precision_macro:.4f}")
print(f"Macro recall:        {recall_macro:.4f}")
print(f"Macro F1-score:      {f1_macro:.4f}")
print(f"Weighted F1-score:   {f1_weighted:.4f}")
print("\nConfusion matrix:\n", cm)
print("\nClassification report:\n", report)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = RESULTS_DIR / f"covid5_test_report_{timestamp}.txt"
matrix_path = RESULTS_DIR / f"covid5_confusion_matrix_{timestamp}.png"
predictions_path = RESULTS_DIR / f"covid5_predictions_{timestamp}.csv"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(f"Model: {FINAL_MODEL_PATH}\n")
    file.write(f"Class order: {CLASS_NAMES}\n")
    file.write(f"Test image count: {len(test_labels)}\n\n")
    file.write(f"Accuracy: {accuracy:.6f}\n")
    file.write(f"Macro precision: {precision_macro:.6f}\n")
    file.write(f"Macro recall: {recall_macro:.6f}\n")
    file.write(f"Macro F1-score: {f1_macro:.6f}\n")
    file.write(f"Weighted F1-score: {f1_weighted:.6f}\n\n")
    file.write("Confusion matrix:\n")
    file.write(np.array2string(cm))
    file.write("\n\nClassification report:\n")
    file.write(report)

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES,
).plot(
    ax=ax,
    values_format="d",
    xticks_rotation=35,
    colorbar=False,
)
ax.set_title("Five-Class Chest X-Ray Confusion Matrix")
fig.tight_layout()
fig.savefig(matrix_path, dpi=160, bbox_inches="tight")
plt.close(fig)

# FIXED: tf.data.Dataset has no `file_paths` attribute.
prediction_data = {
    "image_path": [record["path"] for record in test_records],
    "actual_class": [CLASS_NAMES[index] for index in test_labels],
    "predicted_class": [CLASS_NAMES[index] for index in predictions],
    "correct": test_labels == predictions,
}
for index, class_name in enumerate(CLASS_NAMES):
    prediction_data[f"score_{class_name}"] = scores[:, index]

pd.DataFrame(prediction_data).to_csv(predictions_path, index=False)

print("\nSaved:")
print("New model:", FINAL_MODEL_PATH)
print("Test report:", report_path)
print("Confusion matrix:", matrix_path)
print("Predictions CSV:", predictions_path)
print("\nTraining and evaluation completed.")
print("The original three-class checkpoint was not modified.")
