from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from preprocessing import (
    load_metadata,
    load_image_list,
    create_split_dataframe,
    add_encoded_labels,
    load_and_preprocess_image,
    DISEASE_LABELS,
    IMAGE_SIZE,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "Datasets"

TRAIN_LIST = DATASET_DIR / "train_val_list.txt"
TEST_LIST = DATASET_DIR / "test_list.txt"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

IMAGE_HEIGHT = IMAGE_SIZE[0]
IMAGE_WIDTH = IMAGE_SIZE[1]

NUM_CLASSES = len(DISEASE_LABELS)

# Memory-safe settings
BATCH_SIZE = 8
EPOCHS = 20

VALIDATION_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# GPU CONFIGURATION
# ============================================================

print("=" * 70)
print("GPU CONFIGURATION")
print("=" * 70)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print("GPU detected:")

    for gpu in gpus:
        print(gpu)

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(
                gpu,
                True
            )

    except RuntimeError as error:
        print("GPU configuration warning:", error)

else:
    print("WARNING: No GPU detected.")

print()


# ============================================================
# PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

def create_patient_level_split():

    print("=" * 70)
    print("CREATING PATIENT-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    metadata = load_metadata()

    train_images = load_image_list(TRAIN_LIST)

    train_df = create_split_dataframe(
        train_images,
        metadata
    )

    train_df = add_encoded_labels(train_df)

    print(f"Total training-pool images: {len(train_df)}")

    print(
        f"Unique patients: "
        f"{train_df['Patient ID'].nunique()}"
    )

    # --------------------------------------------------------
    # Patient-level split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, validation_indices = next(
        splitter.split(
            train_df,
            groups=train_df["Patient ID"]
        )
    )

    actual_train_df = (
        train_df
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    validation_df = (
        train_df
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Verify patient separation
    # --------------------------------------------------------

    train_patients = set(
        actual_train_df["Patient ID"]
    )

    validation_patients = set(
        validation_df["Patient ID"]
    )

    patient_overlap = (
        train_patients &
        validation_patients
    )

    print()
    print("Split results:")

    print(
        f"Training images: "
        f"{len(actual_train_df)}"
    )

    print(
        f"Validation images: "
        f"{len(validation_df)}"
    )

    print(
        f"Training patients: "
        f"{len(train_patients)}"
    )

    print(
        f"Validation patients: "
        f"{len(validation_patients)}"
    )

    print(
        f"Patient overlap: "
        f"{len(patient_overlap)}"
    )

    if patient_overlap:
        raise RuntimeError(
            "ERROR: Patient leakage detected!"
        )

    print("Patient-level split verified successfully.")

    return actual_train_df, validation_df


# ============================================================
# TENSORFLOW DATASET
# ============================================================

def create_dataset(
    dataframe,
    shuffle=False
):

    image_names = dataframe[
        "Image Index"
    ].values

    labels = np.stack(
        dataframe[
            "Encoded Labels"
        ].values
    ).astype(np.float32)

    def generator():

        for image_name, label in zip(
            image_names,
            labels
        ):

            image = load_and_preprocess_image(
                image_name
            )

            yield image, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    3
                ),
                dtype=tf.float32
            ),

            tf.TensorSpec(
                shape=(NUM_CLASSES,),
                dtype=tf.float32
            )
        )
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=512,
            seed=RANDOM_STATE,
            reshuffle_each_iteration=True
        )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    # Small prefetch buffer instead of AUTOTUNE
    # to reduce RAM pressure.
    dataset = dataset.prefetch(
        1
    )

    return dataset


# ============================================================
# CUSTOM CNN MODEL
# ============================================================

def build_cnn_model():

    model = tf.keras.Sequential(
        [

            # ------------------------------------------------
            # Block 1
            # ------------------------------------------------

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=(
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    3
                )
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Block 2
            # ------------------------------------------------

            tf.keras.layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Block 3
            # ------------------------------------------------

            tf.keras.layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Block 4
            # ------------------------------------------------

            tf.keras.layers.Conv2D(
                256,
                (3, 3),
                padding="same",
                activation="relu"
            ),

            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            # ------------------------------------------------
            # Classification head
            # ------------------------------------------------

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(
                256,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.5
            ),

            # ------------------------------------------------
            # Multi-label output
            # ------------------------------------------------

            tf.keras.layers.Dense(
                NUM_CLASSES,
                activation="sigmoid",
                name="disease_output"
            )
        ],

        name="Custom_CNN"
    )

    return model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_model(model):

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),

        loss="binary_crossentropy",

        metrics=[

            tf.keras.metrics.BinaryAccuracy(
                name="binary_accuracy"
            ),

            tf.keras.metrics.AUC(
                name="auc",
                multi_label=True
            ),

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

def save_history(history):

    history_path = (
        RESULTS_DIR /
        "cnn_training_history.json"
    )

    history_data = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history_data,
            file,
            indent=4
        )

    print(
        f"Training history saved to:\n"
        f"{history_path}"
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("CUSTOM CNN TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Create patient-level split
    # --------------------------------------------------------

    train_df, validation_df = (
        create_patient_level_split()
    )

    # --------------------------------------------------------
    # Create TensorFlow datasets
    # --------------------------------------------------------

    print()
    print("Creating TensorFlow datasets...")

    train_dataset = create_dataset(
        train_df,
        shuffle=True
    )

    validation_dataset = create_dataset(
        validation_df,
        shuffle=False
    )

    print("TensorFlow datasets created.")

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING CUSTOM CNN")
    print("=" * 70)

    model = build_cnn_model()

    compile_model(model)

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR /
        "custom_cnn_best.keras"
    )

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR /
        "custom_cnn_final.keras"
    )

    model.save(
        final_model_path
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best model:\n"
        f"{model_path}"
    )

    print(
        f"Final model:\n"
        f"{final_model_path}"
    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    save_history(history)

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    print()
    print("Final training metrics:")

    for key, values in history.history.items():

        print(
            f"{key}: "
            f"{values[-1]:.4f}"
        )

    print()
    print("=" * 70)
    print("CUSTOM CNN BASELINE FINISHED")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()