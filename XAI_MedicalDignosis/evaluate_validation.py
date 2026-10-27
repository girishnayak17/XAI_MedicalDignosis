
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import (
    create_patient_level_split,
    create_dataset,
    BATCH_SIZE,
    DISEASE_LABELS,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "custom_cnn_best.keras"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "validation_predictions.npz"


def main():
    print("=" * 65)
    print("VALIDATION PREDICTION GENERATION")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    # Recreate the exact split from train.py.
    # Keep the metadata, list file, and random seed unchanged.
    train_df, validation_df = create_patient_level_split()

    print(f"\nValidation images: {len(validation_df)}")

    # Confirm patient separation.
    train_patients = set(train_df["Patient ID"])
    validation_patients = set(validation_df["Patient ID"])

    overlap = train_patients & validation_patients

    if overlap:
        raise RuntimeError(
            f"Patient leakage detected: {len(overlap)} overlapping patients"
        )

    print("Patient-level separation verified.")

    # Ground-truth multi-label targets in the model's label order.
    y_true = np.stack(
        validation_df["Encoded Labels"].values
    ).astype(np.float32)

    filenames = validation_df["Image Index"].astype(str).to_numpy()

    # Preserve validation order: do not shuffle.
    validation_dataset = create_dataset(
        validation_df,
        shuffle=False,
    )

    print("\nLoading best checkpoint...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(DISEASE_LABELS):
        raise ValueError(
            "Model output count does not match DISEASE_LABELS."
        )

    print("Generating predictions...")
    y_score = model.predict(
        validation_dataset,
        verbose=1,
    )

    if len(y_score) != len(y_true):
        raise RuntimeError(
            "Prediction count does not match validation labels."
        )

    np.savez_compressed(
        OUTPUT_PATH,
        filenames=filenames,
        y_true=y_true,
        y_score=y_score,
        labels=np.asarray(DISEASE_LABELS),
    )

    print("\nValidation predictions saved:")
    print(OUTPUT_PATH)
    print(f"Ground-truth shape: {y_true.shape}")
    print(f"Prediction shape:   {y_score.shape}")

    print("\nGeneration completed successfully.")


if __name__ == "__main__":
    main()
