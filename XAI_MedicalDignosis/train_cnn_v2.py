
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from train import (
    create_patient_level_split,
    create_dataset,
    build_cnn_model,
    BATCH_SIZE,
    EPOCHS,
    RESULTS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    DISEASE_LABELS,
)

from weighted_loss import WeightedBinaryCrossentropy


MAX_POSITIVE_WEIGHT = 8.0


def main():
    print("=" * 65)
    print("CUSTOM CNN V2 — IMBALANCE-AWARE TRAINING")
    print("=" * 65)

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    # Configure GPU memory growth before model operations.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("GPU configuration warning:", error)

    train_df, validation_df = create_patient_level_split()

    # Calculate weights from training labels ONLY.
    y_train = np.stack(
        train_df["Encoded Labels"].values
    ).astype(np.float32)

    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives

    raw_ratios = negatives / np.maximum(positives, 1.0)

    positive_weights = np.clip(
        np.sqrt(raw_ratios),
        1.0,
        MAX_POSITIVE_WEIGHT,
    )

    print("\nPositive-class weights:")
    for label, count, weight in zip(
        DISEASE_LABELS, positives, positive_weights
    ):
        print(f"{label:22s} positives={int(count):6d} weight={weight:.3f}")

    train_dataset = create_dataset(
        train_df,
        shuffle=True,
    )

    validation_dataset = create_dataset(
        validation_df,
        shuffle=False,
    )

    model = build_cnn_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss=WeightedBinaryCrossentropy(
            positive_weights=positive_weights.tolist()
        ),
        metrics=[
            tf.keras.metrics.AUC(
                name="auc",
                multi_label=True,
                num_labels=len(DISEASE_LABELS),
            ),
        ],
    )

    model.summary()

    best_path = MODEL_DIR / "custom_cnn_v2_best.keras"
    final_path = MODEL_DIR / "custom_cnn_v2_final.keras"
    history_path = RESULTS_DIR / "cnn_v2_training_history.json"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
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
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(final_path)

    with open(history_path, "w") as file:
        json.dump(
            {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            file,
            indent=4,
        )

    print("\nV2 training completed.")
    print(f"Best checkpoint: {best_path}")
    print(f"Final model:     {final_path}")
    print(f"History:         {history_path}")
    print("Original CNN checkpoint has not been overwritten.")


if __name__ == "__main__":
    main()
