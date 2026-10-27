
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Datasets"
MODEL_PATH = BASE_DIR / "models" / "custom_cnn_best.keras"
CSV_PATH = DATA_DIR / "Data_Entry_2017_v2020.csv"
TEST_LIST_PATH = DATA_DIR / "test_list.txt"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
THRESHOLD = 0.5

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
]


def main():
    print("Loading NIH test metadata...")

    metadata = pd.read_csv(CSV_PATH)
    test_names = [
        name.strip()
        for name in TEST_LIST_PATH.read_text().splitlines()
        if name.strip()
    ]

    metadata = metadata.set_index("Image Index")

    missing = [
        name for name in test_names
        if name not in metadata.index
    ]
    if missing:
        raise ValueError(
            f"{len(missing)} test images are missing from metadata."
        )

    # Build ground-truth multi-label vectors in the model's label order.
    y_true = np.zeros(
        (len(test_names), len(LABELS)),
        dtype=np.float32,
    )

    for i, filename in enumerate(test_names):
        raw_labels = str(
            metadata.loc[filename, "Finding Labels"]
        )
        image_labels = set(raw_labels.split("|"))

        for j, label in enumerate(LABELS):
            y_true[i, j] = float(label in image_labels)

    print("Test images:", len(test_names))
    print("Loading trained model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(LABELS):
        raise ValueError(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but {len(LABELS)} labels were supplied."
        )

    # Predict in small batches to limit RAM usage.
    all_scores = []

    for start in range(0, len(test_names), BATCH_SIZE):
        batch_names = test_names[start:start + BATCH_SIZE]
        batch_images = []

        for filename in batch_names:
            image_path = DATA_DIR / "images" / filename

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(IMAGE_SIZE)
                array = np.asarray(
                    image, dtype=np.float32
                ) / 255.0

            batch_images.append(array)

        batch_array = np.stack(batch_images)
        batch_scores = model(
            batch_array, training=False
        ).numpy()

        all_scores.append(batch_scores)

        completed = min(start + len(batch_names), len(test_names))
        if completed % 512 < BATCH_SIZE or completed == len(test_names):
            print(f"Evaluated {completed}/{len(test_names)} images")

    y_score = np.concatenate(all_scores, axis=0)
    y_pred = (y_score >= THRESHOLD).astype(np.int32)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(f"Threshold: {THRESHOLD}")
    print(
        f"{'Label':22s} {'Positives':>10s} "
        f"{'ROC-AUC':>10s} {'AvgPrec':>10s} "
        f"{'Precision':>10s} {'Recall':>10s} {'F1':>10s}"
    )

    for j, label in enumerate(LABELS):
        truth = y_true[:, j].astype(np.int32)
        predicted = y_pred[:, j]
        positives = int(truth.sum())

        if len(np.unique(truth)) == 2:
            auc = roc_auc_score(truth, y_score[:, j])
            avg_precision = average_precision_score(
                truth, y_score[:, j]
            )
            auc_text = f"{auc:.4f}"
            ap_text = f"{avg_precision:.4f}"
        else:
            auc_text = "N/A"
            ap_text = "N/A"

        precision = precision_score(
            truth, predicted, zero_division=0
        )
        recall = recall_score(
            truth, predicted, zero_division=0
        )
        f1 = f1_score(
            truth, predicted, zero_division=0
        )

        print(
            f"{label:22s} {positives:10d} "
            f"{auc_text:>10s} {ap_text:>10s} "
            f"{precision:10.4f} {recall:10.4f} {f1:10.4f}"
        )

    output_path = BASE_DIR / "results" / "test_predictions.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        filenames=np.asarray(test_names),
        y_true=y_true,
        y_score=y_score,
        labels=np.asarray(LABELS),
    )

    print(f"\nSaved detailed predictions to: {output_path}")
    print("Evaluation completed. The model was not retrained.")


if __name__ == "__main__":
    main()
