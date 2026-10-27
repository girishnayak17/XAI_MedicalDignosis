
from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "kaggle_chest_xray_best.keras"

TEST_DIR = BASE_DIR.parent / "KDatasets" / "test"

RESULTS_DIR = BASE_DIR / "results"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

EXPECTED_CLASSES = ["normal", "pneumonia", "tuberculosis"]


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("=" * 65)
    print("LOADING TRAINED MODEL")
    print("=" * 65)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(f"Model: {MODEL_PATH}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    if model.output_shape[-1] != len(EXPECTED_CLASSES):
        raise ValueError(
            f"Expected {len(EXPECTED_CLASSES)} output classes, "
            f"but model has {model.output_shape[-1]} outputs."
        )

    return model


# ============================================================
# 3. LOAD COMPLETE TEST DATASET IN BATCHES
# ============================================================

def load_test_dataset():
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_DIR}"
        )

    print("\n" + "=" * 65)
    print("LOADING TEST DATASET")
    print("=" * 65)

    dataset = tf.keras.utils.image_dataset_from_directory(
        str(TEST_DIR),
        labels="inferred",
        label_mode="int",
        class_names=EXPECTED_CLASSES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # Save class names before applying prefetch.
    class_names = dataset.class_names

    print(f"Test directory: {TEST_DIR}")
    print(f"Class order: {class_names}")
    print(f"Batch size: {BATCH_SIZE}")

    # Prefetch batches to improve data-loading efficiency.
    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# 4. EVALUATE EVERY TEST IMAGE
# ============================================================

def evaluate_model(model, dataset):
    print("\n" + "=" * 65)
    print("RUNNING BATCH-BY-BATCH PREDICTIONS")
    print("=" * 65)

    all_true_labels = []
    all_predicted_labels = []
    all_scores = []

    total_images = sum(
        len([
            file for file in (TEST_DIR / class_name).rglob("*")
            if file.is_file()
            and file.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ])
        for class_name in EXPECTED_CLASSES
    )

    processed_images = 0

    for batch_number, (images, labels) in enumerate(
        dataset,
        start=1,
    ):
        # Predict the whole batch in one model call.
        scores = model.predict(
            images,
            verbose=0,
        )

        predicted_labels = np.argmax(scores, axis=1)

        true_labels = labels.numpy()

        all_true_labels.extend(true_labels.tolist())
        all_predicted_labels.extend(predicted_labels.tolist())
        all_scores.extend(scores.tolist())

        processed_images += len(true_labels)

        print(
            f"Batch {batch_number:4d} | "
            f"Processed {processed_images:5d}"
            f"/{total_images} images"
        )

    y_true = np.asarray(all_true_labels)
    y_pred = np.asarray(all_predicted_labels)
    scores = np.asarray(all_scores)

    return y_true, y_pred, scores


# ============================================================
# 5. DISPLAY METRICS
# ============================================================

def display_results(y_true, y_pred):
    class_ids = list(range(len(EXPECTED_CLASSES)))

    accuracy = accuracy_score(y_true, y_pred)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=class_ids,
    )

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(f"Total images tested: {len(y_true)}")
    print(f"Correct predictions: {np.sum(y_true == y_pred)}")
    print(f"Incorrect predictions: {np.sum(y_true != y_pred)}")
    print(f"Overall accuracy: {accuracy * 100:.2f}%")

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=EXPECTED_CLASSES,
        digits=4,
        zero_division=0,
    )

    print(report)

    print("CONFUSION MATRIX")
    print("-" * 65)
    print("Rows = actual class")
    print("Columns = predicted class")
    print()
    print("Class order:", EXPECTED_CLASSES)
    print(matrix)

    print("\nPER-CLASS ACCURACY")
    print("-" * 65)

    for index, class_name in enumerate(EXPECTED_CLASSES):
        class_total = np.sum(y_true == index)
        class_correct = matrix[index, index]

        class_accuracy = (
            class_correct / class_total
            if class_total > 0
            else 0.0
        )

        print(
            f"{class_name:15s}: "
            f"{class_accuracy * 100:.2f}% "
            f"({class_correct}/{class_total})"
        )

    return accuracy, matrix, report


# ============================================================
# 6. DISPLAY MISCLASSIFIED EXAMPLES
# ============================================================

def show_misclassified(y_true, y_pred, scores, limit=30):
    wrong_indices = np.where(y_true != y_pred)[0]

    print("\n" + "=" * 65)
    print(f"MISCLASSIFIED IMAGES (showing up to {limit})")
    print("=" * 65)

    if len(wrong_indices) == 0:
        print("No misclassified images found.")
        return

    for index in wrong_indices[:limit]:
        actual = EXPECTED_CLASSES[int(y_true[index])]
        predicted = EXPECTED_CLASSES[int(y_pred[index])]

        confidence = float(scores[index, y_pred[index]]) * 100

        print(
            f"Image: {index + 1:4d} | "
            f"Actual: {actual:12s} | "
            f"Predicted: {predicted:12s} | "
            f"Top score: {confidence:.2f}%"
        )

    print(f"\nTotal misclassified images: {len(wrong_indices)}")


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_results(y_true, y_pred, scores, accuracy, matrix, report):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        RESULTS_DIR / "kaggle_batch_test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        class_names=np.asarray(EXPECTED_CLASSES),
    )

    np.savetxt(
        RESULTS_DIR / "kaggle_batch_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(EXPECTED_CLASSES),
        comments="",
    )

    report_path = RESULTS_DIR / "kaggle_batch_test_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("KAGGLE CHEST X-RAY TEST RESULTS\n")
        file.write("=" * 50 + "\n")
        file.write(f"Model: {MODEL_PATH}\n")
        file.write(f"Test directory: {TEST_DIR}\n")
        file.write(f"Total images: {len(y_true)}\n")
        file.write(f"Overall accuracy: {accuracy * 100:.2f}%\n\n")
        file.write("Class order: " + str(EXPECTED_CLASSES) + "\n\n")
        file.write("Classification report:\n")
        file.write(report + "\n")
        file.write("Confusion matrix:\n")
        file.write(str(matrix) + "\n")

    print("\n" + "=" * 65)
    print("RESULTS SAVED")
    print("=" * 65)
    print(RESULTS_DIR / "kaggle_batch_test_predictions.npz")
    print(RESULTS_DIR / "kaggle_batch_confusion_matrix.csv")
    print(report_path)


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():
    print("\nCHEST X-RAY BATCH EVALUATION")
    print("Classes: Normal, Pneumonia, Tuberculosis")
    print(f"Batch size: {BATCH_SIZE}")

    model = load_model()
    dataset = load_test_dataset()

    y_true, y_pred, scores = evaluate_model(
        model,
        dataset,
    )

    accuracy, matrix, report = display_results(
        y_true,
        y_pred,
    )

    show_misclassified(
        y_true,
        y_pred,
        scores,
        limit=30,
    )

    save_results(
        y_true,
        y_pred,
        scores,
        accuracy,
        matrix,
        report,
    )

    print("\nEvaluation complete, bro!")


if __name__ == "__main__":
    main()
