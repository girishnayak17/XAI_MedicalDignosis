
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

VALIDATION_FILE = RESULTS_DIR / "validation_predictions.npz"
TEST_FILE = RESULTS_DIR / "test_predictions.npz"

BASELINE_THRESHOLD = 0.5


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    data = np.load(path, allow_pickle=False)

    y_true = data["y_true"].astype(np.int32)
    y_score = data["y_score"].astype(np.float64)
    labels = data["labels"].astype(str)

    if y_true.shape != y_score.shape:
        raise ValueError(f"Shape mismatch in {path}")

    if y_true.shape[1] != len(labels):
        raise ValueError(f"Label count mismatch in {path}")

    return y_true, y_score, labels


def best_f1_threshold(y_true, y_score):
    # The threshold is selected using validation data only.
    precision, recall, thresholds = precision_recall_curve(
        y_true, y_score
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )

    best_index = int(np.argmax(f1))
    return float(thresholds[best_index])


def calculate_metrics(y_true, y_score, thresholds):
    y_pred = (y_score >= thresholds.reshape(1, -1)).astype(np.int32)

    rows = []

    for i, label in enumerate(LABELS):
        actual = y_true[:, i]
        scores = y_score[:, i]
        predicted = y_pred[:, i]

        rows.append({
            "Label": label,
            "Positives": int(actual.sum()),
            "Threshold": float(thresholds[i]),
            "ROC_AUC": (
                roc_auc_score(actual, scores)
                if len(np.unique(actual)) == 2
                else np.nan
            ),
            "AvgPrec": average_precision_score(actual, scores),
            "Precision": precision_score(
                actual, predicted, zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, zero_division=0
            ),
            "F1": f1_score(
                actual, predicted, zero_division=0
            ),
            "Predicted_Positive": int(predicted.sum()),
        })

    return pd.DataFrame(rows)


def main():
    global LABELS

    print("Loading validation predictions...")
    val_true, val_score, val_labels = load_predictions(
        VALIDATION_FILE
    )

    print("Loading held-out test predictions...")
    test_true, test_score, test_labels = load_predictions(
        TEST_FILE
    )

    if not np.array_equal(val_labels, test_labels):
        raise ValueError(
            "Validation and test label orders differ. Stop and investigate."
        )

    if val_score.shape[1] != test_score.shape[1]:
        raise ValueError("Validation/test output dimensions differ.")

    LABELS = val_labels.tolist()

    # Tune each label independently on validation data.
    thresholds = np.array([
        best_f1_threshold(val_true[:, i], val_score[:, i])
        for i in range(len(LABELS))
    ])

    # Baseline: fixed 0.5 threshold.
    baseline_thresholds = np.full(len(LABELS), BASELINE_THRESHOLD)

    val_baseline = calculate_metrics(
        val_true, val_score, baseline_thresholds
    )
    val_tuned = calculate_metrics(
        val_true, val_score, thresholds
    )

    # Apply validation-selected thresholds to test data.
    test_baseline = calculate_metrics(
        test_true, test_score, baseline_thresholds
    )
    test_tuned = calculate_metrics(
        test_true, test_score, thresholds
    )

    # Combined report.
    report = pd.DataFrame({
        "Label": LABELS,
        "Validation_Positive": val_tuned["Positives"],
        "Threshold": thresholds,
        "Val_F1_at_0.5": val_baseline["F1"],
        "Val_F1_tuned": val_tuned["F1"],
        "Test_Positive": test_tuned["Positives"],
        "Test_F1_at_0.5": test_baseline["F1"],
        "Test_F1_tuned": test_tuned["F1"],
        "Test_Precision_tuned": test_tuned["Precision"],
        "Test_Recall_tuned": test_tuned["Recall"],
        "Test_AvgPrec": test_tuned["AvgPrec"],
        "Test_ROC_AUC": test_tuned["ROC_AUC"],
        "Test_Predicted_Positive": test_tuned["Predicted_Positive"],
    })

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    print("\n=== VALIDATION-SELECTED THRESHOLDS ===")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    report_path = RESULTS_DIR / "threshold_comparison.csv"
    report.to_csv(report_path, index=False)

    np.savez_compressed(
        RESULTS_DIR / "tuned_thresholds.npz",
        labels=val_labels,
        thresholds=thresholds,
    )

    print(f"\nReport saved to: {report_path}")
    print(f"Thresholds saved to: {RESULTS_DIR / 'tuned_thresholds.npz'}")
    print("\nDone. The model was not retrained.")


if __name__ == "__main__":
    main()
