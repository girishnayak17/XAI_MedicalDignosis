
import time
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.transfer_learning.models import (
    load_resnet50_checkpoint
)


CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19"
]


# ============================================================
# EXTRACT TRUE LABELS
# ============================================================

def get_true_labels(dataset):

    return np.concatenate([
        labels.numpy()
        for _, labels in dataset
    ])


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model_path,
    dataset,
    dataset_name="Validation"
):

    model = load_resnet50_checkpoint(
        model_path
    )

    y_true = get_true_labels(dataset)

    start_time = time.perf_counter()

    probabilities = model.predict(
        dataset,
        verbose=1
    )

    inference_time = (
        time.perf_counter() - start_time
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro"
        )

    except ValueError:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES)))
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    metrics = {

        "Accuracy": float(accuracy),

        "Precision": float(precision),

        "Recall": float(recall),

        "Macro F1": float(macro_f1),

        "ROC-AUC": float(roc_auc),

        "Inference Time (seconds)": float(
            inference_time
        )

    }

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print(f"{dataset_name.upper()} RESULTS")

    print("=" * 60)

    for name, value in metrics.items():

        print(
            f"{name:30s}: {value:.4f}"
        )

    print("\nConfusion Matrix:\n")
    print(cm)

    print("\nClassification Report:\n")
    print(report)

    return {

        "metrics": metrics,

        "confusion_matrix": cm,

        "classification_report": report

    }
