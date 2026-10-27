
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model_path,
    test_dir,
    class_names=None,
    image_size=224,
    batch_size=8,
    output_dir=None,
):
    """
    Evaluate a saved Keras single-label classification model.

    Parameters
    ----------
    model_path : str
        Path to the saved .keras model.
    test_dir : str
        Test directory containing one subfolder per class.
    class_names : list
        Class names in the exact order used during model training.
    image_size : int
        Image width and height.
    batch_size : int
        Number of images processed per batch.
    output_dir : str or None
        Directory in which evaluation results are saved.

    Returns
    -------
    dict
        Evaluation metrics and confusion matrix.
    """

    model_path = Path(model_path).resolve()
    test_dir = Path(test_dir).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    if not test_dir.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}"
        )

    # Create a new output folder for each evaluation.
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "results"
            / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"
        )
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading trained model...")

    # compile=False loads the model for inference only.
    # This does not retrain or overwrite the model.
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)

    if isinstance(model.output_shape, list):
        raise ValueError(
            "This evaluator expects a model with one output tensor."
        )

    if len(model.output_shape) != 2:
        raise ValueError(
            "Expected model output shape (batch_size, number_of_classes)."
        )

    number_of_classes = model.output_shape[-1]

    if number_of_classes is None:
        raise ValueError("Unable to determine the number of classes.")

    print("\nLoading test dataset...")

    # shuffle=False keeps predictions aligned with the image paths.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=False,
    )

    actual_class_names = list(test_ds.class_names)

    if len(actual_class_names) != number_of_classes:
        raise ValueError(
            f"The test dataset has {len(actual_class_names)} classes, "
            f"but the model has {number_of_classes} outputs."
        )

    print("Class/output order:", actual_class_names)
    print("Number of test images:", len(test_ds.file_paths))

    # Collect true labels in the same order as the dataset.
    y_true = np.concatenate(
        [labels.numpy() for _, labels in test_ds],
        axis=0,
    )

    # Predict class scores.
    scores = model.predict(test_ds, verbose=1)

    if scores.ndim != 2:
        raise ValueError(
            f"Expected 2D predictions; received shape {scores.shape}."
        )

    if scores.shape[1] != len(actual_class_names):
        raise ValueError("Prediction count does not match class count.")

    # For softmax multiclass classification, choose the highest score.
    y_pred = np.argmax(scores, axis=1)

    # Calculate metrics.
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall_macro = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_weighted = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1_macro = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_weighted = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    labels = list(range(len(actual_class_names)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=actual_class_names,
        digits=4,
        zero_division=0,
    )

    # Print results.
    print("\n" + "=" * 55)
    print("MODEL EVALUATION RESULTS")
    print("=" * 55)

    print(f"Accuracy:              {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision (macro):     {precision_macro:.4f}")
    print(f"Precision (weighted):  {precision_weighted:.4f}")
    print(f"Recall (macro):        {recall_macro:.4f}")
    print(f"Recall (weighted):     {recall_weighted:.4f}")
    print(f"F1-score (macro):      {f1_macro:.4f}")
    print(f"F1-score (weighted):   {f1_weighted:.4f}")

    print("\nConfusion Matrix")
    print("Rows = actual class; columns = predicted class")
    print(cm)

    print("\nClassification Report")
    print(report)

    # Save text report.
    report_path = output_dir / "evaluation_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {model_path}\n")
        file.write(f"Test dataset: {test_dir}\n")
        file.write(f"Class order: {actual_class_names}\n")
        file.write(f"Test images: {len(y_true)}\n\n")

        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Precision (macro): {precision_macro:.6f}\n")
        file.write(f"Precision (weighted): {precision_weighted:.6f}\n")
        file.write(f"Recall (macro): {recall_macro:.6f}\n")
        file.write(f"Recall (weighted): {recall_weighted:.6f}\n")
        file.write(f"F1-score (macro): {f1_macro:.6f}\n")
        file.write(f"F1-score (weighted): {f1_weighted:.6f}\n\n")

        file.write("Confusion Matrix:\n")
        file.write(np.array2string(cm))
        file.write("\n\nClassification Report:\n")
        file.write(report)

    # Save confusion matrix image.
    fig, ax = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=actual_class_names,
    )

    display.plot(
        ax=ax,
        values_format="d",
        xticks_rotation=35,
        colorbar=False,
    )

    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    confusion_path = output_dir / "confusion_matrix.png"
    fig.savefig(confusion_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Save predictions for every test image.
    prediction_data = {
        "image_path": list(test_ds.file_paths),
        "actual_class": [
            actual_class_names[i] for i in y_true
        ],
        "predicted_class": [
            actual_class_names[i] for i in y_pred
        ],
        "correct": (y_true == y_pred),
    }

    for i, class_name in enumerate(actual_class_names):
        safe_name = "".join(
            c if c.isalnum() else "_"
            for c in class_name.lower()
        )
        prediction_data[f"score_{safe_name}"] = scores[:, i]

    predictions_path = output_dir / "test_predictions.csv"

    pd.DataFrame(prediction_data).to_csv(
        predictions_path,
        index=False,
    )

    print("\nSaved evaluation files:")
    print("Report:", report_path)
    print("Confusion matrix:", confusion_path)
    print("Predictions:", predictions_path)

    print("\nEvaluation completed. The model was not modified.")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "class_names": actual_class_names,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Keras image classification model."
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--test-dir", required=True)

    parser.add_argument(
        "--class-names",
        nargs="+",
        default=None,
        help="Class names in the exact training/output order.",
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default=None)

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        test_dir=args.test_dir,
        class_names=args.class_names,
        image_size=args.image_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )



#runing cmd 
#python evaluate_model.py \
#  --model /mnt/d/MCA_Final_Project/XAI_MedicalDignosis/models/kaggle_chest_xray_clean_best.keras \
#  --test-dir /mnt/d/MCA_Final_Project/KDatasets_Clean/test \
#  --image-size 224 \
#  --batch-size 8 \
#  --class-names normal pneumonia tuberculosis