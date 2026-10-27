"""
Run Grad-CAM on the trained Custom CNN.

Usage:

python src/explainability/run_gradcam.py "path/to/image.png"
"""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image

# Import our Grad-CAM functions
from gradcam import (
    load_image,
    generate_gradcam,
    create_gradcam_overlay,
    get_predictions,
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "custom_cnn_best.keras"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
    / "gradcam"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GRAD-CAM - CUSTOM CNN")
    print("=" * 70)

    # --------------------------------------------------------
    # Check image argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print("\nUsage:")
        print(
            'python src/explainability/run_gradcam.py '
            '"path/to/xray.png"'
        )

        return

    image_path = Path(sys.argv[1])

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    print(f"\nModel:")
    print(MODEL_PATH)

    print(f"\nInput image:")
    print(image_path)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print("\nLoading image...")

    original_image, input_tensor = load_image(
        image_path
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print("\nGenerating predictions...")

    predictions = get_predictions(
        model,
        input_tensor,
    )

    print("\nPredictions:")
    print("-" * 70)

    for result in predictions:

        print(
            f"{result['label']:20s} "
            f"{result['probability']:.4f}"
        )

    # --------------------------------------------------------
    # Select highest prediction
    # --------------------------------------------------------

    top_prediction = predictions[0]

    class_index = top_prediction["index"]

    class_label = top_prediction["label"]

    probability = top_prediction["probability"]

    print("\n" + "=" * 70)

    print(
        f"Explaining: {class_label}"
    )

    print(
        f"Probability: {probability:.4f}"
    )

    print(
        f"Class index: {class_index}"
    )

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    print("\nGenerating Grad-CAM...")

    heatmap, prediction, last_conv_layer = generate_gradcam(
        model=model,
        input_tensor=input_tensor,
        class_index=class_index,
    )

    print(
        f"Last convolutional layer: "
        f"{last_conv_layer.name}"
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_gradcam_overlay(
        original_image,
        heatmap,
        alpha=0.4,
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_stem = image_path.stem

    original_output = (
        RESULTS_DIR
        / f"{image_stem}_original.png"
    )

    heatmap_output = (
        RESULTS_DIR
        / f"{image_stem}_heatmap.png"
    )

    overlay_output = (
        RESULTS_DIR
        / f"{image_stem}_gradcam.png"
    )

    # Save original
    Image.fromarray(
        original_image
    ).save(original_output)

    # Save heatmap
    heatmap_image = np.uint8(
        heatmap * 255
    )

    Image.fromarray(
        heatmap_image
    ).save(heatmap_output)

    # Save overlay
    Image.fromarray(
        overlay
    ).save(overlay_output)

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GRAD-CAM COMPLETED")
    print("=" * 70)

    print(
        f"\nOriginal image:\n{original_output}"
    )

    print(
        f"\nHeatmap:\n{heatmap_output}"
    )

    print(
        f"\nGrad-CAM overlay:\n{overlay_output}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()