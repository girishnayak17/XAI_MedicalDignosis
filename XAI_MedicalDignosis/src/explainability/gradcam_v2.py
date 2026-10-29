
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import argparse

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "kaggle_chest_xray_covid4_best.keras"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "gradcam_v2"
    / "standalone"
)

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]

IMAGE_SIZE = (224, 224)
DEFAULT_LAYER = "activation_3"


# ============================================================
# GPU AND MODEL
# ============================================================

def load_model_and_grad_model(layer_name=DEFAULT_LAYER):
    """Load the existing model and expose its feature maps."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        raise RuntimeError(
            "TensorFlow cannot detect a GPU. "
            "Check the existing WSL GPU environment."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(
                gpu, True
            )
        except RuntimeError as exc:
            print("GPU memory-growth setting:", exc)

    print("TensorFlow:", tf.__version__)
    print("Available GPU(s):", gpus)
    print("Loading existing trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    try:
        feature_layer = model.get_layer(layer_name)
    except ValueError as exc:
        available = [
            layer.name for layer in model.layers
        ]
        raise ValueError(
            f"Layer '{layer_name}' was not found. "
            f"Available layers: {available}"
        ) from exc

    if len(feature_layer.output.shape) != 4:
        raise ValueError(
            f"Layer '{layer_name}' must produce a "
            "four-dimensional spatial feature map."
        )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            feature_layer.output,
            model.output,
        ],
        name="gradcam_v2_model",
    )

    print("Model loaded:", MODEL_PATH)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Grad-CAM layer:", layer_name)

    return grad_model


# ============================================================
# IMAGE LOADING
# ============================================================

def load_xray(image_path, image_size=IMAGE_SIZE):
    """Load an RGB X-ray without changing the model's input scale."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )

    with Image.open(image_path) as source:
        original = np.asarray(
            source.convert("RGB"),
            dtype=np.uint8,
        ).copy()

    resized = Image.fromarray(original).resize(
        image_size,
        Image.Resampling.BILINEAR,
    )

    input_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    # The model contains Rescaling(1/255).
    # Therefore, retain pixel values in the 0-255 range.
    input_tensor = input_array[None, ...]

    return original, input_tensor


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    input_tensor,
    grad_model,
    class_index=None,
):
    """
    Generate Grad-CAM for the selected class.

    If class_index is None, explain the predicted class.

    Returns:
        heatmap: normalized raw heatmap in [0, 1]
        class_index: explained class index
        class_scores: model output scores
    """

    image_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            image_tensor,
            training=False,
        )

        if class_index is None:
            class_index = int(
                tf.argmax(predictions[0]).numpy()
            )

        class_index = int(class_index)

        if not 0 <= class_index < len(CLASS_NAMES):
            raise ValueError(
                f"class_index must be between 0 and "
                f"{len(CLASS_NAMES) - 1}."
            )

        # Use the selected model output as the target score.
        target_score = predictions[:, class_index]

    gradients = tape.gradient(
        target_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. Check the selected layer "
            "and the model's computational graph."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(gradients)
        ).numpy()
    ):
        raise RuntimeError(
            "Non-finite gradients were detected."
        )

    # Average gradients spatially to calculate channel weights.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    # Weighted sum of the feature maps.
    cam = tf.reduce_sum(
        feature_maps[0]
        * channel_weights[0][None, None, :],
        axis=-1,
    )

    # Keep positive contributions.
    cam = tf.nn.relu(cam)

    heatmap = cam.numpy().astype(np.float32)

    # Resize to the model input dimensions.
    heatmap = cv2.resize(
        heatmap,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.maximum(heatmap, 0.0)

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Normalize safely.
    maximum = float(heatmap.max())

    if maximum > 1e-8:
        heatmap /= maximum
    else:
        print(
            "WARNING: Grad-CAM produced an all-zero heatmap."
        )

    class_scores = predictions[0].numpy()

    return heatmap, class_index, class_scores


# ============================================================
# DISPLAY-ONLY CONTRAST ENHANCEMENT
# ============================================================

def enhance_heatmap(heatmap):
    """Enhance heatmap contrast for display only."""

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    heatmap = np.clip(heatmap, 0.0, 1.0)

    positive = heatmap[heatmap > 1e-8]

    if positive.size == 0:
        print(
            "WARNING: No positive Grad-CAM activations."
        )
        return np.zeros_like(heatmap)

    upper = float(np.percentile(positive, 99))

    if upper <= 1e-8:
        upper = float(positive.max())

    heatmap = np.clip(
        heatmap / max(upper, 1e-8),
        0.0,
        1.0,
    )

    # Make weaker activations more visible.
    heatmap = np.power(heatmap, 0.75)

    # Smooth the visualization slightly.
    heatmap = cv2.GaussianBlur(
        heatmap,
        (0, 0),
        sigmaX=1.0,
    )

    return np.clip(
        heatmap,
        0.0,
        1.0,
    ).astype(np.float32)


# ============================================================
# COLOR HEATMAP AND OVERLAY
# ============================================================

def create_overlay(
    original_image,
    heatmap,
    alpha=0.55,
    colormap=cv2.COLORMAP_JET,
):
    """
    Create a classic colored heatmap and blend it with
    the original X-ray.

    Returns:
        overlay_rgb, colored_heatmap_rgb
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    original = np.asarray(
        original_image,
        dtype=np.uint8,
    )

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError(
            "original_image must be an RGB image."
        )

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            "heatmap must be a two-dimensional array."
        )

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    heatmap_uint8 = np.round(
        heatmap * 255.0
    ).astype(np.uint8)

    # Classic blue-green-yellow-red color map.
    colored_bgr = cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )

    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Strong activations are more opaque than weak activations.
    strength = np.clip(
        (heatmap - 0.02) / 0.98,
        0.0,
        1.0,
    )

    alpha_map = (
        alpha * np.power(strength, 0.65)
    )[..., None]

    overlay = (
        original.astype(np.float32)
        * (1.0 - alpha_map)
        + colored_rgb.astype(np.float32)
        * alpha_map
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return overlay, colored_rgb


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    image_path,
    original,
    raw_heatmap,
    class_index,
    class_scores,
    alpha=0.55,
):
    """Save the raw heatmap and all display images."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    enhanced_heatmap = enhance_heatmap(
        raw_heatmap
    )

    overlay, colored_heatmap = create_overlay(
        original,
        enhanced_heatmap,
        alpha=alpha,
    )

    stem = Path(image_path).stem

    original_path = (
        OUTPUT_DIR / f"{stem}_original.png"
    )

    raw_path = (
        OUTPUT_DIR / f"{stem}_raw_gradcam.npy"
    )

    heatmap_path = (
        OUTPUT_DIR / f"{stem}_heatmap.png"
    )

    overlay_path = (
        OUTPUT_DIR / f"{stem}_overlay.png"
    )

    comparison_path = (
        OUTPUT_DIR / f"{stem}_comparison.png"
    )

    Image.fromarray(original).save(
        original_path
    )

    # Save the raw normalized Grad-CAM separately.
    np.save(
        raw_path,
        raw_heatmap,
    )

    Image.fromarray(colored_heatmap).save(
        heatmap_path
    )

    Image.fromarray(overlay).save(
        overlay_path
    )

    # Three-panel comparison.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original Chest X-ray")

    axes[1].imshow(colored_heatmap)
    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")

    confidence = float(
        class_scores[class_index]
    ) * 100.0

    fig.suptitle(
        f"Predicted class: {CLASS_NAMES[class_index]} "
        f"| Model score: {confidence:.1f}%",
        fontsize=13,
    )

    for axis in axes:
        axis.axis("off")

    fig.tight_layout(
        rect=(0, 0, 1, 0.92)
    )

    fig.savefig(
        comparison_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved results:")
    print("Original:  ", original_path)
    print("Heatmap:   ", heatmap_path)
    print("Overlay:   ", overlay_path)
    print("Comparison:", comparison_path)
    print("Raw values:", raw_path)
    print("Output directory:", OUTPUT_DIR)

    return overlay


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a standalone Grad-CAM v2 "
            "visualization for a chest X-ray."
        )
    )

    parser.add_argument(
        "image",
        help="Path to the chest X-ray image.",
    )

    parser.add_argument(
        "--layer",
        default=DEFAULT_LAYER,
        help=(
            "Spatial feature layer to explain "
            "(default: activation_3)."
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help=(
            "Maximum heatmap overlay opacity, from 0 to 1."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1.")

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )

    print("Project directory:", BASE_DIR)
    print("Model path:", MODEL_PATH)
    print("Input image:", image_path)

    # Load the trained model and configure GPU.
    grad_model = load_model_and_grad_model(
        args.layer
    )

    original, input_tensor = load_xray(
        image_path
    )

    print("\nGenerating Grad-CAM...")

    heatmap, class_index, class_scores = (
        generate_gradcam(
            input_tensor,
            grad_model,
            class_index=None,
        )
    )

    print("\nClass scores:")

    for index, name in enumerate(CLASS_NAMES):
        print(
            f"{name:15s}: "
            f"{float(class_scores[index]) * 100:.2f}%"
        )

    print(
        f"\nPredicted class: {CLASS_NAMES[class_index]}"
    )

    print(
        "Predicted-class score: "
        f"{float(class_scores[class_index]) * 100:.2f}%"
    )

    print(
        "Heatmap shape:", heatmap.shape
    )

    print(
        "Heatmap min/max:",
        float(heatmap.min()),
        float(heatmap.max()),
    )

    print(
        "Heatmap standard deviation:",
        float(heatmap.std()),
    )

    save_results(
        image_path=image_path,
        original=original,
        raw_heatmap=heatmap,
        class_index=class_index,
        class_scores=class_scores,
        alpha=args.alpha,
    )

    print("\nGrad-CAM v2 completed.")


if __name__ == "__main__":
    main()
