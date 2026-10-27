"""
Grad-CAM implementation for the four-class Kaggle Chest X-ray CNN.

Classes:
    0. Normal
    1. Pneumonia
    2. Tuberculosis
    3. COVID-19

The model contains a Rescaling(1./255) layer internally.
Input pixel values must remain in [0, 255].
Do not divide pixels by 255 in this module.
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    """
    Load and prepare an X-ray image.

    Returns:
        original_image: Original RGB image as uint8, shape (H, W, 3).
        input_tensor: Float32 tensor, shape (1, 224, 224, 3),
                      with pixel values in [0, 255].
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            original_image = np.asarray(
                image,
                dtype=np.uint8,
            ).copy()

            resized_image = image.resize(
                IMAGE_SIZE,
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                resized_image,
                dtype=np.float32,
            )

    except Exception as exc:
        raise ValueError(
            f"Could not load image: {image_path}"
        ) from exc

    input_tensor = np.expand_dims(image_array, axis=0)

    return original_image, input_tensor


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):
    """
    Find the last Conv2D layer, including layers inside nested models.

    The four-class model may contain a nested feature extractor.
    """

    def search_layers(layers):
        for layer in reversed(layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer

            if isinstance(layer, tf.keras.Model):
                result = search_layers(layer.layers)

                if result is not None:
                    return result

        return None

    last_conv_layer = search_layers(model.layers)

    if last_conv_layer is None:
        raise ValueError(
            "Could not find a Conv2D layer in the model. "
            "Inspect the model architecture before generating Grad-CAM."
        )

    return last_conv_layer


# ============================================================
# FIND CONVOLUTIONAL ACTIVATIONS
# ============================================================

def build_gradcam_model(model, last_conv_layer):
    """
    Build a model returning:
        1. Activations from the selected convolutional layer.
        2. The final class predictions.

    Handles a convolutional layer nested inside a functional
    submodel when that layer's output cannot be accessed directly
    through the outer model graph.
    """

    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                last_conv_layer.output,
                model.outputs[0],
            ],
        )

    except (ValueError, AttributeError):
        # A nested model may expose its own convolutional
        # activations and predictions, but the outer model's
        # graph may not allow direct use of that internal tensor.
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    nested_conv = None

                    def search(layers):
                        nonlocal nested_conv

                        for item in reversed(layers):
                            if isinstance(
                                item,
                                tf.keras.layers.Conv2D,
                            ):
                                nested_conv = item
                                return

                            if isinstance(
                                item,
                                tf.keras.Model,
                            ):
                                search(item.layers)

                            if nested_conv is not None:
                                return

                    search(layer.layers)

                    if nested_conv is None:
                        continue

                    nested_grad_model = tf.keras.Model(
                        inputs=layer.inputs,
                        outputs=[
                            nested_conv.output,
                            layer.outputs[0],
                        ],
                    )

                    # Only use this fallback when the nested model
                    # itself produces the final class predictions.
                    if layer.output_shape[-1] == NUM_CLASSES:
                        return nested_grad_model

                except (ValueError, AttributeError, IndexError):
                    continue

        raise ValueError(
            "Could not connect the selected convolutional layer "
            "to the model output. The model may contain nested layers. "
            "Inspect model.summary() and build Grad-CAM for the "
            "correct feature-extractor graph."
        )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(model, input_tensor, class_index):
    """
    Generate a Grad-CAM heatmap for a selected class.

    Args:
        model: Trained four-class CNN.
        input_tensor: Image batch of shape (1, 224, 224, 3),
                      with pixel values in [0, 255].
        class_index: 0=Normal, 1=Pneumonia,
                     2=Tuberculosis, 3=COVID-19.

    Returns:
        heatmap: Normalized 2D array in [0, 1].
        prediction: Selected class score.
        last_conv_layer_name: Name of the selected convolutional layer.
    """

    if not isinstance(class_index, (int, np.integer)):
        raise TypeError("class_index must be an integer.")

    if not 0 <= int(class_index) < NUM_CLASSES:
        raise ValueError(
            f"class_index must be between 0 and {NUM_CLASSES - 1}."
        )

    input_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32,
    )

    if len(input_tensor.shape) != 4:
        raise ValueError(
            "Expected input shape (1, 224, 224, 3)."
        )

    if input_tensor.shape[-1] != 3:
        raise ValueError(
            "Expected a three-channel RGB image."
        )

    last_conv_layer = find_last_conv_layer(model)

    # Build a graph exposing the convolutional feature maps.
    grad_model = build_gradcam_model(
        model,
        last_conv_layer,
    )

    # Record gradients with respect to the convolutional features.
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False,
        )

        if predictions.shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"Grad-CAM graph returned {predictions.shape[-1]} "
                f"outputs; expected {NUM_CLASSES}."
            )

        class_score = predictions[:, int(class_index)]

    gradients = tape.gradient(
        class_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. The selected convolutional layer "
            "is not connected to the requested class score. "
            "Inspect the model graph and choose a suitable layer."
        )

    # Standard Grad-CAM: average gradients across spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    feature_maps = conv_outputs[0]
    channel_weights = channel_weights[0]

    heatmap = tf.reduce_sum(
        feature_maps * channel_weights,
        axis=-1,
    )

    # Keep positive class-supporting activations.
    heatmap = tf.maximum(heatmap, 0.0)

    maximum = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(
        heatmap,
        maximum,
    )

    heatmap = heatmap.numpy().astype(np.float32)

    selected_score = float(
        predictions[0, int(class_index)].numpy()
    )

    return (
        heatmap,
        selected_score,
        last_conv_layer.name,
    )


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap,
    alpha=0.4,
):
    """
    Blend a Grad-CAM heatmap with an original RGB X-ray.

    Returns:
        RGB uint8 image suitable for st.image().
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    original = np.asarray(original_image)

    # Convert grayscale images to RGB.
    if original.ndim == 2:
        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB,
        )

    elif original.ndim == 3 and original.shape[-1] == 1:
        original = cv2.cvtColor(
            original[:, :, 0],
            cv2.COLOR_GRAY2RGB,
        )

    if original.ndim != 3 or original.shape[-1] != 3:
        raise ValueError(
            f"Expected an RGB image; got shape {original.shape}."
        )

    # Ensure the original image is uint8 in [0, 255].
    if original.dtype != np.uint8:
        original = np.nan_to_num(
            original.astype(np.float32),
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        if original.size and original.max() <= 1.0:
            original = original * 255.0

        original = np.clip(
            original,
            0,
            255,
        ).astype(np.uint8)

    # Normalize the heatmap safely.
    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError(
            f"Expected a 2D heatmap; got shape {heatmap.shape}."
        )

    heatmap = np.nan_to_num(
        heatmap,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    height, width = original.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.clip(
        heatmap_resized * 255.0,
        0,
        255,
    ).astype(np.uint8)

    # OpenCV's color map is BGR.
    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    # Convert to RGB for consistent handling.
    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    original = np.ascontiguousarray(
        original,
        dtype=np.uint8,
    )

    heatmap_rgb = np.ascontiguousarray(
        heatmap_rgb,
        dtype=np.uint8,
    )

    # Both arrays now have the same shape and dtype.
    overlay = cv2.addWeighted(
        original,
        1.0 - alpha,
        heatmap_rgb,
        alpha,
        0.0,
    )

    return np.ascontiguousarray(
        overlay,
        dtype=np.uint8,
    )


# ============================================================
# PREDICTIONS
# ============================================================

def get_predictions(model, input_tensor):
    """
    Return predictions sorted by descending model score.

    Each result contains:
        index
        label
        probability
    """

    output = model.predict(
        input_tensor,
        verbose=0,
    )

    if isinstance(output, list):
        raise ValueError(
            "Expected a single model output, but multiple outputs "
            "were returned."
        )

    scores = np.asarray(output)[0]

    if scores.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Model returned {scores.shape[0]} outputs, "
            f"but {NUM_CLASSES} class names are configured. "
            "Ensure the four-class model is loaded."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "The model returned NaN or infinite prediction scores."
        )

    results = []

    for index, score in enumerate(scores):
        results.append(
            {
                "index": index,
                "label": CLASS_NAMES[index],
                "probability": float(score),
            }
        )

    results.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return results
