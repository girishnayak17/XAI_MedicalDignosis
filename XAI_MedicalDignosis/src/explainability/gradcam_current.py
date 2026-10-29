
"""
Grad-CAM utilities for the four-class Kaggle Chest X-ray CNN.

Class order:
    0. Normal
    1. Pneumonia
    2. Tuberculosis
    3. COVID-19

The model contains a Rescaling(1./255) layer internally.
Input pixels must remain in [0, 255]. Do not divide by 255 here.
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

DEFAULT_GRADCAM_LAYER = "activation_3"
MIN_HEATMAP_VALUE = 1e-8


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(image_path):
    """
    Load an image and prepare the model input.

    Returns:
        original_image:
            RGB uint8 image at its original dimensions.
        input_tensor:
            Float32 batch with shape (1, 224, 224, 3),
            pixel values in [0, 255].
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

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
            ).copy()

    except Exception as exc:
        raise ValueError(
            f"Could not load image: {image_path}"
        ) from exc

    input_tensor = image_array[np.newaxis, ...]

    return original_image, input_tensor


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

def find_last_conv_layer(model):
    """
    Find the last Conv2D layer, including nested models.

    Note:
        A layer returned from a nested model may not be directly
        connected to the outer model graph. In that situation,
        build_gradcam_model() raises an explicit error rather than
        silently returning an unrelated graph.
    """

    def search_layers(layers):
        for layer in reversed(layers):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D,
            ):
                return layer

            if isinstance(layer, tf.keras.Model):
                result = search_layers(layer.layers)

                if result is not None:
                    return result

        return None

    result = search_layers(model.layers)

    if result is None:
        raise ValueError(
            "No Conv2D layer was found in the model."
        )

    return result


# ============================================================
# BUILD GRAD-CAM MODEL
# ============================================================

def build_gradcam_model(model, feature_layer):
    """
    Build a model returning:
        1. Selected spatial feature maps.
        2. Final model predictions.

    The selected layer must belong to the computational graph
    connecting the original model inputs to its predictions.
    """

    if not isinstance(
        feature_layer,
        tf.keras.layers.Layer,
    ):
        raise TypeError(
            "feature_layer must be a Keras layer."
        )

    try:
        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                feature_layer.output,
                model.outputs[0],
            ],
            name="gradcam_model",
        )

    except (ValueError, AttributeError) as exc:
        raise ValueError(
            f"Cannot connect feature layer "
            f"'{feature_layer.name}' to the original model output. "
            "Choose a spatial layer connected to the model's "
            "final predictions."
        ) from exc


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    class_index,
    layer_name=DEFAULT_GRADCAM_LAYER,
):
    """
    Generate standard Grad-CAM for one image and one class.

    Args:
        model:
            Trained four-class Keras model.
        input_tensor:
            Float32 input with shape (1, 224, 224, 3),
            pixel values in [0, 255].
        class_index:
            0=Normal, 1=Pneumonia, 2=Tuberculosis, 3=COVID-19.
        layer_name:
            Spatial feature layer, default "activation_3".
            "conv2d_3" can also be selected for comparison.

    Returns:
        heatmap:
            Normalized 2D float32 array in [0, 1].
        selected_score:
            Model score for the selected class.
        selected_layer_name:
            Name of the layer used.

    Important:
        A zero heatmap means that no positive Grad-CAM evidence
        was obtained from the selected layer for this class.
        It must not be artificially amplified.
    """

    # --------------------------------------------------------
    # Validate class index
    # --------------------------------------------------------

    if not isinstance(
        class_index,
        (int, np.integer),
    ):
        raise TypeError(
            "class_index must be an integer."
        )

    class_index = int(class_index)

    if not 0 <= class_index < NUM_CLASSES:
        raise ValueError(
            f"class_index must be between 0 and "
            f"{NUM_CLASSES - 1}."
        )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    input_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32,
    )

    if input_tensor.shape.rank != 4:
        raise ValueError(
            "Expected input shape (1, 224, 224, 3)."
        )

    if input_tensor.shape[-1] != 3:
        raise ValueError(
            "Expected a three-channel RGB input."
        )

    if input_tensor.shape[0] != 1:
        raise ValueError(
            "Generate Grad-CAM for one image at a time."
        )

    tf.debugging.assert_all_finite(
        input_tensor,
        "Input contains NaN or infinite values.",
    )

    # --------------------------------------------------------
    # Select spatial feature layer
    # --------------------------------------------------------

    if not isinstance(layer_name, str) or not layer_name.strip():
        raise ValueError(
            "layer_name must be a non-empty string."
        )

    try:
        feature_layer = model.get_layer(layer_name)

    except ValueError as exc:
        available_layers = [
            layer.name for layer in model.layers
        ]

        raise ValueError(
            f"Layer '{layer_name}' was not found. "
            f"Available top-level layers: {available_layers}"
        ) from exc

    output_shape = feature_layer.output.shape

    if len(output_shape) != 4:
        raise ValueError(
            f"Layer '{layer_name}' must output a 4D spatial "
            f"feature map. Actual shape: {output_shape}"
        )

    # --------------------------------------------------------
    # Build graph and calculate gradients
    # --------------------------------------------------------

    grad_model = build_gradcam_model(
        model,
        feature_layer,
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            input_tensor,
            training=False,
        )

        if predictions.shape.rank != 2:
            raise ValueError(
                "Expected predictions with shape (batch, classes)."
            )

        if predictions.shape[-1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} output classes; "
                f"received {predictions.shape[-1]}."
            )

        # The selected class score must be calculated inside
        # GradientTape so its gradients can be recorded.
        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError(
            f"Gradients are None for layer '{layer_name}'. "
            "The selected layer may not be connected to the "
            "selected class score."
        )

    tf.debugging.assert_all_finite(
        gradients,
        "Gradients contain NaN or infinite values.",
    )

    # --------------------------------------------------------
    # Standard Grad-CAM calculation
    # --------------------------------------------------------

    # Average gradients across spatial dimensions.
    # Shape: (batch, 1, 1, channels)
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
        keepdims=True,
    )

    # Weighted sum of feature maps.
    # Shape after selecting the image: (height, width)
    raw_heatmap = tf.reduce_sum(
        channel_weights * feature_maps,
        axis=-1,
    )[0]

    # Retain positive evidence for the selected class.
    raw_heatmap = tf.nn.relu(raw_heatmap)

    max_value = float(
        tf.reduce_max(raw_heatmap).numpy()
    )

    heatmap_height = int(feature_maps.shape[1])
    heatmap_width = int(feature_maps.shape[2])

    # --------------------------------------------------------
    # Safe normalization
    # --------------------------------------------------------

    if (
        not np.isfinite(max_value)
        or max_value <= MIN_HEATMAP_VALUE
    ):
        # Preserve an empty map instead of amplifying numerical
        # noise into an apparently strong explanation.
        heatmap = np.zeros(
            (heatmap_height, heatmap_width),
            dtype=np.float32,
        )

    else:
        normalized_heatmap = (
            raw_heatmap / tf.reduce_max(raw_heatmap)
        )

        heatmap = normalized_heatmap.numpy().astype(
            np.float32
        )

        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        heatmap = np.clip(
            heatmap,
            0.0,
            1.0,
        )

    selected_score = float(
        predictions[0, class_index].numpy()
    )

    return (
        heatmap,
        selected_score,
        feature_layer.name,
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
    Blend a heatmap with an RGB image.

    Args:
        original_image:
            RGB uint8 image, grayscale image, or float image.
        heatmap:
            Two-dimensional heatmap, normally in [0, 1].
        alpha:
            Heatmap opacity between 0 and 1.

    Returns:
        RGB uint8 overlay suitable for Streamlit st.image().
    """

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be a finite number between 0 and 1."
        )

    # --------------------------------------------------------
    # Validate and convert original image
    # --------------------------------------------------------

    original = np.asarray(original_image)

    if original.size == 0:
        raise ValueError(
            "original_image is empty."
        )

    if original.ndim == 2:
        original = cv2.cvtColor(
            original,
            cv2.COLOR_GRAY2RGB,
        )

    elif (
        original.ndim == 3
        and original.shape[-1] == 1
    ):
        original = cv2.cvtColor(
            original[:, :, 0],
            cv2.COLOR_GRAY2RGB,
        )

    if (
        original.ndim != 3
        or original.shape[-1] != 3
    ):
        raise ValueError(
            f"Expected an RGB image; got shape {original.shape}."
        )

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

    # --------------------------------------------------------
    # Validate heatmap
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Resize heatmap to image dimensions
    # --------------------------------------------------------

    height, width = original.shape[:2]

    heatmap_resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.round(
        heatmap_resized * 255.0
    ).astype(np.uint8)

    # OpenCV produces a BGR color map.
    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Ensure matching dimensions, data types, and contiguous memory.
    original = np.ascontiguousarray(
        original,
        dtype=np.uint8,
    )

    heatmap_rgb = np.ascontiguousarray(
        heatmap_rgb,
        dtype=np.uint8,
    )

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

    The values are model scores. They should not automatically
    be interpreted as clinically calibrated probabilities.
    """

    output = model.predict(
        input_tensor,
        verbose=0,
    )

    if isinstance(output, (list, tuple, dict)):
        raise ValueError(
            "Expected a single model output, but multiple "
            "or named outputs were returned."
        )

    output = np.asarray(output)

    if output.ndim != 2 or output.shape[0] != 1:
        raise ValueError(
            "Expected model output shape (1, 4). "
            f"Received {output.shape}."
        )

    scores = output[0]

    if scores.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Model returned {scores.shape[0]} outputs, "
            f"but {NUM_CLASSES} class names are configured. "
            "Ensure the correct four-class model is loaded."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "The model returned NaN or infinite prediction scores."
        )

    results = [
        {
            "index": index,
            "label": CLASS_NAMES[index],
            "probability": float(score),
        }
        for index, score in enumerate(scores)
    ]

    results.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return results
