
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "Custom CNN": BASE_DIR / "models" / "kaggle_chest_xray_covid4_best.keras",
    "ResNet50": BASE_DIR / "models" / "resnet50_best.keras",
    "EfficientNetB0": BASE_DIR / "models" / "efficientnetb0_best.keras",
}

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]

DEFAULT_LAYERS = {
    "Custom CNN": "activation_3",
    "ResNet50": "conv4_block6_out",
    "EfficientNetB0": "top_activation",
}

BACKBONE_NAMES = {
    "ResNet50": "resnet50",
    "EfficientNetB0": "efficientnetb0",
}


# ============================================================
# MODEL LOADING
# ============================================================

def load_classification_model(model_name):
    """Load the requested trained model."""

    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")

    model_path = MODEL_PATHS[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_name} checkpoint not found: {model_path}"
        )

    if model_name == "Custom CNN":
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    elif model_name == "ResNet50":
        from src.transfer_learning.models import (
            load_resnet50_checkpoint
        )

        model = load_resnet50_checkpoint(model_path)

    elif model_name == "EfficientNetB0":
        from src.transfer_learning.efficientnetb0 import (
            load_efficientnetb0_checkpoint
        )

        model = load_efficientnetb0_checkpoint(model_path)

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} must produce {len(CLASS_NAMES)} class scores."
        )

    return model


# ============================================================
# DISCOVER AVAILABLE GRAD-CAM LAYERS
# ============================================================

def get_gradcam_layer_options(model, model_name):
    """Discover actual spatial layers in the selected architecture."""

    if model_name == "Custom CNN":

        spatial_layers = [
            layer.name
            for layer in model.layers
            if len(getattr(layer.output, "shape", ())) == 4
        ]

    elif model_name in BACKBONE_NAMES:

        backbone = model.get_layer(
            BACKBONE_NAMES[model_name]
        )

        spatial_layers = [
            layer.name
            for layer in backbone.layers
            if len(getattr(layer.output, "shape", ())) == 4
            and layer.name != "input_layer"
        ]

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    preferred = {
        "Custom CNN": [
            "activation_3",
            "conv2d_3",
        ],
        "ResNet50": [
            "conv4_block6_out",
            "conv5_block3_out",
        ],
        "EfficientNetB0": [
            "top_activation",
            "block6d_add",
            "block5c_add",
        ],
    }

    options = [
        name
        for name in preferred[model_name]
        if name in spatial_layers
    ]

    # Include the actual spatial layers as alternatives.
    if not options:
        options = spatial_layers[-5:]

    return options


# ============================================================
# CONNECTED GRAD-CAM GRAPH
# ============================================================

def build_gradcam_model(model, model_name, layer_name):
    """
    Build a Grad-CAM model connected to the original classifier.

    Custom CNN:
        Direct intermediate layer and original model output.

    ResNet50:
        Preprocessing -> backbone -> trained classifier head.

    EfficientNetB0:
        Backbone (including its internal preprocessing)
        -> trained classifier head.
    """

    # --------------------------------------------------------
    # CUSTOM CNN
    # --------------------------------------------------------

    if model_name == "Custom CNN":

        try:
            feature_layer = model.get_layer(layer_name)
        except ValueError as exc:
            raise ValueError(
                f"Grad-CAM layer '{layer_name}' not found in Custom CNN."
            ) from exc

        if len(feature_layer.output.shape) != 4:
            raise ValueError(
                f"Layer '{layer_name}' is not a spatial feature layer."
            )

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                feature_layer.output,
                model.output,
            ],
            name=f"gradcam_cnn_{layer_name}",
        )

    # --------------------------------------------------------
    # RESNET50 / EFFICIENTNETB0
    # --------------------------------------------------------

    if model_name not in BACKBONE_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")

    backbone_name = BACKBONE_NAMES[model_name]

    try:
        backbone = model.get_layer(backbone_name)
        feature_layer = backbone.get_layer(layer_name)

        gap_layer = model.get_layer("global_average_pooling")
        bn_layer = model.get_layer("classifier_batch_normalization")
        dropout_layer = model.get_layer("classifier_dropout")
        classifier_layer = model.get_layer("four_class_classifier")

    except ValueError as exc:
        raise ValueError(
            f"Could not construct {model_name} Grad-CAM graph: {exc}"
        ) from exc

    if len(feature_layer.output.shape) != 4:
        raise ValueError(
            f"Layer '{layer_name}' does not produce spatial feature maps."
        )

    # Expose the selected intermediate feature maps and final
    # backbone output from the SAME backbone invocation.
    backbone_with_features = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            feature_layer.output,
            backbone.output,
        ],
        name=f"{model_name.lower()}_features_{layer_name}",
    )

    model_input = model.inputs[0]

    # ResNet50 has an external preprocessing layer.
    if model_name == "ResNet50":

        preprocessing_layer = model.get_layer(
            "resnet50_preprocessing"
        )

        backbone_input = preprocessing_layer(model_input)

    else:
        # EfficientNetB0 contains its own preprocessing layers.
        # Pass the original 0-255 RGB input directly.
        backbone_input = model_input

    feature_maps, backbone_output = backbone_with_features(
        backbone_input,
        training=False,
    )

    # Reuse the exact trained classifier head.
    x = gap_layer(backbone_output)
    x = bn_layer(x, training=False)
    x = dropout_layer(x, training=False)

    class_scores = classifier_layer(x)

    grad_model = tf.keras.Model(
        inputs=model_input,
        outputs=[
            feature_maps,
            class_scores,
        ],
        name=f"gradcam_{model_name.lower()}_{layer_name}",
    )

    return grad_model


# ============================================================
# PREDICTION EQUIVALENCE VALIDATION
# ============================================================

def validate_gradcam_graph(
    model,
    grad_model,
    input_tensor,
    tolerance=1e-5,
):
    """
    Ensure reconstructed Grad-CAM graph gives the same
    predictions as the original classification model.
    """

    original_scores = model(
        input_tensor,
        training=False,
    )

    feature_maps, grad_scores = grad_model(
        input_tensor,
        training=False,
    )

    difference = float(
        tf.reduce_max(
            tf.abs(original_scores - grad_scores)
        ).numpy()
    )

    print("\nGrad-CAM graph validation")
    print("Feature map shape:", tuple(feature_maps.shape))
    print("Original scores:", original_scores.numpy())
    print("Grad-CAM scores:", grad_scores.numpy())
    print("Maximum prediction difference:", difference)

    if difference > tolerance:
        raise RuntimeError(
            "Grad-CAM graph does not reproduce the original model "
            f"predictions. Difference: {difference}"
        )

    return difference


# ============================================================
# GRAD-CAM GENERATION
# ============================================================

def generate_gradcam(
    input_tensor,
    grad_model,
    class_index,
):
    """Calculate raw Grad-CAM for the specified class."""

    image_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            image_tensor,
            training=False,
        )

        class_index = int(class_index)

        if not 0 <= class_index < len(CLASS_NAMES):
            raise ValueError("Invalid class index.")

        target_score = tf.reduce_sum(
            predictions[:, class_index]
        )

    gradients = tape.gradient(
        target_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are disconnected."
        )

    if not bool(
        tf.reduce_all(
            tf.math.is_finite(gradients)
        ).numpy()
    ):
        raise RuntimeError(
            "Non-finite Grad-CAM gradients were detected."
        )

    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2),
    )

    cam = tf.reduce_sum(
        feature_maps[0]
        * channel_weights[0][None, None, :],
        axis=-1,
    )

    cam = tf.nn.relu(cam).numpy().astype(np.float32)

    cam = cv2.resize(
        cam,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cam = np.maximum(cam, 0.0)

    diagnostics = {
        "feature_shape": tuple(feature_maps.shape),
        "gradient_shape": tuple(gradients.shape),
        "gradient_max": float(
            tf.reduce_max(tf.abs(gradients)).numpy()
        ),
        "gradient_std": float(
            tf.math.reduce_std(gradients).numpy()
        ),
        "raw_cam_max": float(cam.max()),
        "raw_cam_mean": float(cam.mean()),
        "raw_cam_std": float(cam.std()),
    }

    if diagnostics["raw_cam_max"] <= 1e-12:
        raise RuntimeError(
            "Grad-CAM produced an all-zero positive heatmap. "
            f"Diagnostics: {diagnostics}"
        )

    cam /= diagnostics["raw_cam_max"]

    return cam, predictions[0].numpy(), diagnostics


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

    heatmap = np.clip(
        heatmap,
        0.0,
        1.0,
    )

    positive = heatmap[heatmap > 1e-8]

    if positive.size == 0:
        print("WARNING: No positive Grad-CAM activations.")
        return np.zeros_like(heatmap)

    upper = float(
        np.percentile(positive, 99)
    )

    if upper <= 1e-8:
        upper = float(positive.max())

    heatmap = np.clip(
        heatmap / max(upper, 1e-8),
        0.0,
        1.0,
    )

    heatmap = np.power(
        heatmap,
        0.75,
    )

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
    """Create a colored Grad-CAM heatmap and RGB overlay."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

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

    colored_bgr = cv2.applyColorMap(
        heatmap_uint8,
        colormap,
    )

    colored_rgb = cv2.cvtColor(
        colored_bgr,
        cv2.COLOR_BGR2RGB,
    )

    strength = np.clip(
        (heatmap - 0.02) / 0.98,
        0.0,
        1.0,
    )

    alpha_map = (
        alpha * np.power(strength, 0.65)
    )[..., None]

    overlay = (
        original.astype(np.float32) * (1.0 - alpha_map)
        + colored_rgb.astype(np.float32) * alpha_map
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)

    return overlay, colored_rgb


# ============================================================
# COMMON GRAD-CAM V2 INTERFACE
# ============================================================

def generate_gradcam_v2(
    input_tensor,
    original_image,
    class_index,
    model_name,
    layer_name=None,
    alpha=0.45,
    model=None,
    return_details=False,
):
    """
    Common Grad-CAM v2 interface for Custom CNN,
    ResNet50 and EfficientNetB0.

    The caller may pass an already-loaded model.
    """

    if model is None:
        model = load_classification_model(model_name)

    if layer_name is None:
        layer_name = DEFAULT_LAYERS[model_name]

    grad_model = build_gradcam_model(
        model,
        model_name,
        layer_name,
    )

    # Validate that the reconstructed graph follows the
    # original model's prediction path.
    difference = validate_gradcam_graph(
        model,
        grad_model,
        input_tensor,
    )

    raw_heatmap, class_scores, diagnostics = generate_gradcam(
        input_tensor,
        grad_model,
        class_index,
    )

    enhanced_heatmap = enhance_heatmap(
        raw_heatmap
    )

    overlay, colored_heatmap = create_overlay(
        original_image,
        enhanced_heatmap,
        alpha=alpha,
    )

    original = np.asarray(
        original_image,
        dtype=np.uint8,
    )

    pixel_difference = float(
        np.mean(
            np.abs(
                overlay.astype(np.float32)
                - original.astype(np.float32)
            )
        )
    )

    diagnostics.update({
        "model": model_name,
        "layer": layer_name,
        "prediction_difference": difference,
        "overlay_pixel_difference": pixel_difference,
    })

    print("\n========== GRAD-CAM V2 DIAGNOSTICS ==========")

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print("==============================================\n")

    if pixel_difference < 0.1:
        raise RuntimeError(
            "Grad-CAM overlay is almost identical to the original image."
        )

    if return_details:
        return {
            "overlay": overlay,
            "heatmap": enhanced_heatmap,
            "colored_heatmap": colored_heatmap,
            "raw_heatmap": raw_heatmap,
            "class_scores": class_scores,
            "diagnostics": diagnostics,
        }

    return overlay
