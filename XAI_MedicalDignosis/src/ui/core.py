
"""
Explainable Chest X-Ray Classification

Custom CNN + ResNet50 + EfficientNetB0 Transfer Learning
Grad-CAM v2 + SHAP + Gemini LLM

Academic research prototype; not a medical diagnostic tool.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

from src.transfer_learning.models import load_resnet50_checkpoint

from src.explainability.gradcam_v2 import (
    enhance_heatmap,
    create_overlay,
)

from src.explainability.shap_explainer import (
    create_shap_explainer,
    explain_image,
)

from src.explainability.llm_explainer import (
    generate_explanation,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATHS = {
    "Custom CNN": (
        BASE_DIR
        / "models"
        / "kaggle_chest_xray_covid4_best.keras"
    ),

    "ResNet50": (
        BASE_DIR
        / "models"
        / "resnet50_best.keras"
    ),

    "EfficientNetB0": (
        BASE_DIR
        / "models"
        / "efficientnetb0_best.keras"
    ),
}

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]

CNN_GRADCAM_LAYER = "activation_3"
RESNET_GRADCAM_LAYER = "conv4_block6_out"
EFFICIENTNET_GRADCAM_LAYER = "top_activation"

SHAP_MAX_EVALS = 1000
SHAP_BATCH_SIZE = 8


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_selected_model(model_name):

    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    model_path = MODEL_PATHS[model_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_name} model not found: {model_path}"
        )

    if model_name == "Custom CNN":

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    elif model_name == "ResNet50":

        model = load_resnet50_checkpoint(
            model_path
        )

    elif model_name == "EfficientNetB0":

        # EfficientNetB0 contains its own preprocessing layers.
        # Supply RGB float32 images in the 0-255 range, just as
        # during training. Do not apply an additional /255 here.
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    if isinstance(model.output_shape, list):
        raise ValueError(
            f"{model_name} has multiple outputs."
        )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} must return four class scores. "
            f"Received: {model.output_shape[-1]}"
        )

    return model


def load_available_models():

    available_models = {}
    model_errors = {}

    for model_name in MODEL_PATHS:

        try:
            available_models[model_name] = (
                load_selected_model(model_name)
            )

        except Exception as exc:
            model_errors[model_name] = str(exc)

    return available_models, model_errors


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_image):

    resized = pil_image.convert("RGB").resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    # All three models receive RGB float32 images in the 0-255 range.
    # Custom CNN performs internal rescaling.
    # ResNet50 uses its registered preprocessing layer.
    # EfficientNetB0 uses preprocessing within its backbone.

    return image_array, image_array[None, ...]


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_class_predictions(model, input_tensor):

    probabilities = np.asarray(
        model(
            input_tensor,
            training=False,
        ).numpy()
    )[0]

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            "Prediction output does not match configured classes."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Model returned invalid prediction scores."
        )

    predictions = [
        {
            "index": i,
            "label": name,
            "probability": float(probabilities[i]),
        }
        for i, name in enumerate(CLASS_NAMES)
    ]

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return predictions


def get_all_model_predictions(
    input_tensor,
    available_models,
):

    all_predictions = {}

    for model_name, model in available_models.items():

        all_predictions[model_name] = (
            get_class_predictions(
                model,
                input_tensor,
            )
        )

    return all_predictions


def get_best_prediction(all_predictions):

    candidates = []

    for model_name, predictions in all_predictions.items():

        for prediction in predictions:

            candidates.append({
                **prediction,
                "model": model_name,
            })

    if not candidates:
        raise RuntimeError(
            "No predictions were generated."
        )

    return max(
        candidates,
        key=lambda item: item["probability"],
    )


# ============================================================
# SHAP PREDICTION FUNCTION
# ============================================================

def predict_for_shap(images, model_name):

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    if (
        images.ndim != 4
        or images.shape[1:] != (224, 224, 3)
    ):
        raise ValueError(
            f"Unexpected SHAP input shape: {images.shape}"
        )

    model = load_selected_model(model_name)

    outputs = model(
        images,
        training=False,
    )

    probabilities = np.asarray(
        outputs.numpy(),
        dtype=np.float32,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(CLASS_NAMES)
    ):
        raise ValueError(
            f"Unexpected model output shape: {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Invalid scores returned during SHAP prediction."
        )

    return probabilities


@st.cache_resource
def load_shap_explainer(model_name):

    return create_shap_explainer(
        lambda images: predict_for_shap(
            images,
            model_name,
        )
    )


# ============================================================
# SHAP VISUALIZATION
# ============================================================

def generate_shap_visualization(
    original_image,
    class_values,
    class_label,
    model_name,
):

    spatial_map = np.sum(
        class_values,
        axis=-1,
    )

    if not np.all(np.isfinite(spatial_map)):
        raise ValueError(
            "SHAP visualization contains invalid values."
        )

    max_abs = float(
        np.max(np.abs(spatial_map))
    )

    if max_abs < 1e-12:
        raise ValueError(
            "SHAP returned an effectively zero attribution map."
        )

    normalized_map = spatial_map / max_abs

    fig, ax = plt.subplots(
        figsize=(8, 7),
    )

    ax.imshow(
        original_image.astype(np.uint8),
    )

    heatmap = ax.imshow(
        normalized_map,
        cmap="seismic",
        alpha=0.45,
        vmin=-1,
        vmax=1,
    )

    ax.set_title(
        f"SHAP Explanation - {class_label}"
    )

    ax.axis("off")

    colorbar = fig.colorbar(
        heatmap,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Relative signed SHAP attribution"
    )

    fig.tight_layout()

    return fig


# ============================================================
# GRAD-CAM MODEL CONSTRUCTION
# ============================================================

def get_gradcam_layer_options(model_name):
    """Return spatial feature layers that actually exist in the model."""

    if model_name == "Custom CNN":
        model = load_selected_model(model_name)
        preferred = ["activation_3", "conv2d_3"]
        available = {
            layer.name
            for layer in model.layers
            if len(getattr(layer.output, "shape", ())) == 4
        }
        return [name for name in preferred if name in available]

    if model_name == "ResNet50":
        model = load_selected_model(model_name)
        backbone = model.get_layer("resnet50")

        preferred = [
            "conv4_block6_out",
            "conv5_block3_out",
        ]

        available = {
            layer.name
            for layer in backbone.layers
            if len(getattr(layer.output, "shape", ())) == 4
        }

        # Prefer the requested stages, but never offer a layer
        # that is absent from the actual saved backbone.
        options = [name for name in preferred if name in available]

        if not options:
            # Fallback: expose the final few spatial outputs
            # from this actual backbone, not guessed layer names.
            spatial_layers = [
                layer.name
                for layer in backbone.layers
                if len(getattr(layer.output, "shape", ())) == 4
            ]
            options = spatial_layers[-5:]

        return options

    if model_name == "EfficientNetB0":
        model = load_selected_model(model_name)
        backbone = model.get_layer("efficientnetb0")

        preferred = [
            EFFICIENTNET_GRADCAM_LAYER,
            "block6d_add",
            "block5c_add",
        ]

        available = {
            layer.name
            for layer in backbone.layers
            if len(getattr(layer.output, "shape", ())) == 4
        }

        options = [name for name in preferred if name in available]

        if not options:
            spatial_layers = [
                layer.name
                for layer in backbone.layers
                if len(getattr(layer.output, "shape", ())) == 4
            ]
            options = spatial_layers[-5:]

        return options

    raise ValueError(f"Unsupported model: {model_name}")


@st.cache_resource
def load_gradcam_model(model_name, layer_name):
    """Build a Grad-CAM model with a connected feature-to-score graph."""

    model = load_selected_model(model_name)

    if model_name == "Custom CNN":
        try:
            feature_layer = model.get_layer(layer_name)
        except ValueError as exc:
            raise ValueError(
                f"Grad-CAM layer '{layer_name}' was not found in Custom CNN."
            ) from exc

        if len(feature_layer.output.shape) != 4:
            raise ValueError(
                f"Layer '{layer_name}' does not produce a spatial feature map."
            )

        return tf.keras.Model(
            inputs=model.inputs,
            outputs=[feature_layer.output, model.output],
            name=f"gradcam_cnn_{layer_name}",
        )

    if model_name == "ResNet50":
        try:
            backbone = model.get_layer("resnet50")
            preprocessing_layer = model.get_layer("resnet50_preprocessing")
            feature_layer = backbone.get_layer(layer_name)

            gap_layer = model.get_layer("global_average_pooling")
            bn_layer = model.get_layer("classifier_batch_normalization")
            dropout_layer = model.get_layer("classifier_dropout")
            classifier_layer = model.get_layer("four_class_classifier")
        except ValueError as exc:
            raise ValueError(
                f"A required ResNet50 Grad-CAM layer was not found: {exc}"
            ) from exc

        if len(feature_layer.output.shape) != 4:
            raise ValueError(
                f"Layer '{layer_name}' does not produce a spatial feature map."
            )

        # This model returns the chosen intermediate activation and the
        # final backbone output from ONE call to the same backbone.
        backbone_with_features = tf.keras.Model(
            inputs=backbone.input,
            outputs=[feature_layer.output, backbone.output],
            name=f"resnet50_features_{layer_name}",
        )

        model_input = model.inputs[0]
        preprocessed = preprocessing_layer(model_input)
        feature_maps, backbone_output = backbone_with_features(
            preprocessed,
            training=False,
        )

        # Reuse the trained classifier head. Keep BN/dropout in inference mode.
        x = gap_layer(backbone_output)
        x = bn_layer(x, training=False)
        x = dropout_layer(x, training=False)
        class_scores = classifier_layer(x)

        return tf.keras.Model(
            inputs=model_input,
            outputs=[feature_maps, class_scores],
            name=f"gradcam_resnet50_{layer_name}",
        )

    if model_name == "EfficientNetB0":
        try:
            backbone = model.get_layer("efficientnetb0")
            feature_layer = backbone.get_layer(layer_name)

            gap_layer = model.get_layer("global_average_pooling")
            bn_layer = model.get_layer("classifier_batch_normalization")
            dropout_layer = model.get_layer("classifier_dropout")
            classifier_layer = model.get_layer("four_class_classifier")
        except ValueError as exc:
            raise ValueError(
                f"A required EfficientNetB0 Grad-CAM layer was not found: {exc}"
            ) from exc

        if len(feature_layer.output.shape) != 4:
            raise ValueError(
                f"Layer '{layer_name}' does not produce a spatial feature map."
            )

        # EfficientNetB0 includes its preprocessing inside the backbone.
        # Reuse the same backbone call for the selected activation and
        # final feature tensor, then reuse the trained classifier head.
        backbone_with_features = tf.keras.Model(
            inputs=backbone.input,
            outputs=[feature_layer.output, backbone.output],
            name=f"efficientnetb0_features_{layer_name}",
        )

        model_input = model.inputs[0]
        feature_maps, backbone_output = backbone_with_features(
            model_input,
            training=False,
        )

        x = gap_layer(backbone_output)
        x = bn_layer(x, training=False)
        x = dropout_layer(x, training=False)
        class_scores = classifier_layer(x)

        return tf.keras.Model(
            inputs=model_input,
            outputs=[feature_maps, class_scores],
            name=f"gradcam_efficientnetb0_{layer_name}",
        )

    raise ValueError(f"Unsupported model: {model_name}")


# ============================================================
# GRAD-CAM V2 GENERATION
# ============================================================

def generate_gradcam_v2(
    input_tensor,
    original_image,
    class_index,
    model_name,
    layer_name,
    alpha=0.45,
):
    """Generate a class-specific Grad-CAM overlay."""

    grad_model = load_gradcam_model(model_name, layer_name)
    image_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            image_tensor,
            training=False,
        )
        target_score = tf.reduce_sum(
            predictions[:, int(class_index)]
        )

    gradients = tape.gradient(target_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            f"Grad-CAM gradients are disconnected for {model_name}, "
            f"layer '{layer_name}'."
        )

    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise RuntimeError("Non-finite Grad-CAM gradients were detected.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(
        feature_maps[0] * channel_weights[0][None, None, :],
        axis=-1,
    )
    cam = tf.nn.relu(cam).numpy().astype(np.float32)

    cam = cv2.resize(
        cam,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.maximum(cam, 0.0)

    cam_max = float(cam.max())
    cam_std = float(cam.std())
    gradient_max = float(tf.reduce_max(tf.abs(gradients)).numpy())

    print("\n========== GRAD-CAM DIAGNOSTICS ==========")
    print("Model:", model_name)
    print("Layer:", layer_name)
    print("Feature maps:", tuple(feature_maps.shape))
    print("Gradients:", tuple(gradients.shape))
    print("Maximum absolute gradient:", gradient_max)
    print("Raw CAM max/std:", cam_max, cam_std)
    print("==========================================\n")

    if cam_max <= 1e-12:
        raise RuntimeError(
            f"Grad-CAM generated an all-zero positive heatmap for "
            f"{model_name}, layer '{layer_name}'. Try another available "
            "spatial layer; an artificial heatmap will not be substituted."
        )

    cam = cam / cam_max
    enhanced_heatmap = enhance_heatmap(cam)
    enhanced_heatmap = np.asarray(enhanced_heatmap, dtype=np.float32)

    overlay_result = create_overlay(
        np.asarray(original_image, dtype=np.uint8),
        enhanced_heatmap,
        alpha=alpha,
    )

    # create_overlay() returns (overlay_rgb, colored_heatmap_rgb).
    overlay = (
        overlay_result[0]
        if isinstance(overlay_result, tuple)
        else overlay_result
    )
    overlay = np.asarray(overlay)

    original = np.asarray(original_image, dtype=np.uint8)
    if overlay.shape != original.shape:
        raise ValueError(
            f"Overlay shape {overlay.shape} does not match image {original.shape}."
        )

    difference = float(
        np.mean(np.abs(overlay.astype(np.float32) - original.astype(np.float32)))
    )
    print("Overlay mean pixel difference:", difference)

    if difference < 0.1:
        raise RuntimeError(
            "Grad-CAM overlay is almost identical to the original image. "
            "Inspect the raw CAM diagnostics."
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)




# ============================================================
# SHARED SESSION STATE
# ============================================================

def initialize_session_state():
    defaults = {
        "xray_original": None,
        "xray_input_tensor": None,
        "xray_filename": None,
        "all_predictions": None,
        "best_prediction": None,
        "gradcam_overlay": None,
        "gradcam_layer": None,
        "shap_figure": None,
        "gemini_explanation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_analysis():
    initialize_session_state()
    return (
        st.session_state.get("xray_original") is not None
        and st.session_state.get("all_predictions") is not None
        and st.session_state.get("best_prediction") is not None
    )
