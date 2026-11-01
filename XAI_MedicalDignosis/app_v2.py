
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

BASE_DIR = Path(__file__).resolve().parent

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


st.set_page_config(
    page_title="XAI Chest X-Ray",
    page_icon="🩻",
    layout="wide",
)


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
# APPLICATION INTERFACE
# ============================================================

st.title(
    "🩻 Explainable AI Chest X-Ray Analysis"
)

st.write(
    "Four-class chest X-ray classification using a Custom CNN, "
    "ResNet50 and EfficientNetB0, Grad-CAM v2, SHAP visual "
    "explanations, and a Gemini-generated educational explanation."
)

st.warning(
    "Research and educational prototype only. This model is "
    "not clinically validated and must not be used to diagnose "
    "disease or make treatment decisions. Model scores are not "
    "confirmed diagnoses."
)


# ============================================================
# ABOUT THE MODEL
# ============================================================

with st.expander("About this model"):

    st.markdown("""
This application uses three deep learning models:

- **Custom CNN**
- **ResNet50 Transfer Learning**
- **EfficientNetB0 Transfer Learning**

All three models classify chest X-rays into four classes:

- **Normal**
- **Pneumonia**
- **Tuberculosis**
- **COVID-19**

The application compares their output scores. The highest
class score across available models is displayed as the
primary prediction.

Grad-CAM v2 highlights image regions that influence the
selected model output.

SHAP estimates how image regions contribute to a selected
model output relative to a masking baseline.

These visualizations are not proof of disease. Models may
produce incorrect predictions due to image quality, dataset
differences, class imbalance, and other factors.
""")


# ============================================================
# MODEL EVALUATION RESULTS
# ============================================================

with st.expander("Model evaluation results"):

    st.caption(
        "Recorded held-out test-set results. These evaluation "
        "metrics are different from the prediction scores "
        "generated for an individual uploaded X-ray."
    )

    cnn_col, resnet_col, efficientnet_col = st.columns(3)

    # --------------------------------------------------------
    # CUSTOM CNN
    # --------------------------------------------------------

    with cnn_col:

        st.subheader("Custom CNN")

        st.metric(
            "Test accuracy",
            "72.00%",
        )

        st.metric(
            "Macro precision",
            "71.00%",
        )

        st.metric(
            "Macro recall",
            "67.03%",
        )

        st.metric(
            "Macro F1-score",
            "66.17%",
        )

        st.metric(
            "Weighted F1-score",
            "70.95%",
        )

        st.markdown("**Class-wise recall limitations**")

        st.write("COVID-19: 40.86%")

        st.write("Tuberculosis: 47.58%")

    # --------------------------------------------------------
    # RESNET50
    # --------------------------------------------------------

    with resnet_col:

        st.subheader("ResNet50 Transfer Learning")

        st.metric(
            "Test accuracy",
            "99.28%",
        )

        st.metric(
            "Macro precision",
            "99.27%",
        )

        st.metric(
            "Macro recall",
            "99.40%",
        )

        st.metric(
            "Macro F1-score",
            "99.33%",
        )

        st.metric(
            "ROC-AUC",
            "99.98%",
        )

        st.markdown("**Class-wise recall**")

        st.write("Normal: 99.70%")

        st.write("Pneumonia: 99.77%")

        st.write("Tuberculosis: 98.94%")

        st.write("COVID-19: 99.17%")

    # --------------------------------------------------------
    # EFFICIENTNETB0
    # --------------------------------------------------------

    with efficientnet_col:

        st.subheader("EfficientNetB0 Transfer Learning")

        st.metric(
            "Test accuracy",
            "99.56%",
        )

        st.metric(
            "Macro precision",
            "99.55%",
        )

        st.metric(
            "Macro recall",
            "99.63%",
        )

        st.metric(
            "Macro F1-score",
            "99.59%",
        )

        st.metric(
            "ROC-AUC",
            "99.99%",
        )

        st.markdown("**Class-wise recall**")
        st.write("Normal: 100.00%")
        st.write("Pneumonia: 100.00%")
        st.write("Tuberculosis: 98.79%")
        st.write("COVID-19: 99.72%")

    st.divider()

    st.warning(
        "The three models' metrics were recorded from separate "
        "evaluation runs and may use different test splits. They "
        "must not be treated as a definitive head-to-head comparison "
        "until all models are evaluated on the exact same held-out "
        "images with consistent patient-level splitting and leakage checks."
    )

    st.caption(
        "Neither model is clinically validated. High test "
        "performance does not guarantee correct predictions "
        "on new patients or hospitals."
    )


# ============================================================
# LOAD AVAILABLE MODELS
# ============================================================

try:

    available_models, model_errors = load_available_models()

except Exception as exc:

    st.error(
        "Could not initialize the models."
    )

    st.exception(exc)

    st.stop()


if not available_models:

    st.error(
        "No trained models could be loaded."
    )

    for model_name, error in model_errors.items():

        st.write(
            f"**{model_name}:** {error}"
        )

    st.stop()


if model_errors:

    with st.expander("Model loading warnings"):

        for model_name, error in model_errors.items():

            st.warning(
                f"{model_name}: {error}"
            )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray",
    type=["png", "jpg", "jpeg"],
    help="Upload a chest X-ray image in PNG or JPEG format.",
)


if uploaded_file is not None:

    try:

        with Image.open(uploaded_file) as source:

            display_image = source.convert(
                "RGB"
            ).copy()

    except Exception as exc:

        st.error(
            f"Could not read the uploaded image: {exc}"
        )

        st.stop()

    st.subheader(
        "Uploaded X-ray"
    )

    st.image(
        display_image,
        caption=uploaded_file.name,
        use_container_width=True,
    )

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "Analyze X-ray",
        type="primary",
    ):

        try:

            # ------------------------------------------------
            # IMAGE PREPROCESSING
            # ------------------------------------------------

            with st.spinner(
                "Preparing the chest X-ray..."
            ):

                original_image, input_tensor = preprocess_image(
                    display_image
                )

            # ------------------------------------------------
            # RUN ALL AVAILABLE MODELS
            # ------------------------------------------------

            with st.spinner(
                "Running available classification models..."
            ):

                all_predictions = get_all_model_predictions(
                    input_tensor,
                    available_models,
                )

                best_prediction = get_best_prediction(
                    all_predictions
                )

            # ------------------------------------------------
            # WINNING MODEL
            # ------------------------------------------------

            selected_label = best_prediction["label"]

            selected_score = best_prediction["probability"]

            selected_index = best_prediction["index"]

            selected_model_name = best_prediction["model"]

            selected_model = available_models[
                selected_model_name
            ]

            selected_model_predictions = all_predictions[
                selected_model_name
            ]

            # =================================================
            # SECTION 1: CNN PREDICTION
            # =================================================

            st.divider()

            st.header(
                "1. CNN Prediction"
            )

            st.metric(
                label="Highest-scoring class",
                value=selected_label,
                delta=(
                    f"{selected_model_name} | "
                    f"Model score: {selected_score * 100:.2f}%"
                ),
            )

            st.caption(
                "The main prediction is selected by comparing "
                "the output scores of all available models. "
                "These scores are not necessarily calibrated "
                "probabilities and do not establish disease presence."
            )

            # ------------------------------------------------
            # SCORES FOR ALL CLASSES
            # ------------------------------------------------

            st.subheader(
                "Scores for all classes"
            )

            model_columns = st.columns(len(all_predictions))

            for (model_name, predictions), model_column in zip(
                all_predictions.items(),
                model_columns,
            ):

                with model_column:

                    st.markdown(
                        f"**{model_name}**"
                    )

                    for item in predictions:

                        st.write(
                            f"**{item['label']}:** "
                            f"{item['probability'] * 100:.2f}%"
                        )

                        st.progress(
                            float(
                                np.clip(
                                    item["probability"],
                                    0.0,
                                    1.0,
                                )
                            )
                        )

                    model_best = max(
                        predictions,
                        key=lambda item: item["probability"],
                    )

                    st.caption(
                        f"Highest score: {model_best['label']} "
                        f"({model_best['probability'] * 100:.2f}%)"
                    )

            # =================================================
            # GRAD-CAM SETTINGS
            # =================================================

            available_layers = get_gradcam_layer_options(
                selected_model_name
            )

            if not available_layers:
                raise ValueError(
                    f"No spatial Grad-CAM layers were found for {selected_model_name}."
                )

            preferred_defaults = {
                "Custom CNN": CNN_GRADCAM_LAYER,
                "ResNet50": RESNET_GRADCAM_LAYER,
                "EfficientNetB0": EFFICIENTNET_GRADCAM_LAYER,
            }
            preferred_default = preferred_defaults[selected_model_name]

            default_layer = (
                preferred_default
                if preferred_default in available_layers
                else available_layers[-1]
            )

            with st.expander("Grad-CAM v2 settings"):

                selected_layer = st.selectbox(
                    "Feature layer",
                    options=available_layers,
                    index=available_layers.index(default_layer),
                    help=(
                        "Only spatial layers found in the selected model "
                        "are offered. Changing this setting does not alter weights."
                    ),
                )

                overlay_alpha = st.slider(
                    "Overlay opacity",
                    min_value=0.10,
                    max_value=0.80,
                    value=0.45,
                    step=0.05,
                )

            # =================================================
            # SECTION 2: GRAD-CAM V2
            # =================================================

            st.divider()

            st.header(
                "2. Grad-CAM v2 Visual Explanation"
            )

            with st.spinner(
                f"Generating Grad-CAM for {selected_model_name}..."
            ):

                overlay = generate_gradcam_v2(
                    input_tensor=input_tensor,
                    original_image=original_image,
                    class_index=selected_index,
                    model_name=selected_model_name,
                    layer_name=selected_layer,
                    alpha=overlay_alpha,
                )

            image_col1, image_col2 = st.columns(2)

            with image_col1:

                st.subheader(
                    "Original X-ray"
                )

                st.image(
                    original_image.astype(np.uint8),
                    use_container_width=True,
                )

            with image_col2:

                st.subheader(
                    "Grad-CAM v2 Overlay"
                )

                st.image(
                    overlay,
                    use_container_width=True,
                )

            st.caption(
                f"Explained class: {selected_label} | "
                f"Model: {selected_model_name} | "
                f"Feature layer: {selected_layer}"
            )

            st.info(
                "Grad-CAM highlights image regions that influence "
                "the selected model output. It does not prove that "
                "a disease exists and may highlight irrelevant "
                "image features."
            )

            # =================================================
            # SECTION 3: SHAP EXPLANATION
            # =================================================

            st.divider()

            st.header(
                "3. SHAP Visual Explanation"
            )

            try:

                with st.spinner(
                    f"Generating SHAP explanation for "
                    f"{selected_model_name}..."
                ):

                    shap_explainer = load_shap_explainer(
                        selected_model_name
                    )

                    shap_explanation, shap_values = explain_image(
                        explainer=shap_explainer,
                        image=original_image,
                        class_index=selected_index,
                        max_evals=SHAP_MAX_EVALS,
                        batch_size=SHAP_BATCH_SIZE,
                    )

                shap_fig = generate_shap_visualization(
                    original_image=original_image,
                    class_values=shap_values,
                    class_label=selected_label,
                    model_name=selected_model_name,
                )

                st.pyplot(
                    shap_fig,
                    use_container_width=True,
                )

                plt.close(shap_fig)

                st.caption(
                    f"Explained class: {selected_label} | "
                    f"Model: {selected_model_name} | "
                    f"SHAP evaluations: {SHAP_MAX_EVALS}"
                )

                st.info(
                    "SHAP estimates how image regions contribute "
                    "to the selected model output relative to the "
                    "image masking baseline. Red and blue regions "
                    "represent positive and negative contributions "
                    "to that output, respectively. These attributions "
                    "do not establish disease presence or severity."
                )

            except Exception as shap_error:

                st.error(
                    "SHAP explanation could not be generated."
                )

                st.exception(
                    shap_error
                )

                st.caption(
                    "The prediction and Grad-CAM results are still "
                    "available. You can continue to the Gemini explanation."
                )

            # =================================================
            # SECTION 4: GEMINI EXPLANATION
            # =================================================

            st.divider()

            st.header(
                "4. Gemini-Generated Explanation"
            )

            try:

                with st.spinner(
                    "Generating an educational explanation..."
                ):

                    explanation = generate_explanation(
                        predictions=selected_model_predictions,
                        selected_label=selected_label,
                        gradcam_layer=selected_layer,
                    )

                if explanation:

                    st.markdown(
                        explanation
                    )

                else:

                    st.info(
                        "No explanation was returned. "
                        "Check your Gemini configuration and API key."
                    )

            except Exception as gemini_error:

                st.warning(
                    "Gemini explanation is temporarily unavailable. "
                    "Your prediction, Grad-CAM and SHAP results "
                    "are still available."
                )

                st.caption(
                    str(gemini_error)
                )

            st.caption(
                f"The language model explains the supplied "
                f"{selected_model_name} prediction and Grad-CAM "
                "metadata for educational purposes. It does not "
                "independently examine the X-ray or confirm a diagnosis."
            )

        except Exception as exc:

            st.error(
                "Analysis could not be completed."
            )

            st.exception(exc)
