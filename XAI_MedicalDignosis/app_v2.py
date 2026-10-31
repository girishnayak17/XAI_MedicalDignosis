
"""
Explainable Chest X-Ray Classification
Four-Class CNN + Grad-CAM v2 + SHAP + Gemini LLM

Academic research prototype; not a medical diagnostic tool.
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

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

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "kaggle_chest_xray_covid4_best.keras"
)

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]

DEFAULT_GRADCAM_LAYER = "activation_3"

SHAP_MAX_EVALS = 1000
SHAP_BATCH_SIZE = 8


st.set_page_config(
    page_title="XAI Chest X-Ray",
    page_icon="🩻",
    layout="wide",
)


# ============================================================
# CNN MODEL LOADING
# ============================================================

@st.cache_resource
def load_cnn():
    """Load and validate the existing trained four-class CNN."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    output_shape = model.output_shape

    if isinstance(output_shape, list):
        raise ValueError(
            "Expected one model output, but multiple outputs were found."
        )

    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, "
            f"but the model returns {output_shape[-1]}."
        )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(pil_image):
    """
    Resize and convert image to RGB.

    Pixel values remain in 0-255 because the CNN contains
    an internal Rescaling(1/255) layer.
    """

    resized = pil_image.convert("RGB").resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized,
        dtype=np.float32,
    )

    return image_array, image_array[None, ...]


# ============================================================
# CNN PREDICTION
# ============================================================

def get_class_predictions(model, input_tensor):
    """Return sorted output scores for the four configured classes."""

    probabilities = np.asarray(
        model(input_tensor, training=False).numpy()
    )[0]

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            "Prediction output does not match configured classes."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "The model returned invalid prediction scores."
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


# ============================================================
# SHAP PREDICTION FUNCTION
# ============================================================

def predict_for_shap(images):
    """
    SHAP-compatible prediction function.

    SHAP supplies batches of modified RGB images.
    The function returns four CNN output scores per image.

    Preprocessing is intentionally identical to the
    original CNN preprocessing.
    """

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

    outputs = load_cnn()(
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
            f"Unexpected CNN output shape: {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "CNN returned invalid scores during SHAP prediction."
        )

    return probabilities


# ============================================================
# SHAP EXPLAINER LOADING
# ============================================================

@st.cache_resource
def load_shap_explainer():
    """
    Create and cache the SHAP image explainer.

    The explainer is reused across subsequent uploads.
    """

    return create_shap_explainer(
        predict_for_shap,
    )


# ============================================================
# SHAP VISUALIZATION
# ============================================================

def generate_shap_visualization(
    original_image,
    class_values,
    class_label,
):
    """
    Generate a signed SHAP attribution visualization.

    RGB channel attributions are summed to create
    a two-dimensional spatial attribution map.
    """

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
# GRAD-CAM MODEL
# ============================================================

@st.cache_resource
def load_gradcam_model(layer_name):
    """Expose selected spatial feature map and CNN predictions."""

    model = load_cnn()

    try:
        feature_layer = model.get_layer(
            layer_name,
        )

    except ValueError as exc:
        raise ValueError(
            f"Grad-CAM layer '{layer_name}' was not found."
        ) from exc

    if len(feature_layer.output.shape) != 4:
        raise ValueError(
            f"Layer '{layer_name}' does not produce "
            "a spatial feature map."
        )

    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            feature_layer.output,
            model.output,
        ],
        name=f"gradcam_v2_{layer_name}",
    )


# ============================================================
# GRAD-CAM V2 GENERATION
# ============================================================

def generate_gradcam_v2(
    input_tensor,
    original_image,
    class_index,
    layer_name=DEFAULT_GRADCAM_LAYER,
    alpha=0.45,
):
    """Calculate Grad-CAM internally and return one RGB overlay."""

    grad_model = load_gradcam_model(
        layer_name,
    )

    image_tensor = tf.convert_to_tensor(
        input_tensor,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        feature_maps, predictions = grad_model(
            image_tensor,
            training=False,
        )

        target_score = predictions[:, int(class_index)]

    gradients = tape.gradient(
        target_score,
        feature_maps,
    )

    if gradients is None:
        raise RuntimeError(
            f"Grad-CAM gradients unavailable for layer '{layer_name}'."
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

    cam = tf.nn.relu(cam).numpy().astype(
        np.float32
    )

    cam = cv2.resize(
        cam,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    cam = np.maximum(
        cam,
        0.0,
    )

    maximum = float(cam.max())

    cam = (
        cam / maximum
        if maximum > 1e-8
        else np.zeros_like(cam)
    )

    enhanced_heatmap = enhance_heatmap(
        np.clip(cam, 0.0, 1.0)
    )

    overlay_result = create_overlay(
        np.asarray(
            original_image,
            dtype=np.uint8,
        ),
        enhanced_heatmap,
        alpha=alpha,
    )

    # Support overlay or (overlay, colored_heatmap).

    overlay = (
        overlay_result[0]
        if isinstance(overlay_result, tuple)
        else overlay_result
    )

    overlay = np.asarray(
        overlay,
    )

    if overlay.ndim != 3 or overlay.shape[-1] != 3:
        raise ValueError(
            "create_overlay() must return an RGB overlay."
        )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# APPLICATION INTERFACE
# ============================================================

st.title(
    "🩻 Explainable AI Chest X-Ray Analysis"
)

st.write(
    "Four-class chest X-ray classification using a CNN, "
    "Grad-CAM v2, SHAP visual explanations, and "
    "a Gemini-generated educational explanation."
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
This application uses a CNN for four chest X-ray image classes:

- **Normal**
- **Pneumonia**
- **Tuberculosis**
- **COVID-19**

Grad-CAM v2 highlights image regions that influence a selected
model output.

SHAP estimates how image regions contribute to a selected
model output relative to a masking baseline.

These visualizations are not proof of disease. The model may
produce incorrect predictions due to image quality, dataset
differences, class imbalance, and other factors.
""")


# ============================================================
# MODEL EVALUATION RESULTS
# ============================================================

with st.expander("Model evaluation results"):

    st.caption(
        "Recorded held-out test-set results from the existing "
        "evaluation; they do not guarantee performance on new "
        "patients or hospitals."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test accuracy",
        "72.00%",
    )

    c2.metric(
        "Macro precision",
        "71.00%",
    )

    c3.metric(
        "Macro recall",
        "67.03%",
    )

    c4, c5 = st.columns(2)

    c4.metric(
        "Macro F1-score",
        "66.17%",
    )

    c5.metric(
        "Weighted F1-score",
        "70.95%",
    )

    st.markdown("""
**Important limitation:** During evaluation, COVID-19 recall was
approximately 40.86%, and tuberculosis recall was approximately
47.58%. A Normal prediction must not be interpreted as ruling
out disease.
""")


# ============================================================
# LOAD CNN
# ============================================================

try:

    model = load_cnn()

except Exception as exc:

    st.error(
        "Could not load the four-class CNN model."
    )

    st.exception(exc)

    st.caption(
        f"Expected model path: {MODEL_PATH}"
    )

    st.stop()


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
    # GRAD-CAM SETTINGS
    # ========================================================

    with st.expander("Grad-CAM v2 settings"):

        selected_layer = st.selectbox(
            "Feature layer",
            options=[
                "activation_3",
                "conv2d_3",
            ],
            index=0,
            help=(
                "Changes the visualization layer, "
                "not the trained model or weights."
            ),
        )

        overlay_alpha = st.slider(
            "Overlay opacity",
            min_value=0.10,
            max_value=0.80,
            value=0.45,
            step=0.05,
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
                    display_image,
                )


            # ------------------------------------------------
            # CNN PREDICTION
            # ------------------------------------------------

            with st.spinner(
                "Running the four-class CNN..."
            ):

                predictions = get_class_predictions(
                    model,
                    input_tensor,
                )

            if not predictions:

                raise RuntimeError(
                    "The model did not return predictions."
                )

            selected = predictions[0]

            selected_label = selected["label"]

            selected_score = selected["probability"]

            selected_index = selected["index"]


            # ------------------------------------------------
            # SECTION 1: CNN PREDICTION
            # ------------------------------------------------

            st.divider()

            st.header(
                "1. CNN Prediction"
            )

            st.metric(
                label="Highest-scoring class",
                value=selected_label,
                delta=(
                    f"Model score: "
                    f"{selected_score * 100:.2f}%"
                ),
            )

            st.caption(
                "This is the model's output score. It is not "
                "necessarily a calibrated probability and does "
                "not establish whether disease is present."
            )

            st.subheader(
                "Scores for all classes"
            )

            score_col1, score_col2 = st.columns(2)

            for i, item in enumerate(predictions):

                target_col = (
                    score_col1
                    if i % 2 == 0
                    else score_col2
                )

                with target_col:

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


            # ------------------------------------------------
            # SECTION 2: GRAD-CAM V2
            # ------------------------------------------------

            st.divider()

            st.header(
                "2. Grad-CAM v2 Visual Explanation"
            )

            with st.spinner(
                "Generating the Grad-CAM v2 overlay..."
            ):

                overlay = generate_gradcam_v2(
                    input_tensor=input_tensor,
                    original_image=original_image,
                    class_index=selected_index,
                    layer_name=selected_layer,
                    alpha=overlay_alpha,
                )


            # Exactly two images: original and overlay.

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
                f"Feature layer: {selected_layer}"
            )

            st.info(
                "Grad-CAM highlights image regions that influence "
                "the selected model output. It does not prove that "
                "a disease exists and may highlight irrelevant "
                "image features."
            )


            # ------------------------------------------------
            # SECTION 3: SHAP EXPLANATION
            # ------------------------------------------------

            st.divider()

            st.header(
                "3. SHAP Visual Explanation"
            )

            try:

                with st.spinner(
                    "Generating SHAP explanation. "
                    "This may take some time..."
                ):

                    # Load cached SHAP explainer.

                    shap_explainer = load_shap_explainer()

                    # Explain the CNN's selected class.

                    shap_explanation, shap_values = explain_image(
                        explainer=shap_explainer,
                        image=original_image,
                        class_index=selected_index,
                        max_evals=SHAP_MAX_EVALS,
                        batch_size=SHAP_BATCH_SIZE,
                    )


                # Generate SHAP visualization.

                shap_fig = generate_shap_visualization(
                    original_image=original_image,
                    class_values=shap_values,
                    class_label=selected_label,
                )

                st.pyplot(
                    shap_fig,
                    use_container_width=True,
                )

                plt.close(shap_fig)

                st.caption(
                    f"Explained class: {selected_label} | "
                    f"SHAP evaluations: {SHAP_MAX_EVALS}"
                )

                st.info(
                    "SHAP estimates how image regions contribute "
                    "to the selected CNN output relative to the "
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
                    "CNN prediction and Grad-CAM results are still "
                    "available. You can continue to the Gemini explanation."
                )


            # ------------------------------------------------
            # SECTION 4: GEMINI EXPLANATION
            # ------------------------------------------------

            st.divider()

            st.header(
                "4. Gemini-Generated Explanation"
            )

            try:

                with st.spinner(
                    "Generating an educational explanation..."
                ):

                    explanation = generate_explanation(
                        predictions=predictions,
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
                    "Your CNN prediction, Grad-CAM and SHAP results "
                    "are still available."
                )

                st.caption(str(gemini_error))

            st.caption(
                "The language model explains the supplied model "
                "outputs and Grad-CAM metadata for educational "
                "purposes; it does not independently confirm a "
                "medical diagnosis."
            )


        except Exception as exc:

            st.error(
                "Analysis could not be completed."
            )

            st.exception(
                exc
            )
