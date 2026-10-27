"""
Explainable Chest X-Ray Classification
Four-Class CNN + Grad-CAM + Gemini LLM

Classes:
    0. Normal
    1. Pneumonia
    2. Tuberculosis
    3. COVID-19

Academic research prototype.
Not a medical diagnostic tool.
"""

from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.explainability.gradcam import (
    generate_gradcam,
    create_gradcam_overlay,
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

# IMPORTANT:
# This order must match the output order used when training
# kaggle_chest_xray_covid4_best.keras.
CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XAI Chest X-Ray",
    page_icon="🩻",
    layout="wide",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_cnn():
    """Load and validate the trained four-class CNN."""

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
    Prepare an image for the four-class CNN.

    The model contains its own Rescaling(1/255) layer.
    Therefore, input pixels must remain in the 0-255 range
    to avoid normalizing them twice.
    """

    image = pil_image.convert("RGB")

    resized_image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    input_tensor = np.expand_dims(
        image_array,
        axis=0,
    )

    return image_array, input_tensor


# ============================================================
# PREDICTIONS
# ============================================================

def get_class_predictions(model, input_tensor):
    """Return sorted predictions using the four-class labels."""

    output = model.predict(
        input_tensor,
        verbose=0,
    )

    probabilities = np.asarray(output)[0]

    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(
            "Prediction output does not match the configured classes."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "The model returned invalid prediction scores."
        )

    predictions = []

    for index, score in enumerate(probabilities):
        predictions.append(
            {
                "index": index,
                "label": CLASS_NAMES[index],
                "probability": float(score),
            }
        )

    predictions.sort(
        key=lambda item: item["probability"],
        reverse=True,
    )

    return predictions


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🩻 Explainable AI Chest X-Ray Analysis")

st.write(
    "Explore four-class chest X-ray classification using a CNN, "
    "Grad-CAM visual explanations, and a Gemini-generated "
    "educational explanation."
)

st.warning(
    "Research and educational prototype only. This model is not "
    "clinically validated and must not be used to diagnose disease "
    "or make treatment decisions. Model scores are not confirmed "
    "diagnoses."
)

with st.expander("About this model"):
    st.markdown(
        """
        This application uses a convolutional neural network trained
        for four chest X-ray image classes:

        - **Normal**
        - **Pneumonia**
        - **Tuberculosis**
        - **COVID-19**

        Grad-CAM visualizes image regions that influence a selected
        model output. These visualizations are not proof of disease.

        The model may produce incorrect predictions. Dataset differences,
        image quality, class imbalance, and other factors may affect
        its results.
        """
    )

with st.expander("Model evaluation results"):
    st.caption(
        "The following results were obtained on the held-out test set "
        "during the four-class model evaluation. They are not a guarantee "
        "of performance on new patients or other hospitals."
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    metric_col1.metric("Test accuracy", "72.00%")
    metric_col2.metric("Macro precision", "71.00%")
    metric_col3.metric("Macro recall", "67.03%")

    metric_col4, metric_col5 = st.columns(2)

    metric_col4.metric("Macro F1-score", "66.17%")
    metric_col5.metric("Weighted F1-score", "70.95%")

    st.markdown(
        """
        **Important limitation:** During evaluation, COVID-19 recall
        was approximately 40.86%, and tuberculosis recall was
        approximately 47.58%. Some positive cases were classified
        as another class. A Normal prediction must not be interpreted
        as ruling out disease.
        """
    )


# ============================================================
# LOAD MODEL SAFELY
# ============================================================

try:
    model = load_cnn()

except Exception as exc:
    st.error("Could not load the four-class CNN model.")

    st.code(str(exc))

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
        display_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded X-ray")

        st.image(
            display_image,
            caption=uploaded_file.name,
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            f"Could not read the uploaded image: {exc}"
        )

        st.stop()

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if st.button(
        "Analyze X-ray",
        type="primary",
    ):

        temporary_path = None

        try:
            # ------------------------------------------------
            # SAVE TEMPORARY IMAGE
            # ------------------------------------------------

            suffix = Path(
                uploaded_file.name
            ).suffix.lower()

            if suffix not in {".png", ".jpg", ".jpeg"}:
                suffix = ".png"

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:

                temporary_file.write(
                    uploaded_file.getvalue()
                )

                temporary_path = temporary_file.name

            # ------------------------------------------------
            # PREPROCESS IMAGE
            # ------------------------------------------------

            with st.spinner(
                "Preparing the chest X-ray..."
            ):

                original_image, input_tensor = preprocess_image(
                    display_image
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

            selected_prediction = predictions[0]

            selected_label = selected_prediction["label"]
            selected_score = selected_prediction["probability"]
            selected_index = selected_prediction["index"]

            # ------------------------------------------------
            # SECTION 1: PREDICTION
            # ------------------------------------------------

            st.divider()
            st.header("1. CNN Prediction")

            st.metric(
                label="Highest-scoring class",
                value=selected_label,
                delta=f"Model score: {selected_score * 100:.2f}%",
            )

            st.caption(
                "This score comes from the model's softmax output. "
                "It is not necessarily a calibrated probability and "
                "does not establish whether a disease is present."
            )

            st.subheader("Scores for all classes")

            for item in predictions:

                label = item["label"]
                score = item["probability"]

                st.write(
                    f"**{label}:** {score * 100:.2f}%"
                )

                st.progress(
                    float(
                        max(
                            0.0,
                            min(1.0, score),
                        )
                    )
                )

            # ------------------------------------------------
            # SECTION 2: GRAD-CAM
            # ------------------------------------------------

            st.divider()
            st.header("2. Grad-CAM Visual Explanation")

            with st.spinner(
                "Generating the Grad-CAM heatmap..."
            ):

                (
                    heatmap,
                    gradcam_score,
                    conv_layer_name,
                ) = generate_gradcam(
                    model,
                    input_tensor,
                    selected_index,
                )

                overlay = create_gradcam_overlay(
                    original_image,
                    heatmap,
                    alpha=0.4,
                )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original X-ray")

                st.image(
                    original_image.astype(np.uint8),
                    use_container_width=True,
                )

            with col2:
                st.subheader("Grad-CAM overlay")

                st.image(
                    overlay,
                    use_container_width=True,
                )

            st.caption(
                f"Target class: {selected_label} | "
                f"Convolutional layer: {conv_layer_name}"
            )

            st.info(
                "Grad-CAM highlights regions that influenced the selected "
                "model output. These regions do not prove that a disease "
                "exists and cannot independently validate the prediction."
            )

            # ------------------------------------------------
            # SECTION 3: GEMINI EXPLANATION
            # ------------------------------------------------

            st.divider()
            st.header("3. Gemini-Generated Explanation")

            with st.spinner(
                "Generating an educational explanation..."
            ):

                explanation = generate_explanation(
                    predictions=predictions,
                    selected_label=selected_label,
                    gradcam_layer=conv_layer_name,
                )

            if explanation:
                st.markdown(explanation)

            else:
                st.info(
                    "No explanation was returned. Check your Gemini "
                    "configuration and API key."
                )

            st.caption(
                "The language model provides an educational explanation "
                "of the supplied model outputs and Grad-CAM metadata. "
                "It does not independently confirm a medical diagnosis."
            )

        # ====================================================
        # ERROR HANDLING
        # ====================================================

        except Exception as exc:

            st.error(
                "Analysis could not be completed."
            )

            st.exception(exc)

        # ====================================================
        # CLEANUP
        # ====================================================

        finally:

            if temporary_path is not None:

                try:
                    Path(
                        temporary_path
                    ).unlink(
                        missing_ok=True
                    )

                except OSError:
                    pass

