"""
Explainable Chest X-Ray Classification
Four-Class CNN + Grad-CAM v2 + Gemini LLM
Academic research prototype; not a medical diagnostic tool.
"""
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.explainability.gradcam_v2 import enhance_heatmap, create_overlay
from src.explainability.llm_explainer import generate_explanation

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "kaggle_chest_xray_covid4_best.keras"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
DEFAULT_GRADCAM_LAYER = "activation_3"

st.set_page_config(page_title="XAI Chest X-Ray", page_icon="🩻", layout="wide")


@st.cache_resource
def load_cnn():
    """Load and validate the existing trained four-class CNN."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_shape = model.output_shape
    if isinstance(output_shape, list):
        raise ValueError("Expected one model output, but multiple outputs were found.")
    if output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} output classes, "
            f"but the model returns {output_shape[-1]}."
        )
    return model


def preprocess_image(pil_image):
    """Keep pixels in 0-255 because the model has Rescaling(1/255)."""
    resized = pil_image.convert("RGB").resize(
        IMAGE_SIZE, Image.Resampling.BILINEAR
    )
    image_array = np.asarray(resized, dtype=np.float32)
    return image_array, image_array[None, ...]


def get_class_predictions(model, input_tensor):
    """Return sorted output scores for the four configured classes."""
    probabilities = np.asarray(model(input_tensor, training=False).numpy())[0]
    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError("Prediction output does not match configured classes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The model returned invalid prediction scores.")
    predictions = [
        {"index": i, "label": name, "probability": float(probabilities[i])}
        for i, name in enumerate(CLASS_NAMES)
    ]
    predictions.sort(key=lambda item: item["probability"], reverse=True)
    return predictions


@st.cache_resource
def load_gradcam_model(layer_name):
    """Expose the selected spatial feature map and original model predictions."""
    model = load_cnn()
    try:
        feature_layer = model.get_layer(layer_name)
    except ValueError as exc:
        raise ValueError(f"Grad-CAM layer '{layer_name}' was not found.") from exc
    if len(feature_layer.output.shape) != 4:
        raise ValueError(f"Layer '{layer_name}' does not produce a spatial feature map.")
    return tf.keras.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output],
        name=f"gradcam_v2_{layer_name}",
    )


def generate_gradcam_v2(
    input_tensor, original_image, class_index,
    layer_name=DEFAULT_GRADCAM_LAYER, alpha=0.45,
):
    """Calculate Grad-CAM internally and return one RGB overlay only."""
    grad_model = load_gradcam_model(layer_name)
    image_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(image_tensor, training=False)
        target_score = predictions[:, int(class_index)]
    gradients = tape.gradient(target_score, feature_maps)
    if gradients is None:
        raise RuntimeError(f"Grad-CAM gradients unavailable for layer '{layer_name}'.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.reduce_sum(
        feature_maps[0] * channel_weights[0][None, None, :], axis=-1
    )
    cam = tf.nn.relu(cam).numpy().astype(np.float32)
    cam = cv2.resize(
        cam, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_CUBIC
    )
    cam = np.maximum(cam, 0.0)
    maximum = float(cam.max())
    cam = cam / maximum if maximum > 1e-8 else np.zeros_like(cam)
    enhanced_heatmap = enhance_heatmap(np.clip(cam, 0.0, 1.0))

    overlay_result = create_overlay(
        np.asarray(original_image, dtype=np.uint8),
        enhanced_heatmap,
        alpha=alpha,
    )
    # Support either return convention: overlay or (overlay, colored_heatmap).
    overlay = overlay_result[0] if isinstance(overlay_result, tuple) else overlay_result
    overlay = np.asarray(overlay)
    if overlay.ndim != 3 or overlay.shape[-1] != 3:
        raise ValueError(
            "create_overlay() must return an RGB overlay, or a tuple "
            "whose first item is the RGB overlay."
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


st.title("🩻 Explainable AI Chest X-Ray Analysis")
st.write(
    "Four-class chest X-ray classification using a CNN, Grad-CAM v2 "
    "visual explanations, and a Gemini-generated educational explanation."
)
st.warning(
    "Research and educational prototype only. This model is not clinically "
    "validated and must not be used to diagnose disease or make treatment "
    "decisions. Model scores are not confirmed diagnoses."
)

with st.expander("About this model"):
    st.markdown("""
This application uses a CNN for four chest X-ray image classes:
- **Normal**
- **Pneumonia**
- **Tuberculosis**
- **COVID-19**

Grad-CAM v2 highlights image regions that influence a selected model output.
These visualizations are not proof of disease. The model may produce incorrect
predictions due to image quality, dataset differences, class imbalance, and
other factors.
""")

with st.expander("Model evaluation results"):
    st.caption(
        "Recorded held-out test-set results from the existing evaluation; "
        "they do not guarantee performance on new patients or hospitals."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", "72.00%")
    c2.metric("Macro precision", "71.00%")
    c3.metric("Macro recall", "67.03%")
    c4, c5 = st.columns(2)
    c4.metric("Macro F1-score", "66.17%")
    c5.metric("Weighted F1-score", "70.95%")
    st.markdown("""
**Important limitation:** During evaluation, COVID-19 recall was approximately
40.86%, and tuberculosis recall was approximately 47.58%. A Normal prediction
must not be interpreted as ruling out disease.
""")

try:
    model = load_cnn()
except Exception as exc:
    st.error("Could not load the four-class CNN model.")
    st.exception(exc)
    st.caption(f"Expected model path: {MODEL_PATH}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a chest X-ray", type=["png", "jpg", "jpeg"],
    help="Upload a chest X-ray image in PNG or JPEG format.",
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as source:
            display_image = source.convert("RGB").copy()
    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")
        st.stop()

    st.subheader("Uploaded X-ray")
    st.image(display_image, caption=uploaded_file.name, use_container_width=True)

    with st.expander("Grad-CAM v2 settings"):
        selected_layer = st.selectbox(
            "Feature layer", options=["activation_3", "conv2d_3"], index=0,
            help="Changes the visualization layer, not the trained model or weights.",
        )
        overlay_alpha = st.slider(
            "Overlay opacity", min_value=0.10, max_value=0.80,
            value=0.45, step=0.05,
        )

    if st.button("Analyze X-ray", type="primary"):
        try:
            with st.spinner("Preparing the chest X-ray..."):
                original_image, input_tensor = preprocess_image(display_image)

            with st.spinner("Running the four-class CNN..."):
                predictions = get_class_predictions(model, input_tensor)
            if not predictions:
                raise RuntimeError("The model did not return predictions.")

            selected = predictions[0]
            selected_label = selected["label"]
            selected_score = selected["probability"]
            selected_index = selected["index"]

            st.divider()
            st.header("1. CNN Prediction")
            st.metric(
                label="Highest-scoring class", value=selected_label,
                delta=f"Model score: {selected_score * 100:.2f}%",
            )
            st.caption(
                "This is the model's output score. It is not necessarily a "
                "calibrated probability and does not establish whether disease is present."
            )
            st.subheader("Scores for all classes")
            score_col1, score_col2 = st.columns(2)
            for i, item in enumerate(predictions):
                target_col = score_col1 if i % 2 == 0 else score_col2
                with target_col:
                    st.write(f"**{item['label']}:** {item['probability'] * 100:.2f}%")
                    st.progress(float(np.clip(item["probability"], 0.0, 1.0)))

            st.divider()
            st.header("2. Grad-CAM v2 Visual Explanation")
            with st.spinner("Generating the Grad-CAM v2 overlay..."):
                overlay = generate_gradcam_v2(
                    input_tensor=input_tensor,
                    original_image=original_image,
                    class_index=selected_index,
                    layer_name=selected_layer,
                    alpha=overlay_alpha,
                )

            # Exactly two images are displayed here: original and overlay.
            image_col1, image_col2 = st.columns(2)
            with image_col1:
                st.subheader("Original X-ray")
                st.image(original_image.astype(np.uint8), use_container_width=True)
            with image_col2:
                st.subheader("Grad-CAM v2 Overlay")
                st.image(overlay, use_container_width=True)

            st.caption(f"Explained class: {selected_label} | Feature layer: {selected_layer}")
            st.info(
                "Grad-CAM highlights image regions that influence the selected "
                "model output. It does not prove that a disease exists and may "
                "highlight irrelevant image features."
            )

            st.divider()
            st.header("3. Gemini-Generated Explanation")
            with st.spinner("Generating an educational explanation..."):
                explanation = generate_explanation(
                    predictions=predictions,
                    selected_label=selected_label,
                    gradcam_layer=selected_layer,
                )
            if explanation:
                st.markdown(explanation)
            else:
                st.info("No explanation was returned. Check your Gemini configuration and API key.")
            st.caption(
                "The language model explains the supplied model outputs and "
                "Grad-CAM metadata for educational purposes; it does not "
                "independently confirm a medical diagnosis."
            )

        except Exception as exc:
            st.error("Analysis could not be completed.")
            st.exception(exc)
