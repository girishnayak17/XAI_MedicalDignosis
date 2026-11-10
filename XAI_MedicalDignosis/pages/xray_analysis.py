import streamlit as st
from PIL import Image
from src.ui.core import (
    CLASS_NAMES, IMAGE_SIZE, load_available_models, preprocess_image,
    get_all_model_predictions, get_best_prediction, initialize_session_state,
)

initialize_session_state()
st.title("🩻 X-Ray Analysis")
st.write("Upload one chest X-ray. Every model checkpoint available in the project will be evaluated.")

uploaded = st.file_uploader("Upload chest X-ray", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption=uploaded.name, width=420)
        if st.button("Analyze X-ray", type="primary"):
            with st.spinner("Loading available models and analyzing image..."):
                original, tensor = preprocess_image(image)
                models, errors = load_available_models()
                if not models:
                    st.error("No model checkpoints could be loaded.")
                    for name, error in errors.items():
                        st.write(f"**{name}:** {error}")
                    st.stop()
                predictions = get_all_model_predictions(tensor, models)
                best = get_best_prediction(predictions)
                st.session_state.xray_original = original
                st.session_state.xray_input_tensor = tensor
                st.session_state.xray_filename = uploaded.name
                st.session_state.all_predictions = predictions
                st.session_state.best_prediction = best
                st.session_state.gradcam_overlay = None
                st.session_state.gradcam_layer = None
                st.session_state.shap_figure = None
                st.session_state.gemini_explanation = None
            st.success("Analysis completed. Open the Grad-CAM, SHAP, and AI Explanation pages for further analysis.")
            if errors:
                with st.expander("Model loading warnings"):
                    for name, error in errors.items():
                        st.warning(f"{name}: {error}")
    except Exception as exc:
        st.error(f"Could not process the uploaded image: {exc}")

if st.session_state.all_predictions and st.session_state.best_prediction:
    best = st.session_state.best_prediction
    st.divider()
    st.subheader("Highest-scoring model/class pair")
    a, b, c = st.columns(3)
    a.metric("Selected class", best["label"])
    b.metric("Selected model", best["model"])
    c.metric("Individual model score", f"{best['probability'] * 100:.2f}%")
    st.caption("Scores from different architectures may not be calibrated or directly comparable. The highest score is a selection rule, not a validated ensemble or diagnosis.")
    st.subheader("All model class scores")
    columns = st.columns(max(1, len(st.session_state.all_predictions)))
    for (model_name, rows), col in zip(st.session_state.all_predictions.items(), columns):
        with col:
            st.markdown(f"**{model_name}**")
            for row in sorted(rows, key=lambda x: x["index"]):
                st.write(f"{row['label']}: {row['probability'] * 100:.2f}%")
                st.progress(float(max(0, min(1, row["probability"]))))
else:
    st.info("No analysis is stored in this session yet. Upload an X-ray and click Analyze X-ray.")
