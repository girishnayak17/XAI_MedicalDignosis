import streamlit as st
from src.ui.core import require_analysis, generate_explanation

st.title("✨ Gemini AI Explanation")
if not require_analysis():
    st.info("Run an X-ray analysis first.")
    st.page_link("pages/xray_analysis.py", label="Go to X-Ray Analysis", icon="🩻")
    st.stop()

best = st.session_state.best_prediction
predictions = st.session_state.all_predictions[best["model"]]
layer = st.session_state.get("gradcam_layer") or "Not generated"
st.write(f"Selected model: **{best['model']}** · Selected class: **{best['label']}**")
st.caption("Gemini receives prediction metadata and the selected Grad-CAM layer name. It does not receive or visually inspect the uploaded X-ray or heatmap.")
if st.button("Generate educational explanation", type="primary"):
    try:
        with st.spinner("Generating explanation..."):
            result = generate_explanation(
                predictions=predictions,
                selected_label=best["label"],
                gradcam_layer=layer,
            )
            st.session_state.gemini_explanation = result
    except Exception as exc:
        st.error(f"Gemini explanation is unavailable: {exc}")
        st.exception(exc)

if st.session_state.get("gemini_explanation"):
    st.markdown(st.session_state.gemini_explanation)
else:
    st.info("Click the button to generate an explanation from the selected model's prediction metadata.")
