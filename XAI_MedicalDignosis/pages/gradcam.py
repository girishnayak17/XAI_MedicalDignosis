import streamlit as st
from src.ui.core import (
    require_analysis, get_gradcam_layer_options, generate_gradcam_v2,
    CNN_GRADCAM_LAYER, RESNET_GRADCAM_LAYER, EFFICIENTNET_GRADCAM_LAYER,
)

st.title("🔍 Grad-CAM Explanation")
if not require_analysis():
    st.info("Run an X-ray analysis first.")
    st.page_link("pages/xray_analysis.py", label="Go to X-Ray Analysis", icon="🩻")
    st.stop()

best = st.session_state.best_prediction
st.write(f"Explaining **{best['label']}** from **{best['model']}**.")
try:
    layers = get_gradcam_layer_options(best["model"])
    if not layers:
        st.error("No spatial Grad-CAM layers were found in this checkpoint.")
        st.stop()
    preferred = {
        "Custom CNN": CNN_GRADCAM_LAYER,
        "ResNet50": RESNET_GRADCAM_LAYER,
        "EfficientNetB0": EFFICIENTNET_GRADCAM_LAYER,
    }.get(best["model"])
    default = preferred if preferred in layers else layers[-1]
    layer = st.selectbox("Feature layer", layers, index=layers.index(default))
    alpha = st.slider("Overlay opacity", 0.10, 0.80, 0.45, 0.05)
    if st.button("Generate Grad-CAM", type="primary"):
        with st.spinner(f"Generating Grad-CAM using {best['model']}..."):
            overlay = generate_gradcam_v2(
                input_tensor=st.session_state.xray_input_tensor,
                original_image=st.session_state.xray_original,
                class_index=best["index"],
                model_name=best["model"],
                layer_name=layer,
                alpha=alpha,
            )
            st.session_state.gradcam_overlay = overlay
            st.session_state.gradcam_layer = layer
    if st.session_state.gradcam_overlay is not None:
        left, right = st.columns(2)
        left.image(st.session_state.xray_original.astype("uint8"), caption="Original X-ray", use_container_width=True)
        right.image(st.session_state.gradcam_overlay, caption="Grad-CAM overlay", use_container_width=True)
        st.caption(f"Class: {best['label']} · Model: {best['model']} · Layer: {st.session_state.gradcam_layer}")
    st.info("Grad-CAM highlights regions that influence the selected model output. Highlighted regions are not proof of disease.")
except Exception as exc:
    st.error(f"Grad-CAM could not be generated: {exc}")
    st.exception(exc)
