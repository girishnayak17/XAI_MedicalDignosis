import streamlit as st
from src.ui.core import require_analysis, load_shap_explainer, explain_image, generate_shap_visualization, SHAP_MAX_EVALS, SHAP_BATCH_SIZE

st.title("🧠 SHAP Explanation")
if not require_analysis():
    st.info("Run an X-ray analysis first.")
    st.page_link("pages/xray_analysis.py", label="Go to X-Ray Analysis", icon="🩻")
    st.stop()

best = st.session_state.best_prediction
st.write(f"Explaining **{best['label']}** from **{best['model']}**.")
max_evals = st.select_slider("SHAP evaluation budget", options=[500, 1000, 2000], value=1000)
if st.button("Generate SHAP explanation", type="primary"):
    try:
        with st.spinner("Computing image attributions. This may take some time..."):
            explainer = load_shap_explainer(best["model"])
            explanation, class_values = explain_image(
                explainer=explainer,
                image=st.session_state.xray_original,
                class_index=best["index"],
                max_evals=max_evals,
                batch_size=SHAP_BATCH_SIZE,
            )
            fig = generate_shap_visualization(
                original_image=st.session_state.xray_original,
                class_values=class_values,
                class_label=best["label"],
                model_name=best["model"],
            )
            st.session_state.shap_figure = fig
            st.session_state.shap_class_label = best["label"]
            st.session_state.shap_model_name = best["model"]
        st.success("SHAP explanation generated.")
    except Exception as exc:
        st.error(f"SHAP explanation could not be generated: {exc}")
        st.exception(exc)

if st.session_state.get("shap_figure") is not None:
    st.pyplot(st.session_state.shap_figure, use_container_width=True)
    st.caption(f"Class: {st.session_state.shap_class_label} · Model: {st.session_state.shap_model_name}")
st.info("SHAP estimates how image regions contribute to the selected model output relative to its image-masking baseline. It does not establish disease presence or severity.")
