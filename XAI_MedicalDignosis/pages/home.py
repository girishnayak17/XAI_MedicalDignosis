import streamlit as st
from src.ui.core import CLASS_NAMES, MODEL_PATHS, load_available_models

st.title("Welcome to XAI ChestVision")
st.write("An academic research prototype for four-class chest X-ray classification with model explanations.")
st.warning("Research and educational use only. This application is not clinically validated and must not be used to diagnose disease or make treatment decisions.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Supported classes", "4")
c2.metric("Model architectures", "3")
c3.metric("Input size", "224 × 224")
c4.metric("Explanation methods", "2")

st.subheader("Supported chest X-ray classes")
cols = st.columns(4)
for col, name in zip(cols, CLASS_NAMES):
    with col:
        st.markdown(f"### {name}")

st.subheader("Available models")
for name, path in MODEL_PATHS.items():
    st.write(f"**{name}** — {'Checkpoint found' if path.is_file() else 'Checkpoint not found'}")

st.subheader("Application workflow")
st.markdown("1. Upload a chest X-ray in **X-Ray Analysis**.\n2. Run the available classifiers and review the highest-scoring model/class pair.\n3. Explore the dataset through **EDA** and evaluation graphs through **Model Performance**.\n4. Inspect Grad-CAM and SHAP explanations.\n5. Read the educational explanation generated from prediction metadata.")
st.info("Use the top navigation bar to move between application pages. Your current analysis is shared between pages during the session.")
