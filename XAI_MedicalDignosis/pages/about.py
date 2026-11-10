import streamlit as st

st.title("ℹ️ About & User Guide")
st.subheader("Project overview")
st.write("Explainable AI for four-class chest X-ray classification using a Custom CNN, ResNet50, and EfficientNetB0, with Grad-CAM and SHAP visual explanations.")
st.markdown("**Supported classes:** Normal, Pneumonia, Tuberculosis, and COVID-19.")
st.markdown("**Main technologies:** Python, TensorFlow/Keras, Streamlit, OpenCV, SHAP, Grad-CAM, and Gemini API integration.")

st.subheader("Quick user manual")
steps = [
    ("1. Home", "Review the project overview, supported classes, model availability, and workflow."),
    ("2. X-Ray Analysis", "Upload a PNG or JPEG chest X-ray and select Analyze X-ray. The app runs every model checkpoint it can load."),
    ("3. Review scores", "Compare all four class scores for each available model. The main result is the highest individual model/class score."),
    ("4. EDA", "Explore class distribution, sample X-ray images, image dimensions, and dataset split distribution when the dataset is configured."),
    ("5. Model Performance", "Review the recorded evaluation metrics and graphs for each model. Check the notes about test-set comparability."),
    ("6. Grad-CAM", "Choose an available spatial feature layer and generate a heatmap overlay for the selected model and class."),
    ("7. SHAP", "Choose an evaluation budget and generate class-specific image attributions. This may take time."),
    ("8. AI Explanation", "Generate an educational explanation based on prediction metadata. Gemini does not inspect the image."),
]
for title, body in steps:
    with st.expander(title):
        st.write(body)

st.subheader("Important limitations")
st.warning("This is an academic research prototype, not a clinically validated medical device. It must not be used for diagnosis, triage, or treatment decisions. Model scores can be wrong and may not generalize to other hospitals, devices, or patient populations.")
st.write("The application compares raw model scores using a highest-score selection rule. Scores from different architectures may not be calibrated or directly comparable. Fair model comparison requires the same patient-level held-out test set and leakage checks.")
