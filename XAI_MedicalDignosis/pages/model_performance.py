import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Model Performance")
st.write("Recorded held-out evaluation metrics for the three model architectures.")

# Values are the recorded results supplied during project development.
# These were produced in different evaluation runs and are not a fair
# head-to-head comparison until re-evaluated on identical patient-level data.
metrics = {
    "Custom CNN": {"Accuracy": 72.00, "Macro Precision": 71.00, "Macro Recall": 67.03, "Macro F1": 66.17, "Weighted F1": 70.95},
    "ResNet50": {"Accuracy": 99.28, "Macro Precision": 99.27, "Macro Recall": 99.40, "Macro F1": 99.33, "ROC-AUC": 99.98},
    "EfficientNetB0": {"Accuracy": 99.56, "Macro Precision": 99.55, "Macro Recall": 99.63, "Macro F1": 99.59, "ROC-AUC": 99.99},
}

st.warning("These recorded metrics came from different evaluation runs/test splits. Do not interpret this chart as a definitive head-to-head ranking. Evaluate all models on the same patient-level held-out test set for a fair comparison.")

for model_name, values in metrics.items():
    st.subheader(model_name)
    cols = st.columns(len(values))
    for col, (metric, value) in zip(cols, values.items()):
        col.metric(metric, f"{value:.2f}%")
    df = pd.DataFrame({"Metric": list(values.keys()), "Score (%)": list(values.values())})
    st.bar_chart(df.set_index("Metric"))
    st.divider()

st.subheader("Available class-wise recall details")
recall_data = pd.DataFrame([
    {"Model": "Custom CNN", "Class": "COVID-19", "Recall (%)": 40.86},
    {"Model": "Custom CNN", "Class": "Tuberculosis", "Recall (%)": 47.58},
    {"Model": "ResNet50", "Class": "Normal", "Recall (%)": 99.70},
    {"Model": "ResNet50", "Class": "Pneumonia", "Recall (%)": 99.77},
    {"Model": "ResNet50", "Class": "Tuberculosis", "Recall (%)": 98.94},
    {"Model": "ResNet50", "Class": "COVID-19", "Recall (%)": 99.17},
    {"Model": "EfficientNetB0", "Class": "Normal", "Recall (%)": 100.00},
    {"Model": "EfficientNetB0", "Class": "Pneumonia", "Recall (%)": 100.00},
    {"Model": "EfficientNetB0", "Class": "Tuberculosis", "Recall (%)": 98.79},
    {"Model": "EfficientNetB0", "Class": "COVID-19", "Recall (%)": 99.72},
])
st.dataframe(recall_data, use_container_width=True, hide_index=True)
for model_name in recall_data["Model"].unique():
    st.markdown(f"**{model_name} — class-wise recall**")
    subset = recall_data[recall_data["Model"] == model_name]
    st.bar_chart(subset.set_index("Class")["Recall (%)"])

st.info("Confusion matrices and ROC curves are intentionally not fabricated. Add the actual per-sample test predictions/probabilities from each model to generate those plots accurately.")
