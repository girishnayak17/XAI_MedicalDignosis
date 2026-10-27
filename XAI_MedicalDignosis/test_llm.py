from src.explainability.llm_explainer import generate_explanation

sample_predictions = [
    {"label": "Infiltration", "probability": 0.2189},
    {"label": "Atelectasis", "probability": 0.1589},
    {"label": "Effusion", "probability": 0.1148},
    {"label": "Nodule", "probability": 0.0977},
    {"label": "Mass", "probability": 0.0847},
]

result = generate_explanation(
    predictions=sample_predictions,
    selected_label="Infiltration",
    gradcam_layer="conv2d_3",
)

print("\n--- LLM EXPLANATION ---\n")
print(result)