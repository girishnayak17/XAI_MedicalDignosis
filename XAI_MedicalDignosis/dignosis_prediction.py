
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "Datasets"
MODEL_PATH = BASE_DIR / "models" / "custom_cnn_best.keras"

# CHANGE THIS to the exact image filename you tested.
FILENAME = "00000013_010.png"

LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
]

image_path = DATA_DIR / "images" / FILENAME
metadata_path = DATA_DIR / "Data_Entry_2017_v2020.csv"

if not image_path.exists():
    raise FileNotFoundError(f"Image not found: {image_path}")

metadata = pd.read_csv(metadata_path)
row = metadata.loc[metadata["Image Index"] == FILENAME]

if row.empty:
    raise ValueError(f"Filename not found in NIH metadata: {FILENAME}")

ground_truth = str(row.iloc[0]["Finding Labels"]).split("|")

print("\n=== NIH DATASET METADATA ===")
print("Image:", FILENAME)
print("Patient ID:", row.iloc[0]["Patient ID"])
print("Ground-truth labels:", ground_truth)

print("\n=== CHECK DATA SPLIT ===")
for list_name in ["train_val_list.txt", "test_list.txt"]:
    list_path = DATA_DIR / list_name
    if list_path.exists():
        filenames = set(
            list_path.read_text().splitlines()
        )
        print(f"{list_name}:",
              "FOUND" if FILENAME in filenames else "not listed")

print("\n=== LOAD MODEL ===")
model = tf.keras.models.load_model(MODEL_PATH)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of labels:", len(LABELS))

if model.output_shape[-1] != len(LABELS):
    raise ValueError("Model output count does not match label count.")

print("\n=== RUN PREDICTION ===")
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

array = np.asarray(image, dtype=np.float32) / 255.0
input_tensor = np.expand_dims(array, axis=0)

scores = model.predict(input_tensor, verbose=0)[0]

for index in np.argsort(scores)[::-1]:
    label = LABELS[index]
    marker = " <-- NIH LABEL" if label in ground_truth else ""
    print(f"{label:22s} {scores[index] * 100:7.2f}%{marker}")

print("\n=== END OF DIAGNOSTIC ===")
