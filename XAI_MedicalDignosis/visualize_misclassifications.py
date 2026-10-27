
import csv
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

CSV_PATH = "results/original_misclassifications.csv"
OUTPUT_PATH = "results/original_misclassified_samples.png"

# Focus on the Normal <-> Tuberculosis errors.
selected = []

with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        actual = row["actual_class"]
        predicted = row["predicted_class"]

        if (
            (actual == "normal" and predicted == "tuberculosis")
            or
            (actual == "tuberculosis" and predicted == "normal")
        ):
            selected.append(row)

random.seed(42)
random.shuffle(selected)
selected = selected[:12]

if not selected:
    raise RuntimeError("No matching misclassified images found.")

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for ax, row in zip(axes, selected):
    image_path = row["image_path"]
    image = Image.open(image_path).convert("RGB")

    ax.imshow(image, cmap="gray")
    ax.set_title(
        f"Actual: {row['actual_class']}\n"
        f"Predicted: {row['predicted_class']}\n"
        f"Scores: N={float(row['normal_score']):.2f}, "
        f"P={float(row['pneumonia_score']):.2f}, "
        f"T={float(row['tuberculosis_score']):.2f}",
        fontsize=8,
    )
    ax.axis("off")

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print("Images selected:", len(selected))
print("Saved visualization:", OUTPUT_PATH)
