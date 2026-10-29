
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models/kaggle_chest_xray_covid4_best.keras"
IMAGE_PATH = Path(
    "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
    "pneumonia/00222ba2d6b03db1_pneumonia-383.jpg"
)

CLASSES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
original = np.asarray(image, dtype=np.float32)

marker_covered = original.copy()
marker_covered[:45, :45] = np.median(
    original[45:90, :45], axis=(0, 1)
)

both_covered = marker_covered.copy()
both_covered[205:224, :, :] = np.median(
    original[185:205, :, :], axis=(0, 1)
)

def predict(image_array):
    return model(
        image_array[np.newaxis, ...], training=False
    ).numpy()[0]

tests = {
    "Original": original,
    "Marker covered": marker_covered,
    "Marker and bottom covered": both_covered,
}

print(f"\n{'Test':30} " + " ".join(f"{c:>13}" for c in CLASSES))

for name, image_array in tests.items():
    scores = predict(image_array)
    print(f"{name:30} " + " ".join(f"{s:13.6f}" for s in scores))

output_dir = BASE / "results/gradcam_v2_marker_test"
output_dir.mkdir(parents=True, exist_ok=True)

Image.fromarray(original.astype(np.uint8)).save(
    output_dir / "original.png"
)
Image.fromarray(marker_covered.astype(np.uint8)).save(
    output_dir / "marker_covered.png"
)
Image.fromarray(both_covered.astype(np.uint8)).save(
    output_dir / "marker_and_bottom_covered.png"
)

print("\nSaved test images to:", output_dir)
