
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "kaggle_chest_xray_covid4_best.keras"

IMAGE_PATH = Path(
    "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
    "normal/00270b00cbbac310_normal-724.jpg"
)

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]


def load_input(path):
    with Image.open(path) as image:
        image = image.convert("RGB").resize(
            IMAGE_SIZE, Image.Resampling.BILINEAR
        )
        return np.asarray(image, dtype=np.float32)


def predict(model, image):
    return model(image[None, ...], training=False).numpy()[0]


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)
    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU(s):", gpus)

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    image = load_input(IMAGE_PATH)

    baseline = predict(model, image)
    target_class = int(np.argmax(baseline))

    print("\nBaseline prediction:")
    for name, score in zip(CLASS_NAMES, baseline):
        print(f"{name:15s}: {score * 100:.2f}%")

    print(f"\nTesting influence on: {CLASS_NAMES[target_class]}")
    print("Each test replaces a region with the image's mean pixel value.")

    h, w = IMAGE_SIZE
    patch = 32
    stride = 32

    # Test border regions and central regions separately.
    regions = {
        "top_border": (0, 0, w, patch),
        "bottom_border": (0, h - patch, w, h),
        "left_border": (0, 0, patch, h),
        "right_border": (w - patch, 0, w, h),
        "center": (w // 2 - patch // 2, h // 2 - patch // 2,
                   w // 2 + patch // 2, h // 2 + patch // 2),
    }

    mean_pixel = image.mean(axis=(0, 1), keepdims=True)[0, 0]
    results = []

    # Single-region occlusions.
    for name, (x1, y1, x2, y2) in regions.items():
        masked = image.copy()
        masked[y1:y2, x1:x2] = mean_pixel
        scores = predict(model, masked)
        delta = float(scores[target_class] - baseline[target_class])
        results.append((name, delta, scores))

    # Sliding patches: record the largest score reductions.
    patch_results = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            masked = image.copy()
            masked[y:y + patch, x:x + patch] = mean_pixel
            scores = predict(model, masked)
            delta = float(scores[target_class] - baseline[target_class])
            patch_results.append((delta, x, y, scores))

    print("\nRegional occlusion results:")
    print("Positive change means the score decreased after masking.")
    for name, delta, scores in results:
        print(
            f"{name:15s} score change: {delta:+.4f} "
            f"({delta * 100:+.2f} percentage points)"
        )

    print("\nLargest score decreases from individual patches:")
    for delta, x, y, scores in sorted(patch_results)[:8]:
        print(
            f"x={x:3d}, y={y:3d}, "
            f"score change={delta:+.4f}, "
            f"new prediction={CLASS_NAMES[int(np.argmax(scores))]}"
        )

    output_dir = PROJECT_DIR / "results" / "gradcam_v2" / "occlusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "occlusion_results.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"Image: {IMAGE_PATH}\n")
        f.write(f"Baseline scores: {baseline.tolist()}\n")
        f.write(f"Target class: {CLASS_NAMES[target_class]}\n\n")
        for name, delta, scores in results:
            f.write(f"{name}: change={delta:+.6f}; scores={scores.tolist()}\n")
        f.write("\nLargest patch score decreases:\n")
        for delta, x, y, scores in sorted(patch_results)[:8]:
            f.write(
                f"x={x}, y={y}, change={delta:+.6f}, "
                f"scores={scores.tolist()}\n"
            )

    print("\nReport saved to:", report_path)


if __name__ == "__main__":
    main()
