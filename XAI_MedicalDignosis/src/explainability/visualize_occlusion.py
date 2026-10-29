
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR / "models" /
    "kaggle_chest_xray_covid4_best.keras"
)

IMAGE_PATH = Path(
    "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
    "normal/00270b00cbbac310_normal-724.jpg"
)

OUTPUT_DIR = (
    PROJECT_DIR / "results" /
    "gradcam_v2" / "occlusion"
)

IMAGE_SIZE = (224, 224)
PATCH_SIZE = 32
STRIDE = 16

CLASS_NAMES = [
    "Normal", "Pneumonia", "Tuberculosis", "COVID-19"
]


# ------------------------------------------------------------
# Load model and image
# ------------------------------------------------------------

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(MODEL_PATH)

    if not IMAGE_PATH.is_file():
        raise FileNotFoundError(IMAGE_PATH)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU(s):", gpus)

    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    with Image.open(IMAGE_PATH) as source:
        original = np.asarray(
            source.convert("RGB"), dtype=np.uint8
        )

    resized = cv2.resize(
        original,
        (IMAGE_SIZE[1], IMAGE_SIZE[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    image = resized.astype(np.float32)

    # The trained model performs its own 1/255 rescaling.
    baseline_scores = model(
        image[None, ...], training=False
    ).numpy()[0]

    target_class = int(np.argmax(baseline_scores))
    baseline_score = float(baseline_scores[target_class])

    print("\nBaseline prediction:")
    for name, score in zip(CLASS_NAMES, baseline_scores):
        print(f"{name:15s}: {score * 100:.2f}%")

    print(
        f"\nExplaining: {CLASS_NAMES[target_class]} "
        f"({baseline_score * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # Sliding-window occlusion sensitivity
    # --------------------------------------------------------

    height, width = IMAGE_SIZE
    mean_pixel = image.mean(axis=(0, 1))

    # Positive values mean masking reduced the target score.
    sensitivity_sum = np.zeros(
        (height, width), dtype=np.float32
    )
    sensitivity_count = np.zeros(
        (height, width), dtype=np.float32
    )

    patches = []

    for y in range(
        0, height - PATCH_SIZE + 1, STRIDE
    ):
        for x in range(
            0, width - PATCH_SIZE + 1, STRIDE
        ):
            masked = image.copy()

            masked[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] = mean_pixel

            patches.append((x, y, masked))

    # Batch predictions to reduce inference overhead.
    batch_size = 16

    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size]

        batch_images = np.stack(
            [item[2] for item in batch]
        )

        scores = model(
            batch_images, training=False
        ).numpy()

        for (x, y, _), prediction in zip(batch, scores):
            change = baseline_score - float(
                prediction[target_class]
            )

            sensitivity_sum[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += change

            sensitivity_count[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] += 1.0

    sensitivity = np.divide(
        sensitivity_sum,
        sensitivity_count,
        out=np.zeros_like(sensitivity_sum),
        where=sensitivity_count > 0,
    )

    # Positive = masking reduced score.
    # Negative = masking increased score.
    print("\nSensitivity map statistics:")
    print("Minimum:", float(sensitivity.min()))
    print("Maximum:", float(sensitivity.max()))

    # Normalize symmetrically for visualization.
    scale = float(np.percentile(
        np.abs(sensitivity), 99
    ))

    if scale > 1e-8:
        display_map = np.clip(
            sensitivity / scale, -1.0, 1.0
        )
    else:
        display_map = np.zeros_like(sensitivity)

    # Red: masking reduces target score.
    # Blue: masking increases target score.
    # Neutral: little observed change.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resized_original = cv2.resize(
        original,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5)
    )

    axes[0].imshow(resized_original)
    axes[0].set_title("Original X-ray")

    image_plot = axes[1].imshow(
        display_map,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    axes[1].set_title(
        "Occlusion Sensitivity\nRed: score decreases"
    )
    fig.colorbar(
        image_plot, ax=axes[1], fraction=0.046
    )

    axes[2].imshow(resized_original)
    axes[2].imshow(
        display_map,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        alpha=0.45,
    )
    axes[2].set_title("Sensitivity Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"Occlusion sensitivity for {CLASS_NAMES[target_class]} "
        f"| Baseline score: {baseline_score * 100:.1f}%"
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / "occlusion_sensitivity_comparison.png"
    )

    fig.savefig(
        str(output_path),
        format="png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    np.save(
        OUTPUT_DIR / "occlusion_sensitivity.npy",
        sensitivity,
    )

    print("\nSaved visualization:", output_path)
    print("Saved sensitivity values:",
          OUTPUT_DIR / "occlusion_sensitivity.npy")


if __name__ == "__main__":
    main()
