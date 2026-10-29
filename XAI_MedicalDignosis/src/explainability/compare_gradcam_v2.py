import os

# Keep GPU enabled.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path("/mnt/d/MCA_Final_Project/XAI_MedicalDignosis")
MODEL_PATH = PROJECT_DIR / "models" / "kaggle_chest_xray_covid4_best.keras"

# New folder preserves the previous output files.
OUTPUT_DIR = PROJECT_DIR / "results" / "gradcam_v2" / "lung_only_contrast_v1"

IMAGE_SIZE = 224
CONV_LAYER_NAME = "activation_3"
CLASSIFIER_NAME = "four_class_classifier"
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]

IMAGE_PATHS = {
    "Normal": (
        "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
        "normal/00270b00cbbac310_normal-724.jpg"
    ),
    "Pneumonia": (
        "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
        "pneumonia/00222ba2d6b03db1_pneumonia-383.jpg"
    ),
    "Tuberculosis": (
        "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
        "tuberculosis"
    ),
    "COVID-19": (
        "/mnt/d/MCA_Final_Project/covid19Dataset/"
        "COVID-19_Radiography_Dataset/COVID/images/COVID-1.png"
    ),
}


def create_lung_roi_mask(size=224):
    """Approximate display mask only; this is NOT anatomical segmentation."""
    mask = np.zeros((size, size), dtype=np.uint8)
    left_lung = np.array([
        [104, 48], [91, 47], [78, 52], [65, 62], [53, 78],
        [44, 98], [40, 121], [43, 145], [51, 166], [62, 183],
        [77, 192], [91, 190], [99, 177], [103, 155], [105, 130],
        [105, 104], [105, 78],
    ], dtype=np.int32)
    right_lung = np.array([
        [120, 48], [133, 47], [146, 52], [159, 62], [171, 78],
        [180, 98], [184, 121], [181, 145], [173, 166], [162, 183],
        [147, 192], [133, 190], [125, 177], [121, 155], [119, 130],
        [119, 104], [119, 78],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [left_lung, right_lung], 255)
    return mask > 0


def load_model_and_grad_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("TensorFlow cannot detect the GPU. Check WSL GPU setup.")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as error:
            print("GPU memory-growth setting:", error)

    print("TensorFlow version:", tf.__version__)
    print("Available GPU(s):", gpus)
    print("Loading existing trained model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    conv_layer = model.get_layer(CONV_LAYER_NAME)
    classifier = model.get_layer(CLASSIFIER_NAME)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, classifier.input, model.output],
        name="gradcam_intermediate_model",
    )
    print("Model loaded successfully.")
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Grad-CAM layer:", CONV_LAYER_NAME)
    return grad_model, classifier


def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    # The trained model already includes Rescaling(1/255).
    return image, image.astype(np.float32)[None, ...]


def compute_gradcam(input_array, grad_model, classifier):
    input_tensor = tf.convert_to_tensor(input_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_output, classifier_features, predictions = grad_model(
            input_tensor, training=False
        )
        # Reconstruct the Dense layer's pre-softmax logits.
        logits = tf.linalg.matmul(classifier_features, classifier.kernel)
        if classifier.use_bias:
            logits = tf.nn.bias_add(logits, classifier.bias)
        predicted_class = tf.argmax(predictions[0], output_type=tf.int32)
        target_score = tf.gather(logits[0], predicted_class)

    gradients = tape.gradient(target_score, conv_output)
    if gradients is None:
        raise RuntimeError("Gradients are unavailable. Check the selected convolution layer.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap_tensor = tf.reduce_sum(
        conv_output[0] * channel_weights[0][None, None, :], axis=-1
    )
    heatmap = tf.nn.relu(heatmap_tensor).numpy().astype(np.float32)
    # Resize before ROI normalization, so normalization matches the displayed map.
    heatmap = cv2.resize(
        heatmap, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    return heatmap, int(predicted_class.numpy()), predictions[0].numpy()


def normalize_inside_mask(heatmap, mask):
    """Robust contrast normalization using positive values inside the ROI."""
    values = heatmap[mask]
    positive = values[values > 0]
    normalized = np.zeros_like(heatmap, dtype=np.float32)
    if positive.size == 0:
        print("Warning: no positive Grad-CAM values inside the lung ROI.")
        return normalized

    upper = float(np.percentile(positive, 99))
    if upper <= 1e-12:
        upper = float(positive.max())
    if upper <= 1e-12:
        return normalized

    normalized[mask] = np.clip(heatmap[mask] / upper, 0.0, 1.0)
    # Display-only gamma correction enhances weaker activations.
    normalized[mask] = np.power(normalized[mask], 0.75)
    return normalized


def apply_color_map(heatmap):
    heatmap_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def save_outputs(label, image, raw_heatmap, predicted_class, probabilities, lung_mask):
    normalized = normalize_inside_mask(raw_heatmap, lung_mask)

    # Smooth the display only, to reduce pixelation.
    smooth = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1.2)
    smooth[~lung_mask] = 0.0
    smooth = np.clip(smooth, 0.0, 1.0)

    colored = apply_color_map(smooth)
    alpha = (0.58 * np.power(smooth, 0.65))[..., None]
    overlay = (
        image.astype(np.float32) * (1.0 - alpha)
        + colored.astype(np.float32) * alpha
    )
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay[~lung_mask] = image[~lung_mask]

    plt.imsave(OUTPUT_DIR / f"{label}_original.png", image)
    plt.imsave(
        OUTPUT_DIR / f"{label}_lung_only_heatmap.png",
        smooth, cmap="turbo", vmin=0, vmax=1
    )
    plt.imsave(OUTPUT_DIR / f"{label}_lung_only_overlay.png", overlay)
    plt.imsave(OUTPUT_DIR / f"{label}_mask_check.png", lung_mask, cmap="gray")

    # Preserve raw values for reproducibility and comparison.
    np.save(OUTPUT_DIR / f"{label}_raw_gradcam.npy", raw_heatmap)
    np.save(OUTPUT_DIR / f"{label}_normalized_lung_gradcam.npy", smooth)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original X-ray")
    axes[1].imshow(smooth, cmap="turbo", vmin=0, vmax=1)
    axes[1].set_title("Contrast-enhanced ROI heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Lung ROI overlay")
    axes[3].imshow(lung_mask, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Approximate ROI mask")
    fig.suptitle(
        f"Input label: {label} | Prediction: {CLASS_NAMES[predicted_class]} "
        f"({probabilities[predicted_class] * 100:.1f}%)",
        fontsize=12,
    )
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{label}_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grad_model, classifier = load_model_and_grad_model()
    lung_mask = create_lung_roi_mask(IMAGE_SIZE)
    plt.imsave(OUTPUT_DIR / "lung_roi_check.png", lung_mask, cmap="gray")

    processed = 0
    for label, image_path in IMAGE_PATHS.items():
        if not Path(image_path).is_file():
            print(f"Skipping {label}: image not found:\n{image_path}")
            continue
        print(f"\nProcessing: {label}")
        image, input_array = load_image(image_path)
        raw_heatmap, predicted_class, probabilities = compute_gradcam(
            input_array, grad_model, classifier
        )
        save_outputs(label, image, raw_heatmap, predicted_class, probabilities, lung_mask)
        print(
            f"Prediction: {CLASS_NAMES[predicted_class]} | "
            f"Score: {probabilities[predicted_class] * 100:.1f}%"
        )
        processed += 1

    print(f"\nFinished. Images processed: {processed}")
    print(f"Results saved in:\n{OUTPUT_DIR}")
    if processed == 0:
        print("No images processed. Check IMAGE_PATHS.")


if __name__ == "__main__":
    main()
