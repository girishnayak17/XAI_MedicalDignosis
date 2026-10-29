
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


# Four-class model configuration
CLASS_NAMES = ["Normal", "Pneumonia", "Tuberculosis", "COVID-19"]
IMAGE_SIZE = (224, 224)

# Use the feature map after batch normalization and activation.
# This is the last spatial feature map before pooling.
DEFAULT_LAYER = "activation_3"


def load_xray(image_path, image_size=IMAGE_SIZE):
    """
    Load an X-ray as RGB.
    Preserve pixel values in [0, 255] because the model has
    its own Rescaling(1./255) layer.
    """
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)

    resized = image.resize(
        image_size,
        Image.Resampling.BILINEAR,
    )
    input_array = np.asarray(resized, dtype=np.float32)
    input_tensor = np.expand_dims(input_array, axis=0)

    return original, input_tensor


def get_gradcam_model(model, layer_name=DEFAULT_LAYER):
    """
    Return a model that exposes both the selected feature map
    and the original model's predictions.
    """
    try:
        feature_layer = model.get_layer(layer_name)
    except ValueError as exc:
        available = [layer.name for layer in model.layers]
        raise ValueError(
            f"Layer '{layer_name}' was not found. "
            f"Available layers: {available}"
        ) from exc

    if len(feature_layer.output.shape) != 4:
        raise ValueError(
            f"{layer_name} must produce a spatial feature map; "
            f"got shape {feature_layer.output.shape}"
        )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[feature_layer.output, model.output],
    )

    return grad_model


def generate_gradcam(
    model,
    input_tensor,
    class_index=None,
    layer_name=DEFAULT_LAYER,
):
    """
    Standard Grad-CAM for one selected class.

    Returns:
        heatmap: normalized 2D float array in [0, 1]
        class_index: class explained
        class_scores: model output scores
        layer_name: feature layer used
    """
    grad_model = get_gradcam_model(model, layer_name)

    image_tensor = tf.convert_to_tensor(
        input_tensor, dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(
            image_tensor, training=False
        )

        if class_index is None:
            class_index = int(
                tf.argmax(predictions[0]).numpy()
            )

        if not 0 <= int(class_index) < len(CLASS_NAMES):
            raise ValueError(
                f"class_index must be between 0 and "
                f"{len(CLASS_NAMES) - 1}"
            )

        # Explain the selected class's model output.
        target_score = predictions[:, int(class_index)]

    gradients = tape.gradient(target_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. Check the selected layer "
            "and the model's computational graph."
        )

    # Standard Grad-CAM:
    # average gradients over height and width to obtain
    # one importance weight per feature-map channel.
    channel_weights = tf.reduce_mean(
        gradients, axis=(1, 2)
    )

    feature_maps = feature_maps[0]
    channel_weights = channel_weights[0]

    # Weighted sum of feature maps.
    cam = tf.reduce_sum(
        feature_maps * channel_weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )

    # Retain positive contributions and normalize safely.
    cam = tf.maximum(cam, 0.0)
    maximum = tf.reduce_max(cam)

    heatmap = tf.math.divide_no_nan(cam, maximum)
    heatmap = tf.where(
        tf.math.is_finite(heatmap), heatmap, 0.0
    )

    heatmap = heatmap.numpy().astype(np.float32)
    class_scores = predictions[0].numpy()

    return heatmap, int(class_index), class_scores, layer_name


def create_overlay(
    original_image,
    heatmap,
    alpha=0.40,
    colormap=cv2.COLORMAP_JET,
):
    """
    Resize and blend the heatmap with the original X-ray.
    alpha controls heatmap opacity.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    original = np.asarray(original_image)

    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("original_image must be an RGB image")

    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = np.squeeze(heatmap)

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")

    heatmap = np.nan_to_num(
        heatmap, nan=0.0, posinf=0.0, neginf=0.0
    )
    heatmap = np.clip(heatmap, 0.0, 1.0)

    height, width = original.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap, (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    resized_heatmap = np.clip(resized_heatmap, 0.0, 1.0)

    heatmap_uint8 = np.uint8(resized_heatmap * 255.0)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(
        colored_bgr, cv2.COLOR_BGR2RGB
    )

    original_uint8 = np.clip(original, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(
        original_uint8, 1.0 - alpha,
        colored_rgb, alpha, 0,
    )

    return overlay, colored_rgb


def explain_xray(
    model,
    image_path,
    output_dir="results/gradcam_v2",
    layer_name=DEFAULT_LAYER,
    alpha=0.40,
):
    """
    Generate the original image, heatmap, and overlay.
    Save outputs and print diagnostic information.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original, input_tensor = load_xray(image_path)

    heatmap, class_index, scores, used_layer = generate_gradcam(
        model,
        input_tensor,
        class_index=None,
        layer_name=layer_name,
    )

    overlay, colored_heatmap = create_overlay(
        original, heatmap, alpha=alpha
    )

    stem = image_path.stem

    Image.fromarray(original).save(
        output_dir / f"{stem}_original.png"
    )
    Image.fromarray(colored_heatmap).save(
        output_dir / f"{stem}_heatmap.png"
    )
    Image.fromarray(overlay).save(
        output_dir / f"{stem}_overlay.png"
    )

    print("\n--- Grad-CAM v2 diagnostics ---")
    print("Image:", image_path)
    print("Layer:", used_layer)
    print("Predicted class:", CLASS_NAMES[class_index])
    print("Selected class index:", class_index)
    print("Model output scores:", scores)
    print("Feature heatmap shape:", heatmap.shape)
    print("Heatmap min/max:", float(heatmap.min()),
          float(heatmap.max()))
    print("Heatmap standard deviation:",
          float(heatmap.std()))
    print("Output directory:", output_dir.resolve())

    if float(heatmap.std()) < 1e-4:
        print(
            "WARNING: The heatmap has very little variation. "
            "Inspect gradients and feature activations."
        )

    return heatmap, overlay



if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]

    MODEL_PATH = (
        BASE_DIR / "models" /
        "kaggle_chest_xray_covid4_best.keras"
    )

    IMAGE_PATH = Path(
        "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
        "pneumonia/00222ba2d6b03db1_pneumonia-383.jpg"
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"X-ray not found: {IMAGE_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH, compile=False
    )

    original, input_tensor = load_xray(IMAGE_PATH)

    output_dir = BASE_DIR / "results" / "gradcam_v2_layer_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer_name in ["conv2d_3", "activation_3"]:
        heatmap, class_index, scores, used_layer = generate_gradcam(
            model,
            input_tensor,
            class_index=1,  # Pneumonia, for a controlled comparison
            layer_name=layer_name,
        )

        overlay, colored_heatmap = create_overlay(
            original,
            heatmap,
            alpha=0.40,
        )

        Image.fromarray(colored_heatmap).save(
            output_dir / f"{layer_name}_heatmap.png"
        )
        Image.fromarray(overlay).save(
            output_dir / f"{layer_name}_overlay.png"
        )

        print(f"\nLayer: {used_layer}")
        print("Heatmap shape:", heatmap.shape)
        print("Heatmap min/max:", float(heatmap.min()),
              float(heatmap.max()))
        print("Heatmap standard deviation:", float(heatmap.std()))

    print("\nSaved comparison images to:", output_dir)
