
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE / "models/kaggle_chest_xray_covid4_best.keras"
IMAGE_PATH = Path(
    "/mnt/d/MCA_Final_Project/KDatasets_Clean/test/"
    "pneumonia/00222ba2d6b03db1_pneumonia-383.jpg"
)

OUT_DIR = BASE / "results/gradcam_v2_logits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Expose the feature map and the input to the final Dense layer.
feature_layer = model.get_layer("activation_3")
classifier = model.get_layer("four_class_classifier")

# Reuse the existing trained classifier weights, but calculate
# its linear output before softmax.
logit_layer = tf.keras.layers.Dense(
    classifier.units,
    activation=None,
    use_bias=classifier.use_bias,
    name="temporary_logits",
)
logit_layer.build((None, classifier.kernel.shape[0]))
logit_layer.set_weights(classifier.get_weights())

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        feature_layer.output,
        logit_layer(model.get_layer("dropout").output),
        model.output,
    ],
)

image = Image.open(IMAGE_PATH).convert("RGB").resize((224, 224))
x = np.expand_dims(
    np.asarray(image, dtype=np.float32), axis=0
)

with tf.GradientTape() as tape:
    features, logits, probabilities = grad_model(
        x, training=False
    )
    target = logits[:, 1]  # Pneumonia logit

gradients = tape.gradient(target, features)

if gradients is None:
    raise RuntimeError("Logit gradients are None.")

weights = tf.reduce_mean(gradients, axis=(1, 2))
cam = tf.reduce_sum(
    features[0] * weights[0][None, None, :],
    axis=-1,
)
cam = tf.maximum(cam, 0.0)
cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam))
cam = cam.numpy()

print("Probabilities:", probabilities.numpy()[0])
print("Pneumonia logit:", float(logits[0, 1]))
print("Gradient absolute mean:",
      float(tf.reduce_mean(tf.abs(gradients))))
print("Gradient absolute max:",
      float(tf.reduce_max(tf.abs(gradients))))
print("Heatmap min/max:", float(cam.min()), float(cam.max()))
print("Heatmap standard deviation:", float(cam.std()))

heatmap = cv2.resize(cam, (image.width, image.height))
colored = cv2.applyColorMap(
    np.uint8(np.clip(heatmap, 0, 1) * 255),
    cv2.COLORMAP_JET,
)
colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

original = np.asarray(image)
overlay = cv2.addWeighted(original, 0.6, colored, 0.4, 0)

Image.fromarray(colored).save(OUT_DIR / "logits_heatmap.png")
Image.fromarray(overlay).save(OUT_DIR / "logits_overlay.png")

print("Saved:", OUT_DIR)
