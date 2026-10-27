import tensorflow as tf
import numpy as np

print("=" * 60)
print("TENSORFLOW GPU TEST")
print("=" * 60)

print("TensorFlow version:", tf.__version__)

gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)

if not gpus:
    print("❌ GPU NOT DETECTED")
    exit()

print("\nGPU details:")

for gpu in gpus:
    print(gpu)

print("\nTesting GPU convolution...")

try:

    with tf.device("/GPU:0"):

        x = tf.random.normal(
            [16, 224, 224, 3]
        )

        conv = tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            activation="relu"
        )

        y = conv(x)

        result = y.numpy()

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)

    print("\n✅ GPU CONVOLUTION TEST PASSED")

except Exception as e:

    print("\n❌ GPU CONVOLUTION TEST FAILED")

    print(type(e).__name__)
    print(e)

print("=" * 60)