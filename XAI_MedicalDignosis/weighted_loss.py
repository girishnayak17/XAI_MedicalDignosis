
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="XAI")
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy with per-label positive weights."""

    def __init__(
        self,
        positive_weights,
        name="weighted_binary_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.positive_weights = list(positive_weights)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred = tf.clip_by_value(
            y_pred,
            epsilon,
            1.0 - epsilon,
        )

        weights = tf.constant(
            self.positive_weights,
            dtype=tf.float32,
        )

        loss = -(
            weights * y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )

        return tf.reduce_mean(loss, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "positive_weights": self.positive_weights,
        })
        return config
