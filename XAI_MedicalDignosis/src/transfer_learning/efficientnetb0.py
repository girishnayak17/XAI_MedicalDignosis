
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

IMAGE_SIZE = 224
NUM_CLASSES = 4

CLASS_NAMES = [
    "normal",
    "pneumonia",
    "tuberculosis",
    "covid"
]


def build_efficientnetb0():
    """
    Build EfficientNetB0 using ImageNet pretrained weights
    for four-class chest X-ray classification.
    """

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="chest_xray_input"
    )

    # Load ImageNet pretrained EfficientNetB0.
    # Its internal preprocessing expects pixels in [0, 255].
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="efficientnetb0"
    )

    # Stage 1: Freeze pretrained backbone.
    base_model.trainable = False

    x = base_model(inputs, training=False)

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.BatchNormalization(
        name="classifier_batch_normalization"
    )(x)

    x = layers.Dropout(
        0.4,
        name="classifier_dropout"
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="four_class_classifier"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="EfficientNetB0_Transfer_Learning"
    )

    return model, base_model


def compile_efficientnetb0(model, learning_rate=1e-3):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
        jit_compile=False
    )

    return model


def load_efficientnetb0_checkpoint(model_path):

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )
