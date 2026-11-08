
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50


IMAGE_SIZE = 224
NUM_CLASSES = 4

CLASS_NAMES = [
    "normal",
    "pneumonia",
    "tuberculosis",
    "covid"
]


# ============================================================
# RESNET50 PREPROCESSING
# ============================================================

@tf.keras.utils.register_keras_serializable(
    package="TransferLearning"
)
def resnet50_preprocessing(inputs):

    return tf.keras.applications.resnet50.preprocess_input(
        inputs
    )


# ============================================================
# BUILD RESNET50 MODEL
# ============================================================

def build_resnet50():

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="chest_xray_input"
    )

    # ImageNet-pretrained ResNet50
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="resnet50"
    )

    # Freeze pretrained backbone initially
    base_model.trainable = False

    # ResNet50-specific preprocessing
    x = layers.Lambda(
        resnet50_preprocessing,
        name="resnet50_preprocessing"
    )(inputs)

    # Feature extraction
    x = base_model(
        x,
        training=False
    )

    # Classification head
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
        name="ResNet50_Transfer_Learning"
    )

    return model, base_model


# ============================================================
# COMPILE MODEL
# ============================================================

def compile_resnet50(
    model,
    learning_rate=1e-3
):

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


# ============================================================
# LOAD SAVED RESNET50 CHECKPOINT
# ============================================================

def load_resnet50_checkpoint(model_path):

    return tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={
            "resnet50_preprocessing": resnet50_preprocessing,
            "TransferLearning>resnet50_preprocessing":
                resnet50_preprocessing
        }
    )
