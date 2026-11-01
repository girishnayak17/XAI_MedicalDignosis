
"""
SHAP image explanations for four-class chest X-ray models.

Supported models:
- Custom CNN
- ResNet50
- EfficientNetB0

The selected model's own prediction function is supplied
when creating the explainer.

The selected class index is provided by the application.

Class mapping:
    0 = Normal
    1 = Pneumonia
    2 = Tuberculosis
    3 = COVID-19

Input images:
    RGB format
    Shape: (224, 224, 3)
    Pixel range: 0-255

Note:
    Model-specific preprocessing must be handled inside
    the prediction function supplied by the application.
"""

import numpy as np
import shap


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 4

CLASS_NAMES = [
    "Normal",
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
]


# ============================================================
# PREDICTION FUNCTION VALIDATION
# ============================================================

def _validate_prediction_function(predict_fn):
    """
    Validate the prediction function supplied by the application.

    Expected input:
        (batch_size, 224, 224, 3)

    Expected output:
        (batch_size, 4)
    """

    if not callable(predict_fn):
        raise TypeError(
            "predict_fn must be a callable prediction function."
        )

    def validated_predict_fn(images):

        images = np.asarray(
            images,
            dtype=np.float32,
        )

        # Validate batch dimensions
        if images.ndim != 4:
            raise ValueError(
                "Prediction input must have four dimensions: "
                "(batch_size, 224, 224, 3). "
                f"Received: {images.shape}"
            )

        # Validate image dimensions
        if images.shape[1:] != IMAGE_SHAPE:
            raise ValueError(
                f"Expected image dimensions {IMAGE_SHAPE}, "
                f"received {images.shape[1:]}."
            )

        if len(images) == 0:
            raise ValueError(
                "Prediction input batch cannot be empty."
            )

        if not np.all(np.isfinite(images)):
            raise ValueError(
                "Prediction input contains NaN or infinite values."
            )

        # Get predictions from the selected model
        predictions = predict_fn(images)

        predictions = np.asarray(
            predictions,
            dtype=np.float32,
        )

        # Validate prediction dimensions
        expected_prediction_shape = (
            len(images),
            NUM_CLASSES,
        )

        if predictions.shape != expected_prediction_shape:
            raise ValueError(
                "The selected model must return four class scores "
                "for every image.\n"
                f"Expected: {expected_prediction_shape}\n"
                f"Received: {predictions.shape}"
            )

        if not np.all(np.isfinite(predictions)):
            raise ValueError(
                "Model predictions contain NaN or infinite values."
            )

        return predictions

    return validated_predict_fn


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_shap_explainer(predict_fn):
    """
    Create a partition-based SHAP image explainer.

    Parameters
    ----------
    predict_fn : callable
        Prediction function belonging to the selected model.

        Supported models:
            Custom CNN
            ResNet50
            EfficientNetB0

        Input:
            (batch_size, 224, 224, 3)

        Output:
            (batch_size, 4)

    Returns
    -------
    shap.Explainer
        Partition-based image explainer.
    """

    # Validate and wrap the selected model's prediction function
    validated_predict_fn = _validate_prediction_function(
        predict_fn
    )

    # Image masker
    masker = shap.maskers.Image(
        "blur(32,32)",
        shape=IMAGE_SHAPE,
    )

    # Create SHAP explainer
    explainer = shap.Explainer(
        validated_predict_fn,
        masker,
        algorithm="partition",
    )

    return explainer


# ============================================================
# EXPLAIN ONE CHEST X-RAY IMAGE
# ============================================================

def explain_image(
    explainer,
    image,
    class_index,
    max_evals=1000,
    batch_size=8,
):
    """
    Generate SHAP values for one chest X-ray image.

    Parameters
    ----------
    explainer : shap.Explainer
        Explainer created using create_shap_explainer().

    image : numpy.ndarray
        RGB image with shape (224, 224, 3).
        Pixel values should remain in the 0-255 range.

    class_index : int
        Selected class index:

        0 = Normal
        1 = Pneumonia
        2 = Tuberculosis
        3 = COVID-19

    max_evals : int
        Maximum number of model evaluations used by SHAP.

    batch_size : int
        Number of masked images evaluated in a batch.

    Returns
    -------
    explanation : shap.Explanation
        Complete SHAP explanation object.

    class_values : numpy.ndarray
        Selected class attribution map.

        Expected shape:
            (224, 224, 3)
    """

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    if explainer is None:
        raise ValueError(
            "A valid SHAP explainer must be provided."
        )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    # Validate image shape
    if image.shape != IMAGE_SHAPE:
        raise ValueError(
            f"Expected image shape {IMAGE_SHAPE}, "
            f"received {image.shape}."
        )

    # Validate pixel values
    if not np.all(np.isfinite(image)):
        raise ValueError(
            "Input image contains NaN or infinite values."
        )

    if np.min(image) < 0 or np.max(image) > 255:
        raise ValueError(
            "Expected RGB image pixel values in the 0-255 range."
        )

    # Validate selected class
    if not isinstance(class_index, (int, np.integer)):
        raise TypeError(
            "class_index must be an integer."
        )

    class_index = int(class_index)

    if not 0 <= class_index < NUM_CLASSES:
        raise ValueError(
            f"class_index must be between 0 and {NUM_CLASSES - 1}."
        )

    # Validate SHAP evaluation parameters
    if (
        not isinstance(max_evals, (int, np.integer))
        or isinstance(max_evals, (bool, np.bool_))
        or max_evals < 1
    ):
        raise ValueError(
            "max_evals must be a positive integer."
        )

    if (
        not isinstance(batch_size, (int, np.integer))
        or isinstance(batch_size, (bool, np.bool_))
        or batch_size < 1
    ):
        raise ValueError(
            "batch_size must be a positive integer."
        )

    max_evals = int(max_evals)
    batch_size = int(batch_size)

    # --------------------------------------------------------
    # GENERATE SHAP EXPLANATION
    # --------------------------------------------------------

    explanation = explainer(
        image[None, ...],
        max_evals=max_evals,
        batch_size=batch_size,
    )

    # --------------------------------------------------------
    # EXTRACT SHAP VALUES
    # --------------------------------------------------------

    values = np.asarray(
        explanation.values,
        dtype=np.float32,
    )

    # Expected multi-class output:
    #
    # (1, 224, 224, 3, 4)
    #
    # Dimension 0: Image batch
    # Dimension 1: Image height
    # Dimension 2: Image width
    # Dimension 3: RGB channels
    # Dimension 4: Class outputs

    expected_shape = (
        1,
        IMAGE_SHAPE[0],
        IMAGE_SHAPE[1],
        IMAGE_SHAPE[2],
        NUM_CLASSES,
    )

    if values.shape != expected_shape:
        raise ValueError(
            "Unexpected SHAP output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {values.shape}\n"
            "Check the installed SHAP version and the selected "
            "model's prediction output."
        )

    # Validate SHAP values
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "SHAP returned NaN or infinite attribution values."
        )

    # --------------------------------------------------------
    # EXTRACT SELECTED CLASS ATTRIBUTIONS
    # --------------------------------------------------------

    class_values = values[
        0,
        ...,
        class_index,
    ]

    # Validate selected class attribution
    if class_values.shape != IMAGE_SHAPE:
        raise ValueError(
            "Unexpected class-specific SHAP shape: "
            f"{class_values.shape}"
        )

    if not np.all(np.isfinite(class_values)):
        raise ValueError(
            "Selected class attribution contains invalid values."
        )

    return explanation, class_values


# ============================================================
# OPTIONAL: GET CLASS NAME
# ============================================================

def get_class_name(class_index):
    """
    Return the class name corresponding to a class index.
    """

    if not isinstance(class_index, (int, np.integer)):
        raise TypeError(
            "class_index must be an integer."
        )

    class_index = int(class_index)

    if not 0 <= class_index < NUM_CLASSES:
        raise ValueError(
            f"class_index must be between 0 and {NUM_CLASSES - 1}."
        )

    return CLASS_NAMES[class_index]
