
import numpy as np
import shap


def create_shap_explainer(predict_fn):
    """
    Create a SHAP image explainer for a four-class CNN.
    """
    masker = shap.maskers.Image(
        "blur(32,32)",
        shape=(224, 224, 3),
    )

    return shap.Explainer(
        predict_fn,
        masker,
        algorithm="partition",
    )


def explain_image(
    explainer,
    image,
    class_index=None,
    max_evals=1000,
    batch_size=8,
):
    """
    Generate SHAP values for one chest X-ray image.

    Explains all model outputs. class_index selects the
    class for which the attribution map is returned.
    """
    image = np.asarray(image, dtype=np.float32)

    if image.shape != (224, 224, 3):
        raise ValueError(
            f"Expected image shape (224, 224, 3), got {image.shape}"
        )

    explanation = explainer(
        image[None, ...],
        max_evals=max_evals,
        batch_size=batch_size,
    )

    values = np.asarray(explanation.values)

    if not np.all(np.isfinite(values)):
        raise ValueError("SHAP returned invalid attribution values.")

    if values.ndim != 5 or values.shape[-1] != 4:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if class_index is None:
        class_index = int(np.argmax(
            explainer.model(image[None, ...])[0]
        ))

    if not 0 <= class_index < 4:
        raise ValueError("Invalid class index.")

    class_values = values[0, ..., class_index]

    return explanation, class_values
